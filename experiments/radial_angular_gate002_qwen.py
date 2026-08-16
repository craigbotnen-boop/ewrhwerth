import json
import math
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 26081602
MODEL = "Qwen/Qwen2.5-0.5B"
SEQ = 256
N_CAL = 4
N_EVAL = 16
CLIP_Q = 0.995
BLOCK = 64
RADIUS_MARGIN_NLL = 0.25

random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

print("GATE002_START", json.dumps({
    "model": MODEL,
    "seed": SEED,
    "seq": SEQ,
    "n_cal": N_CAL,
    "n_eval": N_EVAL,
    "eval_prediction_tokens": N_EVAL * (SEQ - 1),
    "matched_hadamard_control": True,
    "radius_margin_nll": RADIUS_MARGIN_NLL,
}), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
layers = model.model.layers
D = model.config.hidden_size
if D % BLOCK:
    raise RuntimeError(f"hidden size {D} not divisible by Hadamard block {BLOCK}")
print("MODEL_INFO", json.dumps({
    "layers": len(layers),
    "hidden_size": D,
    "rms_norm_eps": getattr(model.config, "rms_norm_eps", None),
    "hadamard_blocks": D // BLOCK,
}), flush=True)

# Frozen WikiText-2 test stream, assembled line-by-line to avoid asking the
# tokenizer to process a sequence beyond the model's configured context.
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
needed = (N_CAL + N_EVAL) * SEQ
parts = []
count = 0
for text in ds["text"]:
    if not text or text.isspace():
        continue
    t = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if t.numel() == 0:
        continue
    parts.append(t)
    count += t.numel()
    if count >= needed:
        break
ids = torch.cat(parts)[:needed]
if ids.numel() < needed:
    raise RuntimeError(f"Need {needed} frozen tokens, got {ids.numel()}")
segments = [ids[i * SEQ:(i + 1) * SEQ].unsqueeze(0) for i in range(N_CAL + N_EVAL)]
cal_segments = segments[:N_CAL]
eval_segments = segments[N_CAL:]


def hidden(o):
    return o[0] if isinstance(o, tuple) else o


def repl(o, h):
    return (h,) + o[1:] if isinstance(o, tuple) else h


def rms(x):
    xf = x.float()
    return torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True).clamp_min(1e-12))


def had(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


H = had(BLOCK)


def rot(x):
    shp = x.shape
    xb = x.float().reshape(*shp[:-1], D // BLOCK, BLOCK)
    return torch.matmul(xb, H.T).reshape(shp)


def irot(x):
    shp = x.shape
    xb = x.float().reshape(*shp[:-1], D // BLOCK, BLOCK)
    return torch.matmul(xb, H).reshape(shp)


# Independent calibration bank: one fixed angular grid and one fixed log-RMS
# codebook range per layer. Evaluation tokens never choose a new angular scale.
zs = [[] for _ in layers]
lrs = [[] for _ in layers]
handles = []
for li, layer in enumerate(layers):
    def mk(i):
        def hook(_m, _inp, o):
            h = hidden(o).detach()
            rr = rms(h)
            z = rot(h.float() / rr).abs().flatten()
            n = min(8192, z.numel())
            idx = torch.linspace(0, z.numel() - 1, n).long()
            zs[i].append(z[idx].cpu())
            lrs[i].append(torch.log(rr.flatten()).cpu())
            return o
        return hook
    handles.append(layer.register_forward_hook(mk(li)))

with torch.inference_mode():
    for s in cal_segments:
        model(s, use_cache=False)
for x in handles:
    x.remove()

clips, lo, hi, med = [], [], [], []
for zbank, lbank in zip(zs, lrs):
    z = torch.cat(zbank)
    lr = torch.cat(lbank)
    clips.append(max(float(torch.quantile(z, CLIP_Q)), 1.0))
    lo.append(float(torch.quantile(lr, 0.005)))
    hi.append(float(torch.quantile(lr, 0.995)))
    med.append(float(torch.exp(torch.median(lr))))

print("CAL", json.dumps({
    "clip_min": min(clips),
    "clip_median": float(torch.tensor(clips).median()),
    "clip_max": max(clips),
    "logr_width_median": float((torch.tensor(hi) - torch.tensor(lo)).median()),
}), flush=True)


def sym_a4(z):
    zf = z.float()
    qmax = 7
    a = zf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    sc = a / qmax
    return torch.round(zf / sc).clamp(-qmax, qmax) * sc


def h_dyn4(h):
    return irot(sym_a4(rot(h))).to(h.dtype)


def qdir4(z, c):
    step = float(c) / 7
    return torch.round(torch.clamp(z, -float(c), float(c)) / step) * step


def qlog4(rr, l, h):
    lr = torch.log(rr.float()).clamp(float(l), float(h))
    levels = 15
    step = (float(h) - float(l)) / levels
    return torch.exp(torch.round((lr - float(l)) / step) * step + float(l))


def angularH4(h, i, rmode):
    rr = rms(h)
    u = h.float() / rr
    uh = irot(qdir4(rot(u), clips[i]))
    uh = uh / rms(uh)
    if rmode == "fp32":
        rq = rr
    elif rmode == "log4":
        rq = qlog4(rr, lo[i], hi[i])
    elif rmode == "shift1":
        rq = torch.roll(rr, 1, dims=-2)
    elif rmode == "constant":
        rq = torch.full_like(rr, med[i])
    else:
        raise ValueError(rmode)
    return (rq * uh).to(h.dtype)


def evaluate(name, mode=None, rmode="fp32"):
    hs = []
    if mode:
        for li, layer in enumerate(layers):
            def mk(i):
                def hook(_m, _inp, o):
                    h = hidden(o)
                    if mode == "hmax":
                        hh = h_dyn4(h)
                    elif mode == "angular":
                        hh = angularH4(h, i, rmode)
                    else:
                        raise ValueError(mode)
                    return repl(o, hh)
                return hook
            hs.append(layer.register_forward_hook(mk(li)))

    total = 0.0
    nt = 0
    t0 = time.time()
    with torch.inference_mode():
        for s in eval_segments:
            o = model(s, labels=s, use_cache=False)
            n = s.shape[1] - 1
            total += float(o.loss) * n
            nt += n
    for x in hs:
        x.remove()
    nll = total / nt
    r = {
        "name": name,
        "nll": nll,
        "ppl": math.exp(nll) if nll < 50 else float("inf"),
        "seconds": time.time() - t0,
    }
    print("RESULT", json.dumps(r), flush=True)
    return r


results = [
    evaluate("native"),
    evaluate("hadamard_dynamic_A4_absmax", "hmax"),
    evaluate("angularH_A4_radius_fp32", "angular", "fp32"),
    evaluate("angularH_A4_radius_log4", "angular", "log4"),
    evaluate("angularH_A4_radius_shift1", "angular", "shift1"),
    evaluate("angularH_A4_radius_constant", "angular", "constant"),
]

native = results[0]["nll"]
by = {x["name"]: x for x in results}
for x in results:
    x["delta_nll_vs_native"] = x["nll"] - native

ang = by["angularH_A4_radius_log4"]["nll"]
hmax = by["hadamard_dynamic_A4_absmax"]["nll"]
shift = by["angularH_A4_radius_shift1"]["nll"]
const = by["angularH_A4_radius_constant"]["nll"]
replicate_pass = ang < hmax
radial_specificity_pass = ang + RADIUS_MARGIN_NLL <= min(shift, const)
gate_pass = replicate_pass and radial_specificity_pass

final = {
    "experiment": "BITNET_PERSISTENT_RADIAL_ANGULAR_001_GATE002_QWEN_REPLICATION",
    "scope": "cross-model RMSNorm replication with 4080 frozen prediction tokens; not BitNet-weight confirmation",
    "preregistered_rules": {
        "replication": "angularH A4 + 4-bit log-radius NLL < matched Hadamard per-token absmax A4 NLL",
        "radial_specificity": f"angularH A4 + 4-bit log-radius NLL + {RADIUS_MARGIN_NLL} <= min(shifted-radius NLL, constant-radius NLL)",
        "overall": "both replication and radial-specificity rules pass",
    },
    "replicate_pass": replicate_pass,
    "radial_specificity_pass": radial_specificity_pass,
    "gate_pass": gate_pass,
    "config": {
        "model": MODEL,
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "seq": SEQ,
        "n_cal": N_CAL,
        "n_eval": N_EVAL,
        "calibration_tokens": N_CAL * SEQ,
        "eval_prediction_tokens": N_EVAL * (SEQ - 1),
        "seed": SEED,
        "hadamard_block": BLOCK,
        "clip_q": CLIP_Q,
    },
    "effects": {
        "delta_nll_log4_vs_matched_hadamard_A4": ang - hmax,
        "delta_nll_log4_vs_shifted_radius": ang - shift,
        "delta_nll_log4_vs_constant_radius": ang - const,
    },
    "results": results,
}
print("FINAL_JSON", json.dumps(final, sort_keys=True), flush=True)
