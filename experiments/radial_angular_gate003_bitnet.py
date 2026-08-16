import json
import math
import os
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 26081603
MODEL = "microsoft/bitnet-b1.58-2B-4T"
REVISION = "main"
SEQ = 256
N_CAL = 4
N_EVAL = 8
CLIP_Q = 0.995
BLOCK = 64
RADIUS_MARGIN = 0.25

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print("GATE003_START", json.dumps({
    "model": MODEL,
    "revision": REVISION,
    "seed": SEED,
    "seq": SEQ,
    "n_cal": N_CAL,
    "n_eval": N_EVAL,
    "eval_prediction_tokens": N_EVAL * (SEQ - 1),
    "accelerator": os.environ.get("ACCELERATOR"),
    "job_id": os.environ.get("JOB_ID"),
    "scope": "native BitNet weight checkpoint; residual-stream activation geometry screen",
}), flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("Gate 003 requires CUDA")
device = torch.device("cuda")
print("CUDA", json.dumps({"name": torch.cuda.get_device_name(0), "torch": torch.__version__}), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    revision=REVISION,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
)
model.eval()

# Find the decoder layer list robustly across native/custom BitNet implementations.
layers = None
for cand in [
    getattr(getattr(model, "model", None), "layers", None),
    getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
    getattr(model, "layers", None),
]:
    if cand is not None and len(cand) == int(model.config.num_hidden_layers):
        layers = cand
        break
if layers is None:
    for _name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == int(model.config.num_hidden_layers):
            layers = mod
            break
if layers is None:
    raise RuntimeError("Could not locate decoder layer ModuleList")
D = int(model.config.hidden_size)
if D % BLOCK:
    raise RuntimeError(f"hidden size {D} is not divisible by Hadamard block {BLOCK}")
print("MODEL_INFO", json.dumps({
    "layers": len(layers),
    "hidden_size": D,
    "rms_norm_eps": float(getattr(model.config, "rms_norm_eps", float("nan"))),
    "quantization_config": getattr(model.config, "quantization_config", None),
}), flush=True)

# Freeze the evaluation stream deterministically.
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(x for x in ds["text"] if x and not x.isspace())
ids = tok(text, return_tensors="pt", add_special_tokens=False, truncation=False)["input_ids"][0]
segments = [ids[i * SEQ:(i + 1) * SEQ].unsqueeze(0).to(device) for i in range(N_CAL + N_EVAL)]
if any(s.shape[1] != SEQ for s in segments):
    raise RuntimeError("Not enough frozen tokens")
cal_segments, eval_segments = segments[:N_CAL], segments[N_CAL:]

def hidden(o):
    if torch.is_tensor(o):
        return o
    if isinstance(o, tuple):
        return o[0]
    if isinstance(o, list):
        return o[0]
    raise TypeError(f"Unsupported layer output type: {type(o)}")

def repl(o, h):
    if torch.is_tensor(o):
        return h
    if isinstance(o, tuple):
        return (h,) + o[1:]
    if isinstance(o, list):
        return [h] + list(o[1:])
    raise TypeError(f"Unsupported layer output type: {type(o)}")

def rms(x):
    xf = x.float()
    return torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True).clamp_min(1e-12))

def had(n):
    H = torch.ones(1, 1, dtype=torch.float32, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)

H = had(BLOCK)

def rot(x):
    shp = x.shape
    return torch.matmul(x.float().reshape(*shp[:-1], D // BLOCK, BLOCK), H.T).reshape(shp)

def irot(x):
    shp = x.shape
    return torch.matmul(x.float().reshape(*shp[:-1], D // BLOCK, BLOCK), H).reshape(shp)

# Calibrate only the fixed angular grid and constant-radius controls on native trajectories.
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
            idx = torch.linspace(0, z.numel() - 1, n, device=z.device).long()
            zs[i].append(z[idx].cpu())
            lrs[i].append(torch.log(rr.flatten()).cpu())
            return None
        return hook
    handles.append(layer.register_forward_hook(mk(li)))
with torch.inference_mode():
    for s in cal_segments:
        model(s, use_cache=False)
for hdl in handles:
    hdl.remove()

clips, med = [], []
for zbank, lbank in zip(zs, lrs):
    z = torch.cat(zbank)
    lr = torch.cat(lbank)
    clips.append(max(float(torch.quantile(z, CLIP_Q)), 1.0))
    med.append(float(torch.exp(torch.median(lr))))
print("CAL", json.dumps({
    "calibration_tokens": N_CAL * SEQ,
    "clip_min": min(clips),
    "clip_median": float(torch.tensor(clips).median()),
    "clip_max": max(clips),
    "radius_median_min": min(med),
    "radius_median_max": max(med),
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
    step = float(c) / 7.0
    return torch.round(torch.clamp(z, -float(c), float(c)) / step) * step

def angularH4(h, i, rmode):
    rr = rms(h)
    u = h.float() / rr
    uh = irot(qdir4(rot(u), clips[i]))
    uh = uh / rms(uh)
    if rmode == "true":
        rq = rr
    elif rmode == "shift1":
        rq = torch.roll(rr, 1, dims=-2)
    elif rmode == "constant":
        rq = torch.full_like(rr, med[i])
    else:
        raise ValueError(rmode)
    return (rq * uh).to(h.dtype)

def evaluate(name, mode=None, rmode="true"):
    hs = []
    if mode is not None:
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
    for hdl in hs:
        hdl.remove()
    nll = total / nt
    out = {"name": name, "nll": nll, "ppl": math.exp(nll), "seconds": time.time() - t0}
    print("RESULT", json.dumps(out), flush=True)
    return out

results = [
    evaluate("native"),
    evaluate("hadamard_dynamic_A4_absmax", "hmax"),
    evaluate("angularH_A4_radius_true", "angular", "true"),
    evaluate("angularH_A4_radius_shift1", "angular", "shift1"),
    evaluate("angularH_A4_radius_constant", "angular", "constant"),
]
by = {x["name"]: x for x in results}
base = by["hadamard_dynamic_A4_absmax"]["nll"]
true = by["angularH_A4_radius_true"]["nll"]
shift = by["angularH_A4_radius_shift1"]["nll"]
const = by["angularH_A4_radius_constant"]["nll"]
replicate_pass = true < base
radial_specificity_pass = true + RADIUS_MARGIN <= min(shift, const)
gate_pass = replicate_pass and radial_specificity_pass
final = {
    "experiment": "BITNET_PERSISTENT_RADIAL_ANGULAR_001_GATE003_NATIVE_BITNET",
    "scope": "official native BitNet b1.58 2B-4T weight checkpoint; post-layer residual-stream A4 geometry screen, not a replacement of BitLinear internal activation quantization",
    "config": {
        "model": MODEL,
        "revision": REVISION,
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "seq": SEQ,
        "n_cal": N_CAL,
        "n_eval": N_EVAL,
        "calibration_tokens": N_CAL * SEQ,
        "eval_prediction_tokens": N_EVAL * (SEQ - 1),
        "hadamard_block": BLOCK,
        "clip_q": CLIP_Q,
        "seed": SEED,
    },
    "preregistered_rules": {
        "replication": "angularH A4 + true token radius NLL < matched Hadamard per-token absmax A4 NLL",
        "radial_specificity": "true-radius NLL + 0.25 <= min(shifted-radius NLL, constant-radius NLL)",
        "overall": "both rules pass",
    },
    "replicate_pass": replicate_pass,
    "radial_specificity_pass": radial_specificity_pass,
    "gate_pass": gate_pass,
    "effects": {
        "delta_nll_true_vs_matched_hadamard_A4": true - base,
        "delta_nll_true_vs_shifted_radius": true - shift,
        "delta_nll_true_vs_constant_radius": true - const,
    },
    "results": results,
}
print("FINAL_JSON", json.dumps(final, sort_keys=True), flush=True)
