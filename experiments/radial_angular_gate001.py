import json
import math
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 260816
MODEL = "HuggingFaceTB/SmolLM2-135M"
SEQ = 128
N_CAL = 2
N_EVAL = 4
CLIP_Q = 0.995
DEVICE = "cpu"
BLOCK = 64

random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

print("GATE001B_START", json.dumps({
    "model": MODEL, "seq": SEQ, "n_cal": N_CAL, "n_eval": N_EVAL,
    "seed": SEED, "clip_quantile": CLIP_Q, "device": DEVICE,
    "hadamard_block": BLOCK,
}), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.to(DEVICE)
model.eval()
layers = model.model.layers
D = model.config.hidden_size
if D % BLOCK != 0:
    raise RuntimeError(f"hidden size {D} not divisible by block {BLOCK}")
print("MODEL_INFO", json.dumps({
    "layers": len(layers), "hidden_size": D,
    "rms_norm_eps": getattr(model.config, "rms_norm_eps", None),
    "hadamard_blocks": D // BLOCK,
}), flush=True)

# Deterministic contiguous WikiText-2 stream.
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(x for x in ds["text"] if x and not x.isspace())
ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
segments = [ids[i * SEQ:(i + 1) * SEQ].unsqueeze(0).to(DEVICE)
            for i in range(N_CAL + N_EVAL)]
cal_segments = segments[:N_CAL]
eval_segments = segments[N_CAL:]


def hidden_from_output(out):
    return out[0] if isinstance(out, tuple) else out


def replace_hidden(out, h):
    return (h,) + out[1:] if isinstance(out, tuple) else h


def token_rms(x):
    xf = x.float()
    return torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True).clamp_min(1e-12))


def hadamard_matrix(n):
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


H = hadamard_matrix(BLOCK).to(DEVICE)
# H is symmetric/orthogonal for Sylvester construction, so the same operation
# is its inverse.  The block transform is exactly norm preserving.
def rotate_blocks(x):
    shp = x.shape
    y = x.float().reshape(*shp[:-1], D // BLOCK, BLOCK)
    y = torch.matmul(y, H.T)
    return y.reshape(shp)


def inverse_rotate_blocks(x):
    shp = x.shape
    y = x.float().reshape(*shp[:-1], D // BLOCK, BLOCK)
    y = torch.matmul(y, H)
    return y.reshape(shp)


# Calibrate fixed grids for raw direction and Hadamard-mixed direction.
raw_samples = [[] for _ in range(len(layers))]
rot_samples = [[] for _ in range(len(layers))]
r_samples = [[] for _ in range(len(layers))]
handles = []
for li, layer in enumerate(layers):
    def make_cal_hook(i):
        def hook(_module, _inp, out):
            h = hidden_from_output(out).detach()
            r = token_rms(h)
            u = h.float() / r
            ur = rotate_blocks(u)
            for src, bank in ((u.abs().flatten(), raw_samples),
                              (ur.abs().flatten(), rot_samples)):
                n = min(4096, src.numel())
                idx = torch.linspace(0, src.numel() - 1, n).long()
                bank[i].append(src[idx].cpu())
            r_samples[i].append(r.flatten().cpu())
            return out
        return hook
    handles.append(layer.register_forward_hook(make_cal_hook(li)))

with torch.inference_mode():
    for s in cal_segments:
        model(s, use_cache=False)
for h in handles:
    h.remove()

raw_clips, rot_clips, median_r = [], [], []
for us, urs, rs in zip(raw_samples, rot_samples, r_samples):
    raw_clips.append(max(float(torch.quantile(torch.cat(us), CLIP_Q)), 1.0))
    rot_clips.append(max(float(torch.quantile(torch.cat(urs), CLIP_Q)), 1.0))
    median_r.append(float(torch.median(torch.cat(rs))))
print("CALIBRATION", json.dumps({
    "raw_clip_median": float(torch.tensor(raw_clips).median()),
    "rot_clip_median": float(torch.tensor(rot_clips).median()),
    "raw_clip_max": max(raw_clips), "rot_clip_max": max(rot_clips),
    "radius_median_min": min(median_r), "radius_median_max": max(median_r),
}), flush=True)


def dynamic_quant(h, bits=4):
    hf = h.float()
    qmax = (2 ** (bits - 1)) - 1
    a = hf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = a / qmax
    return (torch.round(hf / scale).clamp(-qmax, qmax) * scale).to(h.dtype)


def fixed_grid(z, bits, clip):
    if bits == 1:
        return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    qmax = (2 ** (bits - 1)) - 1
    step = float(clip) / qmax
    return torch.round(torch.clamp(z, -float(clip), float(clip)) / step) * step


def angular_code(h, bits, clip, rotated=False, radius_mode="true", fixed_radius=None):
    hf = h.float()
    r = token_rms(h)
    u = hf / r
    z = rotate_blocks(u) if rotated else u
    qz = fixed_grid(z, bits, clip)
    q = inverse_rotate_blocks(qz) if rotated else qz
    uhat = q / token_rms(q)
    if radius_mode == "true":
        rr = r
    elif radius_mode == "shift1":
        rr = torch.roll(r, shifts=1, dims=-2)
    elif radius_mode == "constant":
        rr = torch.full_like(r, float(fixed_radius))
    else:
        raise ValueError(radius_mode)
    return (rr * uhat).to(h.dtype)


def evaluate(name, mode=None, bits=None, radius_mode="true"):
    hs = []
    if mode is not None:
        for li, layer in enumerate(layers):
            def make_hook(i):
                def hook(_module, _inp, out):
                    h = hidden_from_output(out)
                    if mode == "dynamic":
                        hh = dynamic_quant(h, bits)
                    elif mode == "angular_raw":
                        hh = angular_code(h, bits, raw_clips[i], False, radius_mode, median_r[i])
                    elif mode == "angular_h":
                        hh = angular_code(h, bits, rot_clips[i], True, radius_mode, median_r[i])
                    else:
                        raise ValueError(mode)
                    return replace_hidden(out, hh)
                return hook
            hs.append(layer.register_forward_hook(make_hook(li)))
    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()
    with torch.inference_mode():
        for s in eval_segments:
            out = model(s, labels=s, use_cache=False)
            n = s.shape[1] - 1
            total_nll += float(out.loss.item()) * n
            total_tokens += n
    for h in hs:
        h.remove()
    nll = total_nll / total_tokens
    result = {"name": name, "nll": nll,
              "ppl": math.exp(nll) if nll < 50 else float("inf"),
              "seconds": time.time() - t0}
    print("RESULT", json.dumps(result), flush=True)
    return result


results = [
    evaluate("native"),
    evaluate("dynamic_A4", "dynamic", 4),
    evaluate("angular_raw_A2_radius_true", "angular_raw", 2),
]
for b in [4, 3, 2, 1]:
    results.append(evaluate(f"angularH_A{b}_radius_true", "angular_h", b))
results += [
    evaluate("angularH_A2_radius_shift1", "angular_h", 2, "shift1"),
    evaluate("angularH_A2_radius_constant", "angular_h", 2, "constant"),
]

native = results[0]["nll"]
dyn4 = results[1]["nll"]
for r in results:
    r["delta_nll_vs_native"] = r["nll"] - native
    r["delta_nll_vs_dynamic_A4"] = r["nll"] - dyn4
by_name = {r["name"]: r for r in results}
ha2 = by_name["angularH_A2_radius_true"]
gate_pass = ha2["nll"] <= by_name["dynamic_A4"]["nll"]
final = {
    "experiment": "BITNET_PERSISTENT_RADIAL_ANGULAR_001_GATE001B",
    "scope": "Hadamard-mixed activation-geometry mechanism screen on pretrained RMSNorm LLM; not yet BitNet-weight confirmation",
    "config": {
        "model": MODEL, "seq": SEQ, "n_cal": N_CAL, "n_eval": N_EVAL,
        "eval_prediction_tokens": N_EVAL * (SEQ - 1), "seed": SEED,
        "clip_q": CLIP_Q, "hadamard_block": BLOCK,
        "radius_true_precision": "float32 fake-quant upper bound",
        "angular_scale": "one frozen grid per layer after blockwise orthonormal Hadamard; no per-token angular scale",
    },
    "gate_rule": "angularH_A2_radius_true NLL <= dynamic_A4 NLL",
    "gate_pass": gate_pass,
    "results": results,
}
print("FINAL_JSON", json.dumps(final, sort_keys=True), flush=True)
