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

random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

print("GATE001_START", json.dumps({
    "model": MODEL,
    "seq": SEQ,
    "n_cal": N_CAL,
    "n_eval": N_EVAL,
    "seed": SEED,
    "clip_quantile": CLIP_Q,
    "device": DEVICE,
}), flush=True)

# Real pretrained RMSNorm Transformer.  This is a mechanism-screening pilot;
# it does not claim a BitNet-weight result.
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.to(DEVICE)
model.eval()

layers = model.model.layers
print("MODEL_INFO", json.dumps({
    "layers": len(layers),
    "hidden_size": model.config.hidden_size,
    "rms_norm_eps": getattr(model.config, "rms_norm_eps", None),
}), flush=True)

# Deterministic contiguous WikiText-2 evaluation stream.
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(x for x in ds["text"] if x and not x.isspace())
ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
needed = (N_CAL + N_EVAL) * SEQ
if ids.numel() < needed:
    raise RuntimeError(f"Need {needed} tokens, got {ids.numel()}")
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


# Calibrate ONE fixed angular grid per layer.  No token may choose a new
# angular scale at evaluation time.  Also bank the typical radius for a
# negative-control condition.
abs_u_samples = [[] for _ in range(len(layers))]
r_samples = [[] for _ in range(len(layers))]
handles = []
for li, layer in enumerate(layers):
    def make_cal_hook(i):
        def hook(_module, _inp, out):
            h = hidden_from_output(out).detach()
            r = token_rms(h)
            uabs = (h.float() / r).abs().flatten()
            n = min(4096, uabs.numel())
            idx = torch.linspace(0, uabs.numel() - 1, n).long()
            abs_u_samples[i].append(uabs[idx].cpu())
            r_samples[i].append(r.flatten().cpu())
            return out
        return hook
    handles.append(layer.register_forward_hook(make_cal_hook(li)))

with torch.inference_mode():
    for s in cal_segments:
        model(s, use_cache=False)
for h in handles:
    h.remove()

clips = []
median_r = []
for us, rs in zip(abs_u_samples, r_samples):
    uv = torch.cat(us)
    rv = torch.cat(rs)
    clips.append(max(float(torch.quantile(uv, CLIP_Q)), 1.0))
    median_r.append(float(torch.median(rv)))

print("CALIBRATION", json.dumps({
    "clip_min": min(clips),
    "clip_median": float(torch.tensor(clips).median()),
    "clip_max": max(clips),
    "radius_median_min": min(median_r),
    "radius_median_max": max(median_r),
}), flush=True)


def dynamic_quant(h, bits):
    """Strong conventional per-token absmax symmetric fake quantizer."""
    hf = h.float()
    if bits == 1:
        # Binary sign with per-token mean-absolute reconstruction scale.
        a = hf.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        return (torch.where(hf >= 0, a, -a)).to(h.dtype)
    qmax = (2 ** (bits - 1)) - 1
    a = hf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = a / qmax
    return (torch.round(hf / scale).clamp(-qmax, qmax) * scale).to(h.dtype)


def angular_code(h, bits, clip, radius_mode="true", fixed_radius=None):
    """Fixed-grid direction code plus an explicitly separate token radius."""
    hf = h.float()
    r = token_rms(h)
    u = hf / r

    if bits == 1:
        q = torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
    else:
        qmax = (2 ** (bits - 1)) - 1
        step = float(clip) / qmax
        q = torch.round(torch.clamp(u, -float(clip), float(clip)) / step) * step

    # Re-project the coded direction to unit RMS.  Hence all amplitude is
    # carried by the explicitly separate radial state.
    uhat = q / token_rms(q)

    if radius_mode == "true":
        rr = r
    elif radius_mode == "shift1":
        # Causal negative control: retain the radius distribution but attach
        # each token's radius to the next token's direction.
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
                    elif mode == "angular":
                        hh = angular_code(
                            h, bits, clips[i], radius_mode=radius_mode,
                            fixed_radius=median_r[i]
                        )
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
    result = {
        "name": name,
        "nll": nll,
        "ppl": math.exp(nll) if nll < 50 else float("inf"),
        "seconds": time.time() - t0,
    }
    print("RESULT", json.dumps(result), flush=True)
    return result


results = [
    evaluate("native"),
    evaluate("dynamic_A4", "dynamic", 4),
    evaluate("dynamic_A2", "dynamic", 2),
    evaluate("angular_A4_radius_true", "angular", 4),
    evaluate("angular_A3_radius_true", "angular", 3),
    evaluate("angular_A2_radius_true", "angular", 2),
    evaluate("angular_A1_radius_true", "angular", 1),
    evaluate("angular_A2_radius_shift1", "angular", 2, "shift1"),
    evaluate("angular_A2_radius_constant", "angular", 2, "constant"),
]

native = results[0]["nll"]
dyn4 = results[1]["nll"]
for r in results:
    r["delta_nll_vs_native"] = r["nll"] - native
    r["delta_nll_vs_dynamic_A4"] = r["nll"] - dyn4

by_name = {r["name"]: r for r in results}
a2 = by_name["angular_A2_radius_true"]
# Predeclared screening rule: A2 passes only if it is no worse than dynamic A4
# in NLL on this fixed pilot stream.  This is intentionally stringent.
gate_pass = a2["nll"] <= by_name["dynamic_A4"]["nll"]

final = {
    "experiment": "BITNET_PERSISTENT_RADIAL_ANGULAR_001_GATE001A",
    "scope": "activation-geometry mechanism screen on pretrained RMSNorm LLM; not yet a BitNet-weight confirmation",
    "config": {
        "model": MODEL,
        "seq": SEQ,
        "n_cal": N_CAL,
        "n_eval": N_EVAL,
        "eval_prediction_tokens": N_EVAL * (SEQ - 1),
        "seed": SEED,
        "clip_q": CLIP_Q,
        "radius_true_precision": "float32 fake-quant upper bound",
        "angular_scale": "one frozen clip/step per layer; no per-token angular scale",
    },
    "gate_rule": "angular_A2_radius_true NLL <= dynamic_A4 NLL",
    "gate_pass": gate_pass,
    "results": results,
}
print("FINAL_JSON", json.dumps(final, sort_keys=True), flush=True)
