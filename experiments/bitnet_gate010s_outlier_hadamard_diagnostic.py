import json, math, time, hashlib, types, statistics
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.bitnet import AutoBitLinear

SEED = 26081910
MODEL = "microsoft/bitnet-b1.58-2B-4T"
SEQ = 128
N_CAL = 8
SAMPLED_TOKEN_ROWS = 32
PREFERRED_HADAMARD_BLOCK = 256
EXPECTED_CAL_HASH = "b025c5b984588d46b3c5e6d64144ad90a487073be8357b50d2814abac12ac908"

# Frozen diagnostic thresholds. This gate localizes a mechanism; it does not claim
# that a post-training Hadamard retrofit preserves end-to-end model quality.
MIN_PATHOLOGICAL_CONTRAST = 1.25
MIN_HADAMARD_NMSE_REDUCTION = 0.20


torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def segments_from_ids(ids, n, offset=0):
    out = []
    for i in range(offset, offset + n):
        s = ids[i * SEQ:(i + 1) * SEQ]
        if s.numel() != SEQ:
            raise RuntimeError("insufficient tokens")
        out.append(s.unsqueeze(0))
    return out


def ids_hash(segments):
    x = torch.cat(segments, dim=0).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(x).hexdigest()


def choose_hadamard_block(d):
    for b in (PREFERRED_HADAMARD_BLOCK, 128, 64, 32, 16, 8, 4, 2):
        if d % b == 0:
            return b
    raise RuntimeError(f"no supported Hadamard block divides width {d}")


def fwht_blocks(x, block):
    if x.shape[-1] % block != 0 or (block & (block - 1)):
        raise ValueError("invalid Hadamard block")
    shape = x.shape
    y = x.reshape(-1, shape[-1] // block, block)
    h = 1
    while h < block:
        prefix = y.shape[:-1]
        z = y.reshape(*prefix, block // (2 * h), 2 * h)
        a = z[..., :h]
        b = z[..., h:]
        y = torch.cat((a + b, a - b), dim=-1).reshape(*prefix, block)
        h *= 2
    return y.reshape(shape) / math.sqrt(block)


def sample_rows(x, n=SAMPLED_TOKEN_ROWS):
    d = x.shape[-1]
    flat = x.detach().reshape(-1, d)
    nr = flat.shape[0]
    take = min(n, nr)
    if take == nr:
        idx = torch.arange(nr, device=flat.device)
    else:
        idx = torch.linspace(0, nr - 1, take, device=flat.device).long()
    return flat.index_select(0, idx).float()


def distribution_metrics(x):
    absx = x.abs()
    mean_abs = absx.mean(dim=-1).clamp_min(1e-12)
    max_abs = absx.amax(dim=-1)
    ratio = max_abs / mean_abs
    second = x.square().mean().clamp_min(1e-20)
    fourth = x.pow(4).mean()
    kurt = fourth / second.square()
    flat_abs = absx.flatten()
    q50 = torch.quantile(flat_abs, 0.50).clamp_min(1e-12)
    q99 = torch.quantile(flat_abs, 0.99)
    q999 = torch.quantile(flat_abs, 0.999)
    return {
        "max_over_meanabs_median": float(torch.quantile(ratio, 0.50)),
        "max_over_meanabs_p95": float(torch.quantile(ratio, 0.95)),
        "max_over_meanabs_max": float(ratio.max()),
        "second_moment_kurtosis": float(kurt),
        "abs_p99_over_p50": float(q99 / q50),
        "abs_p999_over_p50": float(q999 / q50),
    }


def quant_metrics(x, mode):
    if mode == "absmax_a4":
        scale_base = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = 7.0 / scale_base
    elif mode == "bitnet_v2_absmean_a4":
        # BitNet v2 Eq. (8): beta=mean(|X|), scale=sqrt(7)/beta,
        # then RoundClip to [-8, 7].
        beta = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = math.sqrt(7.0) / beta
    else:
        raise ValueError(mode)
    scaled = x * scale
    rounded = torch.round(scaled)
    saturated = (rounded < -8) | (rounded > 7)
    q = rounded.clamp(-8, 7)
    recon = q / scale
    err = recon - x
    mse = err.square().mean()
    power = x.square().mean().clamp_min(1e-20)
    nmse = mse / power
    return {
        "mse": float(mse),
        "nmse": float(nmse),
        "snr_db": float(-10.0 * torch.log10(nmse.clamp_min(1e-30))),
        "saturation_fraction": float(saturated.float().mean()),
        "mean_abs_error": float(err.abs().mean()),
    }


def group_of(name):
    if name.endswith("self_attn.o_proj"):
        return "attention_o"
    if name.endswith("mlp.down_proj"):
        return "ffn_down"
    if name.endswith("self_attn.q_proj") or name.endswith("self_attn.k_proj") or name.endswith("self_attn.v_proj"):
        return "qkv"
    if name.endswith("mlp.up_proj") or name.endswith("mlp.gate_proj"):
        return "ffn_up_gate"
    return "other"


def median(values):
    return float(statistics.median(values)) if values else None


def summarize_group(rows):
    if not rows:
        return {"module_count": 0}
    keys = {
        "raw_outlier_ratio": lambda r: r["raw_distribution"]["max_over_meanabs_median"],
        "hadamard_outlier_ratio": lambda r: r["hadamard_distribution"]["max_over_meanabs_median"],
        "raw_absmax_nmse": lambda r: r["raw_absmax_a4"]["nmse"],
        "raw_v2_absmean_nmse": lambda r: r["raw_v2_absmean_a4"]["nmse"],
        "hadamard_v2_absmean_nmse": lambda r: r["hadamard_v2_absmean_a4"]["nmse"],
        "raw_v2_saturation": lambda r: r["raw_v2_absmean_a4"]["saturation_fraction"],
        "hadamard_v2_saturation": lambda r: r["hadamard_v2_absmean_a4"]["saturation_fraction"],
    }
    out = {"module_count": len(rows)}
    for k, fn in keys.items():
        vals = [fn(r) for r in rows]
        out[k + "_median"] = median(vals)
        out[k + "_mean"] = float(sum(vals) / len(vals))
    out["hadamard_v2_nmse_reduction_median"] = 1.0 - (
        out["hadamard_v2_absmean_nmse_median"] / max(out["raw_v2_absmean_nmse_median"], 1e-30)
    )
    out["hadamard_outlier_ratio_reduction_median"] = 1.0 - (
        out["hadamard_outlier_ratio_median"] / max(out["raw_outlier_ratio_median"], 1e-30)
    )
    return out


print("GATE010S_START", json.dumps({
    "model": MODEL,
    "seed": SEED,
    "calibration_tokens": N_CAL * SEQ,
    "sampled_token_rows_per_module": SAMPLED_TOKEN_ROWS,
    "preferred_hadamard_block": PREFERRED_HADAMARD_BLOCK,
    "purpose": "test whether BitNet-v2-style activation outlier localization explains Gate010R adaptive-codebook failure",
    "claim_ceiling": "diagnostic only; block-Hadamard reconstruction statistics are not an end-to-end BitNet v2 retrofit",
}), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=False, dtype=torch.bfloat16, device_map={"": "cpu"}
)
model.eval()
bitmods = [(n, m) for n, m in model.named_modules() if isinstance(m, AutoBitLinear)]
if not bitmods:
    raise RuntimeError("No AutoBitLinear modules found")
if any(bool(m.online_quant) for _, m in bitmods):
    raise RuntimeError("Expected offline AutoBitLinear modules")

wval = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
wval_text = "\n\n".join(x for x in wval["text"] if x and not x.isspace())
wval_ids = tok(wval_text, return_tensors="pt", add_special_tokens=False, truncation=False)["input_ids"][0]
cal = segments_from_ids(wval_ids, N_CAL, 0)
cal_hash = ids_hash(cal)
if cal_hash != EXPECTED_CAL_HASH:
    raise RuntimeError(f"calibration data hash mismatch: {cal_hash}")
cal_batch = torch.cat(cal, dim=0)

records = {}
orig_forwards = {n: m.forward for n, m in bitmods}


def collect_module(name, x):
    xs = sample_rows(x)
    d = xs.shape[-1]
    block = choose_hadamard_block(d)
    xh = fwht_blocks(xs, block)
    raw_dist = distribution_metrics(xs)
    had_dist = distribution_metrics(xh)
    raw_absmax = quant_metrics(xs, "absmax_a4")
    raw_v2 = quant_metrics(xs, "bitnet_v2_absmean_a4")
    had_v2 = quant_metrics(xh, "bitnet_v2_absmean_a4")
    records[name] = {
        "module": name,
        "group": group_of(name),
        "input_width": int(d),
        "sampled_token_rows": int(xs.shape[0]),
        "hadamard_block": int(block),
        "raw_distribution": raw_dist,
        "hadamard_distribution": had_dist,
        "raw_absmax_a4": raw_absmax,
        "raw_v2_absmean_a4": raw_v2,
        "hadamard_v2_absmean_a4": had_v2,
        "hadamard_v2_nmse_reduction": 1.0 - had_v2["nmse"] / max(raw_v2["nmse"], 1e-30),
        "hadamard_outlier_ratio_reduction": 1.0 - had_dist["max_over_meanabs_median"] / max(raw_dist["max_over_meanabs_median"], 1e-30),
    }


def make_wrapper(name, orig):
    def forward(self, input):
        x = self.rms_norm(input) if self.rms_norm is not None else input
        collect_module(name, x)
        return orig(input)
    return forward


for n, m in bitmods:
    m.forward = types.MethodType(make_wrapper(n, orig_forwards[n]), m)

t0 = time.time()
try:
    with torch.inference_mode():
        model(cal_batch, use_cache=False)
finally:
    for n, m in bitmods:
        m.forward = orig_forwards[n]
elapsed = time.time() - t0

if len(records) != len(bitmods):
    missing = [n for n, _ in bitmods if n not in records]
    raise RuntimeError(f"not all modules observed; missing={missing[:8]}")

rows = list(records.values())
groups = {}
for g in ("attention_o", "ffn_down", "qkv", "ffn_up_gate", "other"):
    groups[g] = summarize_group([r for r in rows if r["group"] == g])

path_rows = [r for r in rows if r["group"] in ("attention_o", "ffn_down")]
regular_rows = [r for r in rows if r["group"] in ("qkv", "ffn_up_gate")]
path = summarize_group(path_rows)
regular = summarize_group(regular_rows)

outlier_contrast = path["raw_outlier_ratio_median"] / max(regular["raw_outlier_ratio_median"], 1e-30)
nmse_contrast = path["raw_v2_absmean_nmse_median"] / max(regular["raw_v2_absmean_nmse_median"], 1e-30)
had_reduction = path["hadamard_v2_nmse_reduction_median"]
raw_absmax_nmse_contrast = path["raw_absmax_nmse_median"] / max(regular["raw_absmax_nmse_median"], 1e-30)

contrast_pass = (outlier_contrast >= MIN_PATHOLOGICAL_CONTRAST) or (nmse_contrast >= MIN_PATHOLOGICAL_CONTRAST) or (raw_absmax_nmse_contrast >= MIN_PATHOLOGICAL_CONTRAST)
hadamard_pass = had_reduction >= MIN_HADAMARD_NMSE_REDUCTION
supports = bool(contrast_pass and hadamard_pass)
status = (
    "GATE010S_DIAGNOSTIC_SUPPORTS_BITNET_V2_OUTLIER_MECHANISM"
    if supports else
    "GATE010S_DIAGNOSTIC_DOES_NOT_SUPPORT_BITNET_V2_OUTLIER_MECHANISM"
)

ranked = sorted(rows, key=lambda r: r["raw_v2_absmean_a4"]["nmse"], reverse=True)

final = {
    "experiment": "BITNET_GATE010S_OUTLIER_LOCALIZATION_AND_BLOCK_HADAMARD_A4_PROBE",
    "status": status,
    "config": {
        "model": MODEL,
        "seed": SEED,
        "calibration_tokens": N_CAL * SEQ,
        "calibration_hash": cal_hash,
        "autobitlinear_modules": len(bitmods),
        "sampled_token_rows_per_module": SAMPLED_TOKEN_ROWS,
        "preferred_hadamard_block": PREFERRED_HADAMARD_BLOCK,
        "min_pathological_contrast": MIN_PATHOLOGICAL_CONTRAST,
        "min_hadamard_nmse_reduction": MIN_HADAMARD_NMSE_REDUCTION,
        "seconds": elapsed,
    },
    "literature_mapped_prediction": {
        "pathological_groups": ["attention_o", "ffn_down"],
        "regular_groups": ["qkv", "ffn_up_gate"],
        "quantizer_probe": "BitNet v2 Eq.8 per-token absmean INT4",
        "rotation_probe": "orthonormal block Walsh-Hadamard, up to 256 channels per block",
        "important_limitation": "BitNet v2 trains H-BitLinear with Hadamard-aware weights and then continue-trains for INT4; this probe changes representation only to test quantization geometry and is not a functional model retrofit",
    },
    "decision_metrics": {
        "pathological_vs_regular_raw_outlier_ratio": outlier_contrast,
        "pathological_vs_regular_raw_v2_absmean_nmse": nmse_contrast,
        "pathological_vs_regular_raw_absmax_nmse": raw_absmax_nmse_contrast,
        "pathological_median_hadamard_v2_nmse_reduction": had_reduction,
        "contrast_pass": bool(contrast_pass),
        "hadamard_pass": bool(hadamard_pass),
        "all": supports,
    },
    "group_summary": groups,
    "combined_pathological": path,
    "combined_regular": regular,
    "top_20_modules_by_raw_v2_absmean_nmse": ranked[:20],
    "modules": rows,
    "claim_ceiling": {
        "scientific_claims_promoted": 0,
        "authorized_if_supports": "The frozen BitNet-2B4T calibration activations exhibit the BitNet-v2-predicted localization/quantization geometry under this diagnostic probe.",
        "not_authorized": [
            "A Hadamard retrofit preserves language-model perplexity.",
            "BitNet v2 can be reproduced without weight-side transformation or continue-training.",
            "Gate010R failure has a unique causal explanation.",
        ],
    },
}

with open("gate010s_results.json", "w") as f:
    json.dump(final, f, indent=2, sort_keys=True)

print("GROUP_SUMMARY", json.dumps(groups, sort_keys=True), flush=True)
print("DECISION", json.dumps(final["decision_metrics"], sort_keys=True), flush=True)
print("TOP_MODULES", json.dumps([
    {"module": r["module"], "group": r["group"], "raw_v2_nmse": r["raw_v2_absmean_a4"]["nmse"], "had_v2_nmse": r["hadamard_v2_absmean_a4"]["nmse"], "had_reduction": r["hadamard_v2_nmse_reduction"]}
    for r in ranked[:10]
], sort_keys=True), flush=True)
print("FINAL_JSON", json.dumps({"status": status, "decision_metrics": final["decision_metrics"], "seconds": elapsed}), flush=True)
