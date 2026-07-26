from __future__ import annotations

import base64, hashlib, inspect, json, os, platform, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo, snapshot_download, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer

PROTOCOL_CODE = "BITNET_ZERO_DOWNSTREAM_DENOMINATOR_CONTROL_PROSPECTIVE_001"
PACKAGE_SHA256 = "2145f9f2708c622eb08babcf0ced9c972f6f44b78d3ede667d0d2cace9272e25"
MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
OUTPUT_REPO = os.environ.get("OUTPUT_REPO_ID", "craigbotnen/bitnet-denominator-control-stage-ab-001")
NEAR_ZERO = 1e-6
ENERGY_MIN = 0.005
UNIFORM_MULTIPLE_MIN = 32.0
TOKENS_MIN = 3
EPS_EXPECTED = 1e-5
PROMPTS = [
"The capital of France is", "A careful proof begins by", "In a small town near the mountains,",
"Explain why the sky appears blue.", "Write one sentence about a piano.", "Two plus two equals",
"The opposite of hot is", "A scientist tests a hypothesis by", "The main character opened the door and",
"Summarize the purpose of a checksum.", "An efficient computer program should",
"The word 'because' usually introduces", "A triangle has three", "If rain begins, the street may",
"The musician counted four beats and", "A reliable experiment records", "The library closed after",
"Describe a calm lake at sunrise.", "The next prime number after eleven is", "A map helps a traveler",
"One reason to verify a model is", "The sentence ends with a", "When the temperature falls below freezing,",
"A theorem requires assumptions and", "The child placed the book on", "A computer stores information as",
"The judge examined the evidence before", "A good explanation distinguishes",
"The train arrived at the station", "The experiment failed because", "In music, harmony refers to",
"The safest conclusion from limited data is"
]


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk): h.update(b)
    return h.hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_info():
    import accelerate, huggingface_hub, safetensors, transformers
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "accelerate": accelerate.__version__,
            "huggingface_hub": huggingface_hub.__version__, "safetensors": safetensors.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


root = Path("/tmp/bitnet_stage_ab_001")
out = root / "run"
scan = out / "STAGE_AB_001"
scan.mkdir(parents=True, exist_ok=False)
start = time.time()
if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for the authorized A10G scan")

authority = {
    "authority_code": "BITNET_DENOMINATOR_CONTROL_STAGE_AB_AUTHORITY_001",
    "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
    "authorization_source": "Explicit user authorization in the active ChatGPT conversation",
    "prospective_package_sha256": PACKAGE_SHA256,
    "authorized_stages": ["A_STRUCTURAL_SCREEN", "B_ACTIVATION_ENERGY_PROFILING"],
    "stage_c_authorized": False, "threshold_changes_authorized": False,
    "claim_ceiling": "CHECKPOINT_LOCAL_STAGE_AB_SCREEN_ONLY"
}
dump(out / "STAGE_AB_EXECUTION_AUTHORITY.json", authority)

info = HfApi().model_info(MODEL_ID, revision="main")
if not info.sha: raise RuntimeError("Could not resolve immutable model commit")
revision = info.sha
snapshot = Path(snapshot_download(MODEL_ID, revision=revision, cache_dir=str(root / "cache")))
files = []
for p in sorted(x for x in snapshot.rglob("*") if x.is_file()):
    files.append({"path": p.relative_to(snapshot).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})

model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, torch_dtype="auto",
                                             low_cpu_mem_usage=True, device_map={"": "cuda"})
model.eval()
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
layers = model.model.layers
if len(layers) != int(model.config.num_hidden_layers): raise RuntimeError("Layer count mismatch")
for i, layer in enumerate(layers):
    for name in ("gate_proj", "up_proj", "ffn_sub_norm", "down_proj"):
        if not hasattr(layer.mlp, name): raise RuntimeError(f"Layer {i} missing mlp.{name}")

norm = layers[0].mlp.ffn_sub_norm
d = int(model.config.intermediate_size)
probe = torch.linspace(-2, 2, d, dtype=torch.float32, device="cuda").reshape(1, 1, d)
with torch.no_grad():
    observed = norm(probe)
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", float("nan"))))
    expected = (probe * torch.rsqrt(probe.pow(2).mean(-1, keepdim=True) + eps) * norm.weight.float()).to(observed.dtype)
probe_error = float((observed - expected).abs().max().item())
if probe_error > 5e-6: raise RuntimeError(f"RMS probe failed: {probe_error}")
if abs(eps - EPS_EXPECTED) > 1e-12: raise RuntimeError(f"Unexpected RMS epsilon: {eps}")
source_file = inspect.getsourcefile(type(norm))
architecture = {"architecture_class": type(model).__qualname__, "normalization_class": type(norm).__qualname__,
                "source_file": source_file, "source_sha256": sha256_file(Path(source_file)),
                "rms_probe_max_abs_error": probe_error, "hidden_size": int(model.config.hidden_size),
                "intermediate_size": d, "num_hidden_layers": len(layers), "rms_norm_eps": eps,
                "hidden_act": str(model.config.hidden_act)}
source_binding = {"protocol_code": PROTOCOL_CODE, "prospective_package_sha256": PACKAGE_SHA256,
                  "model_id": MODEL_ID, "requested_revision": "main", "resolved_revision": revision,
                  "snapshot_path": str(snapshot), "config_sha256": sha256_file(snapshot / "config.json"),
                  "snapshot_manifest": files, "runtime": runtime_info(), "architecture": architecture,
                  "prompts_sha256": hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest()}
dump(scan / "SOURCE_BINDING.json", source_binding)

rows, layer_summary = [], []
for li, layer in enumerate(layers):
    down, gamma = layer.mlp.down_proj.weight.detach(), layer.mlp.ffn_sub_norm.weight.detach()
    if down.ndim != 2 or down.shape[1] != gamma.numel(): raise RuntimeError(f"Layer {li} shape mismatch")
    l2 = torch.empty(d, dtype=torch.float64); linf = torch.empty(d, dtype=torch.float64); nnz = torch.empty(d, dtype=torch.int64)
    for a in range(0, d, 256):
        b = min(a + 256, d)
        block = down[:, a:b].float() * gamma[a:b].float().unsqueeze(0)
        l2[a:b] = torch.linalg.vector_norm(block, dim=0).cpu().double()
        linf[a:b] = block.abs().amax(dim=0).cpu().double()
        nnz[a:b] = torch.count_nonzero(block, dim=0).cpu()
    positive = l2[l2 > 0]; med = float(positive.median()) if positive.numel() else 0.0
    rel = l2 / med if med > 0 else torch.full_like(l2, float("inf"))
    exact = int((nnz == 0).sum()); near = int(((nnz != 0) & (rel <= NEAR_ZERO)).sum())
    layer_summary.append({"layer": li, "dimension": d, "median_positive_effective_column_l2": med,
                          "exact_zero_count": exact, "near_zero_exploratory_count": near})
    for j in range(d):
        cls = "EXACT_ZERO" if int(nnz[j]) == 0 else "NEAR_ZERO_EXPLORATORY" if float(rel[j]) <= NEAR_ZERO else "NONZERO"
        rows.append({"layer": li, "coordinate": j, "effective_column_l2": float(l2[j]),
                     "effective_column_linf": float(linf[j]), "effective_column_nonzero_count": int(nnz[j]),
                     "relative_l2_to_layer_median": float(rel[j]), "structural_class": cls})
dump(scan / "STRUCTURAL_LAYER_SUMMARY.json", layer_summary)

sum_fraction = [torch.zeros(d, dtype=torch.float64) for _ in layers]
max_fraction = [torch.zeros(d, dtype=torch.float64) for _ in layers]
count_fraction = [torch.zeros(d, dtype=torch.int64) for _ in layers]
count_multiple = [torch.zeros(d, dtype=torch.int64) for _ in layers]
token_count = [0 for _ in layers]
handles = []
for li, layer in enumerate(layers):
    def hook(_m, inputs, li=li):
        z = inputs[0].detach().float().reshape(-1, d)
        frac = (z.pow(2) / (z.pow(2).sum(-1, keepdim=True) + d * eps)).cpu().double()
        sum_fraction[li] += frac.sum(0); max_fraction[li] = torch.maximum(max_fraction[li], frac.amax(0))
        count_fraction[li] += (frac >= ENERGY_MIN).sum(0)
        count_multiple[li] += (frac * d >= UNIFORM_MULTIPLE_MIN).sum(0); token_count[li] += z.shape[0]
    handles.append(layer.mlp.ffn_sub_norm.register_forward_pre_hook(hook))
try:
    with torch.inference_mode():
        for text in PROMPTS:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            encoded = {k: v.cuda() for k, v in encoded.items()}
            model(**encoded, use_cache=False, logits_to_keep=1)
finally:
    for h in handles: h.remove()

by_key = {(r["layer"], r["coordinate"]): r for r in rows}
for li in range(len(layers)):
    denom = max(token_count[li], 1)
    for j in range(d):
        r = by_key[(li, j)]; mx = float(max_fraction[li][j])
        r.update({"mean_energy_fraction": float(sum_fraction[li][j] / denom), "max_energy_fraction": mx,
                  "max_uniform_share_multiple": mx * d, "tokens_above_energy_fraction": int(count_fraction[li][j]),
                  "tokens_above_uniform_multiple": int(count_multiple[li][j])})
        r["controlling_candidate"] = bool(r["structural_class"] == "EXACT_ZERO" and mx >= ENERGY_MIN
            and mx * d >= UNIFORM_MULTIPLE_MIN and int(count_fraction[li][j]) >= TOKENS_MIN)
rows.sort(key=lambda r: (not r["controlling_candidate"], -r["max_energy_fraction"], r["effective_column_l2"], r["layer"], r["coordinate"]))
with (scan / "CANDIDATES.jsonl").open("w") as f:
    for r in rows: f.write(json.dumps(r, sort_keys=True) + "\n")

exact_rows = [r for r in rows if r["structural_class"] == "EXACT_ZERO"]
near_rows = [r for r in rows if r["structural_class"] == "NEAR_ZERO_EXPLORATORY"]
controlling = [{"layer": r["layer"], "coordinate": r["coordinate"]} for r in rows if r["controlling_candidate"]]
status = "CONTROLLING_CANDIDATE_FOUND" if controlling else "NO_CONTROLLING_CANDIDATE_OBSERVED"
summary = {"protocol_code": PROTOCOL_CODE, "status": status, "external_stage_ab_authority": True,
           "resolved_revision": revision, "prompts_completed": len(PROMPTS), "tokens_profiled_by_layer": token_count,
           "exact_zero_coordinates": len(exact_rows), "near_zero_exploratory_coordinates": len(near_rows),
           "controlling_candidates": controlling, "stage_c_executed": False,
           "next_action": "SEPARATE_CAUSAL_EXECUTION_AUTHORITY_REQUIRED" if controlling else "STOP_OR_OPEN_SEPARATELY_JUSTIFIED_REPLICATION"}
dump(scan / "SCAN_SUMMARY.json", summary)

compact = {"summary": summary, "source_binding_core": {k: source_binding[k] for k in
           ("model_id", "requested_revision", "resolved_revision", "config_sha256", "runtime", "architecture", "prompts_sha256")},
           "snapshot_file_count": len(files), "snapshot_total_bytes": sum(x["size_bytes"] for x in files),
           "layer_summary": layer_summary, "exact_zero_rows": exact_rows,
           "near_zero_exploratory_top_100": sorted(near_rows, key=lambda r: r["relative_l2_to_layer_median"])[:100],
           "top_energy_rows_100": sorted(rows, key=lambda r: -r["max_energy_fraction"])[:100],
           "elapsed_seconds": time.time() - start, "stage_c_executed": False}
dump(out / "COMPACT_RESULTS.json", compact)
receipt = {"receipt_code": "BITNET_DENOMINATOR_CONTROL_STAGE_AB_EXECUTION_RECEIPT_001",
           "completed_at_utc": datetime.now(timezone.utc).isoformat(), "process_exit_code": 0,
           "elapsed_seconds": time.time() - start, "prospective_package_sha256": PACKAGE_SHA256,
           "authority_sha256": sha256_file(out / "STAGE_AB_EXECUTION_AUTHORITY.json"),
           "source_binding_sha256": sha256_file(scan / "SOURCE_BINDING.json"),
           "structural_layer_summary_sha256": sha256_file(scan / "STRUCTURAL_LAYER_SUMMARY.json"),
           "candidates_sha256": sha256_file(scan / "CANDIDATES.jsonl"), "scan_summary_sha256": sha256_file(scan / "SCAN_SUMMARY.json"),
           "status": status, "prompts_completed": len(PROMPTS), "exact_zero_coordinates": len(exact_rows),
           "near_zero_exploratory_coordinates": len(near_rows), "controlling_candidates": controlling,
           "stage_c_executed": False, "next_action": summary["next_action"]}
dump(out / "STAGE_AB_EXECUTION_RECEIPT.json", receipt)
manifest = []
for p in sorted(x for x in out.rglob("*") if x.is_file()):
    manifest.append({"path": p.relative_to(out).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
dump(out / "RESULTS_MANIFEST.json", manifest)
(out / "README.md").write_text(f"# Stage A+B results\n\nStatus: `{status}`\n\nExact-zero coordinates: `{len(exact_rows)}`\n\nControlling candidates: `{len(controlling)}`\n\nStage C: not executed.\n")

compact_zip = root / "BITNET_STAGE_AB_COMPACT_RESULTS_001.zip"
with zipfile.ZipFile(compact_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in [out / "COMPACT_RESULTS.json", out / "STAGE_AB_EXECUTION_RECEIPT.json", out / "RESULTS_MANIFEST.json", out / "README.md",
              scan / "SCAN_SUMMARY.json", scan / "SOURCE_BINDING.json", scan / "STRUCTURAL_LAYER_SUMMARY.json"]:
        z.write(p, arcname=p.relative_to(out).as_posix())

token = os.environ.get("HF_TOKEN")
if not token: raise RuntimeError("HF_TOKEN missing")
create_repo(OUTPUT_REPO, repo_type="dataset", private=True, exist_ok=True, token=token)
upload_folder(repo_id=OUTPUT_REPO, repo_type="dataset", folder_path=str(out), path_in_repo="run", token=token)
final = {"status": status, "resolved_revision": revision, "prompts_completed": len(PROMPTS),
         "exact_zero_coordinates": len(exact_rows), "near_zero_exploratory_coordinates": len(near_rows),
         "controlling_candidates": controlling, "elapsed_seconds": time.time() - start,
         "execution_receipt_sha256": sha256_file(out / "STAGE_AB_EXECUTION_RECEIPT.json"),
         "compact_bundle_sha256": sha256_file(compact_zip), "private_dataset_repo": OUTPUT_REPO, "stage_c_executed": False}
print("STAGE_AB_FINAL_SUMMARY=" + json.dumps(final, sort_keys=True), flush=True)
print("STAGE_AB_COMPACT_BUNDLE_B64=" + base64.b64encode(compact_zip.read_bytes()).decode(), flush=True)
