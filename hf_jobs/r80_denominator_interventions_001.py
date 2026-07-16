import os, re, json, base64, hashlib, tempfile, pathlib, types, random, time
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
import transformers, datasets

NS = "craigbotnen"
SOURCE_JOB_ID = os.environ["SOURCE_JOB_ID"]
TOKEN = os.environ["HF_TOKEN"]
REPO_ID = f"{NS}/r80-denominator-interventions-001"

api = HfApi(token=TOKEN)
job = api.inspect_job(job_id=SOURCE_JOB_ID, namespace=NS)
cmd = job.command[-1]
match = re.search(r'echo "([A-Za-z0-9+/=]+)" \| base64 -d', cmd)
if not match:
    raise RuntimeError("Could not recover frozen source from source job command")
source_bytes = base64.b64decode(match.group(1))
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
source = source_bytes.decode("utf-8")
if "random.seed(SEED)" not in source:
    raise RuntimeError("Frozen source split marker absent")
prefix = source.split("random.seed(SEED)", 1)[0]
ns = {}
exec(compile(prefix, "frozen_source_prefix.py", "exec"), ns)

SEED = int(ns["SEED"])
DEVICE = ns["DEVICE"]
STEPS = ns["STEPS"]
SNAPS = ns["SNAPS"]
LR = ns["LR"]
L = ns["L"]
FF = ns["FF"]
EPS = ns["EPS"]
LM = ns["LM"]
TM = ns["TM"]
mapped = ns["mapped"]
sample = ns["sample"]
TR = ns["TR"]
VAL = ns["VAL"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

base = LM("BASE2432").to(torch.bfloat16).cpu()
m = mapped("TYPED_R80", base).to(DEVICE, dtype=torch.bfloat16)
opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
g = torch.Generator().manual_seed(SEED + 404)

def eval_nll(model):
    model.eval()
    total = 0.0
    nt = 0
    with torch.inference_mode():
        for x, y in VAL:
            z = model(x)
            total += float(F.cross_entropy(z.float().reshape(-1, 256), y.reshape(-1), reduction="sum"))
            nt += y.numel()
    return total / nt

def install_mode(model, mode, layer_constants=None):
    originals = []
    call_counts = [0] * L
    for li, block in enumerate(model.blocks):
        tm = block.mlp
        originals.append(tm.forward)
        def patched(self, x, stats=None, zero=False, _li=li):
            ht = F.relu(self.gt(x)).square() * self.ut(x)
            he = F.relu(self.ge(x)).square() * self.ue(x)
            a = ht.float().square().sum(-1, keepdim=True)
            e = he.float().square().sum(-1, keepdim=True)
            if mode == "zero":
                ee = torch.zeros_like(e)
            elif mode == "layer_mean":
                ee = torch.full_like(e, float(layer_constants["mean_e"][_li]))
            elif mode == "denominator_matched":
                ee = torch.full_like(e, float(layer_constants["denom_c"][_li]))
            elif mode == "cross_sequence":
                ee = torch.roll(e, shifts=1, dims=0)
            elif mode == "token_shuffled":
                n = e.numel()
                gen = torch.Generator(device="cpu").manual_seed(
                    9700003 + 1009 * SEED + 97 * _li + call_counts[_li]
                )
                idx = torch.randperm(n, generator=gen).to(e.device)
                ee = e.reshape(-1)[idx].reshape_as(e)
                call_counts[_li] += 1
            else:
                ee = e
            var = (a + ee) / FF
            if stats is not None:
                stats.append((ee / (a + ee).clamp_min(1e-30)).detach().cpu().reshape(-1))
            return self.down(self.gamma * (ht.float() * torch.rsqrt(var + EPS)).to(ht.dtype))
        tm.forward = types.MethodType(patched, tm)
    return originals

def restore_mode(model, originals):
    for block, original in zip(model.blocks, originals):
        block.mlp.forward = original

def calibrate(model):
    captures_a = [[] for _ in range(L)]
    captures_e = [[] for _ in range(L)]
    originals = []
    for li, block in enumerate(model.blocks):
        tm = block.mlp
        originals.append(tm.forward)
        def capture(self, x, stats=None, zero=False, _li=li):
            ht = F.relu(self.gt(x)).square() * self.ut(x)
            he = F.relu(self.ge(x)).square() * self.ue(x)
            a = ht.float().square().sum(-1, keepdim=True)
            e = he.float().square().sum(-1, keepdim=True)
            captures_a[_li].append(a.detach().cpu().reshape(-1))
            captures_e[_li].append(e.detach().cpu().reshape(-1))
            var = (a + e) / FF
            return self.down(self.gamma * (ht.float() * torch.rsqrt(var + EPS)).to(ht.dtype))
        tm.forward = types.MethodType(capture, tm)
    _ = eval_nll(model)
    restore_mode(model, originals)
    means = []
    denom_cs = []
    summary = []
    for li in range(L):
        a = torch.cat(captures_a[li]).double()
        e = torch.cat(captures_e[li]).double()
        mean_e = float(e.mean())
        target = torch.rsqrt(a + e + EPS).mean()
        lo = 0.0
        hi = max(float(e.max()) * 4.0, mean_e * 16.0, 1.0)
        for _ in range(100):
            mid = (lo + hi) / 2.0
            val = torch.rsqrt(a + mid + EPS).mean()
            if val > target:
                lo = mid
            else:
                hi = mid
        c = (lo + hi) / 2.0
        means.append(mean_e)
        denom_cs.append(c)
        summary.append({
            "layer": li,
            "mean_a": float(a.mean()),
            "mean_e": mean_e,
            "denominator_matched_constant_e": c,
            "mean_energy_fraction": float((e / (a + e).clamp_min(1e-30)).mean()),
        })
    return {"mean_e": means, "denom_c": denom_cs, "layers": summary}

snapshots = []
print(json.dumps({
    "event": "START",
    "campaign": "R80_DENOMINATOR_INTERVENTIONS_001",
    "source_job_id": SOURCE_JOB_ID,
    "seed": SEED,
    "source_sha256": source_sha256,
    "gpu": torch.cuda.get_device_name(0),
    "steps": STEPS,
}), flush=True)

t0 = time.time()
for step in range(1, STEPS + 1):
    m.train()
    x, y = sample(TR, g)
    z = m(x)
    loss = F.cross_entropy(z.float().reshape(-1, 256), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if step in SNAPS:
        nll = eval_nll(m)
        rec = {"step": step, "nll": nll, "seconds": time.time() - t0}
        snapshots.append(rec)
        print("SNAP " + json.dumps(rec), flush=True)

native_nll = eval_nll(m)
calibration = calibrate(m)
interventions = {"native": native_nll}
for mode in ["zero", "layer_mean", "denominator_matched", "token_shuffled", "cross_sequence"]:
    originals = install_mode(m, mode, calibration)
    interventions[mode] = eval_nll(m)
    restore_mode(m, originals)

deltas = {k: v - native_nll for k, v in interventions.items() if k != "native"}
result = {
    "campaign": "R80_DENOMINATOR_INTERVENTIONS_001",
    "classification": "FRESH_REPLICATION_PLUS_PROSPECTIVE_CHECKPOINT_INTERVENTIONS",
    "source_job_id": SOURCE_JOB_ID,
    "source_job_url": job.url,
    "seed": SEED,
    "source_sha256": source_sha256,
    "snapshots": snapshots,
    "interventions_nll": interventions,
    "intervention_delta_nll_vs_native": deltas,
    "calibration": calibration["layers"],
    "package_versions": {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
    },
}

api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    out = pathlib.Path(td) / f"seed_{SEED}"
    out.mkdir()
    (out / "frozen_source.py").write_bytes(source_bytes)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "source_job_spec.json").write_text(json.dumps({
        "id": job.id,
        "created_at": str(job.created_at),
        "docker_image": job.docker_image,
        "command": job.command,
        "arguments": job.arguments,
        "environment": job.environment,
        "flavor": job.flavor,
        "status": {"stage": job.status.stage, "message": job.status.message},
        "url": job.url,
    }, indent=2, default=str))
    ckpt = out / "typed_r80_step4000_state_dict.pt"
    torch.save({
        "campaign": result["campaign"],
        "source_job_id": SOURCE_JOB_ID,
        "seed": SEED,
        "source_sha256": source_sha256,
        "model_state_dict": {k: v.detach().cpu() for k, v in m.state_dict().items()},
        "result": result,
    }, ckpt)
    manifest = []
    for p in sorted(out.iterdir()):
        manifest.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out / "SHA256_MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(out),
        path_in_repo=f"seed_{SEED}",
        commit_message=f"Add seed {SEED} checkpoint and intervention results",
    )
result["artifact_repo"] = REPO_ID
print("FINAL " + json.dumps(result, indent=2), flush=True)
