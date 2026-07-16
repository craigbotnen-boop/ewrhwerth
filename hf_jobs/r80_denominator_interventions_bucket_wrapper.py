import pathlib
import urllib.request

SOURCE_URL = (
    "https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/"
    "r80-interventions-001/hf_jobs/r80_denominator_interventions_001.py"
)
source = urllib.request.urlopen(SOURCE_URL, timeout=60).read().decode("utf-8")
start_marker = 'api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)\n'
end_marker = 'result["artifact_repo"] = REPO_ID\n'
start = source.index(start_marker)
end = source.index(end_marker, start) + len(end_marker)
replacement = '''
out = pathlib.Path("/artifacts") / f"seed_{SEED}"
out.mkdir(parents=True, exist_ok=True)
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
    if p.name != "SHA256_MANIFEST.txt":
        manifest.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
(out / "SHA256_MANIFEST.txt").write_text("\\n".join(manifest) + "\\n")
result["artifact_bucket"] = "craigbotnen/r80-denominator-interventions-001"
'''
patched = source[:start] + replacement + source[end:]
compile(patched, "r80_denominator_interventions_bucket_patched.py", "exec")
exec(patched, {"__name__": "__main__"})
