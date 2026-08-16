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
print("GATE001C_START", json.dumps({"model": MODEL, "seq": SEQ, "n_cal": N_CAL,
      "n_eval": N_EVAL, "seed": SEED, "clip_quantile": CLIP_Q,
      "hadamard_block": BLOCK}), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
layers = model.model.layers
D = model.config.hidden_size
if D % BLOCK:
    raise RuntimeError("hidden size must be divisible by block")

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(x for x in ds["text"] if x and not x.isspace())
ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
segments = [ids[i*SEQ:(i+1)*SEQ].unsqueeze(0) for i in range(N_CAL+N_EVAL)]
cal_segments, eval_segments = segments[:N_CAL], segments[N_CAL:]


def hidden(out): return out[0] if isinstance(out, tuple) else out
def repl(out, h): return (h,) + out[1:] if isinstance(out, tuple) else h
def rms(x):
    xf=x.float(); return torch.sqrt(torch.mean(xf*xf, dim=-1, keepdim=True).clamp_min(1e-12))

def hadamard(n):
    H=torch.ones(1,1)
    while H.shape[0] < n:
        H=torch.cat([torch.cat([H,H],1), torch.cat([H,-H],1)],0)
    return H/math.sqrt(n)
H=hadamard(BLOCK)
def rot(x):
    shp=x.shape; y=x.float().reshape(*shp[:-1],D//BLOCK,BLOCK)
    return torch.matmul(y,H.T).reshape(shp)
def irot(x):
    shp=x.shape; y=x.float().reshape(*shp[:-1],D//BLOCK,BLOCK)
    return torch.matmul(y,H).reshape(shp)

# Calibrate fixed post-Hadamard angular grid and fixed log-radius code range per layer.
u_samples=[[] for _ in layers]; lr_samples=[[] for _ in layers]; med_r=[]
handles=[]
for li,layer in enumerate(layers):
    def mk(i):
        def hook(_m,_inp,out):
            h=hidden(out).detach(); rr=rms(h); z=rot(h.float()/rr).abs().flatten()
            n=min(4096,z.numel()); idx=torch.linspace(0,z.numel()-1,n).long()
            u_samples[i].append(z[idx].cpu()); lr_samples[i].append(torch.log(rr.flatten()).cpu())
            return out
        return hook
    handles.append(layer.register_forward_hook(mk(li)))
with torch.inference_mode():
    for s in cal_segments: model(s,use_cache=False)
for h in handles: h.remove()
clips=[]; logr_lo=[]; logr_hi=[]; med_r=[]
for us,lrs in zip(u_samples,lr_samples):
    uv=torch.cat(us); lv=torch.cat(lrs)
    clips.append(max(float(torch.quantile(uv,CLIP_Q)),1.0))
    logr_lo.append(float(torch.quantile(lv,0.005)))
    logr_hi.append(float(torch.quantile(lv,0.995)))
    med_r.append(float(torch.exp(torch.median(lv))))
print("CALIBRATION",json.dumps({"rot_clip_median":float(torch.tensor(clips).median()),
      "logr_width_median":float((torch.tensor(logr_hi)-torch.tensor(logr_lo)).median()),
      "radius_median_min":min(med_r),"radius_median_max":max(med_r)}),flush=True)


def dyn4(h):
    hf=h.float(); qmax=7; a=hf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8); sc=a/qmax
    return (torch.round(hf/sc).clamp(-qmax,qmax)*sc).to(h.dtype)

def qdir4(z,clip):
    step=float(clip)/7
    return torch.round(torch.clamp(z,-float(clip),float(clip))/step)*step

def quant_log_radius(rr,bits,lo,hi):
    lr=torch.log(rr.float()).clamp(float(lo),float(hi)); levels=(2**bits)-1
    step=(float(hi)-float(lo))/levels
    if step<=0: return rr.float()
    lq=torch.round((lr-float(lo))/step)*step+float(lo)
    return torch.exp(lq)

def angularH4(h,i,rmode):
    hf=h.float(); rr=rms(h); u=hf/rr; qz=qdir4(rot(u),clips[i]); uh=irot(qz); uh=uh/rms(uh)
    if rmode=="fp32": rq=rr
    elif rmode=="fp16": rq=rr.to(torch.float16).float()
    elif rmode.startswith("log"):
        bits=int(rmode[3:]); rq=quant_log_radius(rr,bits,logr_lo[i],logr_hi[i])
    elif rmode=="shift1": rq=torch.roll(rr,1,dims=-2)
    elif rmode=="constant": rq=torch.full_like(rr,med_r[i])
    else: raise ValueError(rmode)
    return (rq*uh).to(h.dtype)

def evaluate(name,mode=None,rmode="fp32"):
    hs=[]
    if mode:
        for li,layer in enumerate(layers):
            def mk(i):
                def hook(_m,_inp,out):
                    h=hidden(out); hh=dyn4(h) if mode=="dyn4" else angularH4(h,i,rmode); return repl(out,hh)
                return hook
            hs.append(layer.register_forward_hook(mk(li)))
    total=0.; nt=0; t0=time.time()
    with torch.inference_mode():
        for s in eval_segments:
            o=model(s,labels=s,use_cache=False); n=s.shape[1]-1; total+=float(o.loss)*n; nt+=n
    for h in hs: h.remove()
    nll=total/nt; out={"name":name,"nll":nll,"ppl":math.exp(nll) if nll<50 else float('inf'),"seconds":time.time()-t0}
    print("RESULT",json.dumps(out),flush=True); return out

results=[evaluate("native"),evaluate("dynamic_A4","dyn4")]
for rm in ["fp32","fp16","log8","log6","log4","log2","shift1","constant"]:
    results.append(evaluate(f"angularH_A4_radius_{rm}","ang",rm))
base=results[0]["nll"]; dyn=results[1]["nll"]
for x in results:
    x["delta_nll_vs_native"]=x["nll"]-base; x["delta_nll_vs_dynamic_A4"]=x["nll"]-dyn
by={x["name"]:x for x in results}
fp=by["angularH_A4_radius_fp32"]; q4=by["angularH_A4_radius_log4"]
gate_pass=(q4["nll"] <= dyn) and ((q4["nll"]-fp["nll"]) <= 0.25)
print("FINAL_JSON",json.dumps({
    "experiment":"BITNET_PERSISTENT_RADIAL_ANGULAR_001_GATE001C",
    "scope":"radial bit-budget screen with fixed A4 Hadamard direction on pretrained RMSNorm LLM",
    "gate_rule":"log4 radius + angularH A4 beats dynamic A4 and is within +0.25 NLL of FP32-radius angularH A4",
    "gate_pass":gate_pass,
    "config":{"model":MODEL,"eval_prediction_tokens":N_EVAL*(SEQ-1),"seed":SEED,
              "radius_code":"uniform in calibrated per-layer log-RMS 0.5%-99.5% range"},
    "results":results},sort_keys=True),flush=True)
