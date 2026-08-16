import json
import math
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED=260816; MODEL="HuggingFaceTB/SmolLM2-135M"; SEQ=128; N_CAL=2; N_EVAL=4
CLIP_Q=0.995; BLOCK=64
random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
print("GATE001D_START",json.dumps({"model":MODEL,"seed":SEED,"matched_hadamard_control":True}),flush=True)

tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); model.eval()
layers=model.model.layers; D=model.config.hidden_size
if D%BLOCK: raise RuntimeError("hidden size not block divisible")
ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
text="\n\n".join(x for x in ds["text"] if x and not x.isspace())
ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
segments=[ids[i*SEQ:(i+1)*SEQ].unsqueeze(0) for i in range(N_CAL+N_EVAL)]
cal_segments,eval_segments=segments[:N_CAL],segments[N_CAL:]

def hidden(o): return o[0] if isinstance(o,tuple) else o
def repl(o,h): return (h,)+o[1:] if isinstance(o,tuple) else h
def rms(x):
    xf=x.float(); return torch.sqrt(torch.mean(xf*xf,dim=-1,keepdim=True).clamp_min(1e-12))
def had(n):
    H=torch.ones(1,1)
    while H.shape[0]<n: H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return H/math.sqrt(n)
H=had(BLOCK)
def rot(x):
    shp=x.shape; return torch.matmul(x.float().reshape(*shp[:-1],D//BLOCK,BLOCK),H.T).reshape(shp)
def irot(x):
    shp=x.shape; return torch.matmul(x.float().reshape(*shp[:-1],D//BLOCK,BLOCK),H).reshape(shp)

# fixed angular and log-radius calibration
zs=[[] for _ in layers]; lrs=[[] for _ in layers]; handles=[]
for li,layer in enumerate(layers):
    def mk(i):
        def hook(_m,_inp,o):
            h=hidden(o).detach(); rr=rms(h); z=rot(h.float()/rr).abs().flatten()
            n=min(4096,z.numel()); idx=torch.linspace(0,z.numel()-1,n).long(); zs[i].append(z[idx].cpu())
            lrs[i].append(torch.log(rr.flatten()).cpu()); return o
        return hook
    handles.append(layer.register_forward_hook(mk(li)))
with torch.inference_mode():
    for s in cal_segments: model(s,use_cache=False)
for x in handles:x.remove()
clips=[]; lo=[]; hi=[]; med=[]
for zbank,lbank in zip(zs,lrs):
    z=torch.cat(zbank); lr=torch.cat(lbank); clips.append(max(float(torch.quantile(z,CLIP_Q)),1.0))
    lo.append(float(torch.quantile(lr,0.005))); hi.append(float(torch.quantile(lr,0.995))); med.append(float(torch.exp(torch.median(lr))))
print("CAL",json.dumps({"clip_median":float(torch.tensor(clips).median()),"logr_width_median":float((torch.tensor(hi)-torch.tensor(lo)).median())}),flush=True)

def sym_a4(z,scale_kind="absmax"):
    zf=z.float(); qmax=7
    if scale_kind=="absmax": a=zf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8); sc=a/qmax
    elif scale_kind=="absmean":
        # BitNet-v2-like A4 scale family: per-token absmean / sqrt(7).
        a=zf.abs().mean(dim=-1,keepdim=True).clamp_min(1e-8); sc=a/math.sqrt(7.0)
    else: raise ValueError(scale_kind)
    return torch.round(zf/sc).clamp(-qmax,qmax)*sc

def raw_dyn4(h): return sym_a4(h,"absmax").to(h.dtype)
def h_dyn4(h,kind): return irot(sym_a4(rot(h),kind)).to(h.dtype)
def qdir4(z,c):
    step=float(c)/7; return torch.round(torch.clamp(z,-float(c),float(c))/step)*step
def qlog(rr,bits,l,h):
    lr=torch.log(rr.float()).clamp(float(l),float(h)); levels=2**bits-1; step=(float(h)-float(l))/levels
    return torch.exp(torch.round((lr-float(l))/step)*step+float(l))
def angularH4(h,i,rmode):
    rr=rms(h); u=h.float()/rr; uh=irot(qdir4(rot(u),clips[i])); uh=uh/rms(uh)
    if rmode=="fp32": rq=rr
    elif rmode=="log4": rq=qlog(rr,4,lo[i],hi[i])
    elif rmode=="shift1": rq=torch.roll(rr,1,dims=-2)
    elif rmode=="constant": rq=torch.full_like(rr,med[i])
    else: raise ValueError(rmode)
    return (rq*uh).to(h.dtype)

def evaluate(name,mode=None,rmode="fp32"):
    hs=[]
    if mode:
        for li,layer in enumerate(layers):
            def mk(i):
                def hook(_m,_inp,o):
                    h=hidden(o)
                    if mode=="raw": hh=raw_dyn4(h)
                    elif mode=="hmax": hh=h_dyn4(h,"absmax")
                    elif mode=="hmean": hh=h_dyn4(h,"absmean")
                    elif mode=="angular": hh=angularH4(h,i,rmode)
                    else: raise ValueError(mode)
                    return repl(o,hh)
                return hook
            hs.append(layer.register_forward_hook(mk(li)))
    total=0.; nt=0; t0=time.time()
    with torch.inference_mode():
        for s in eval_segments:
            o=model(s,labels=s,use_cache=False); n=s.shape[1]-1; total+=float(o.loss)*n; nt+=n
    for x in hs:x.remove()
    nll=total/nt; r={"name":name,"nll":nll,"ppl":math.exp(nll),"seconds":time.time()-t0}; print("RESULT",json.dumps(r),flush=True); return r

results=[
 evaluate("native"),
 evaluate("raw_dynamic_A4_absmax","raw"),
 evaluate("hadamard_dynamic_A4_absmax","hmax"),
 evaluate("hadamard_dynamic_A4_absmean","hmean"),
 evaluate("angularH_A4_radius_fp32","angular","fp32"),
 evaluate("angularH_A4_radius_log4","angular","log4"),
 evaluate("angularH_A4_radius_shift1","angular","shift1"),
 evaluate("angularH_A4_radius_constant","angular","constant"),
]
base=results[0]["nll"]; by={x["name"]:x for x in results}
for x in results:x["delta_nll_vs_native"]=x["nll"]-base
ang=by["angularH_A4_radius_log4"]; hmax=by["hadamard_dynamic_A4_absmax"]
# Non-redundant screen: separate radial/fixed-angular code must beat the matched
# rotated per-token dynamic baseline, not merely an unrotated A4 baseline.
gate_pass=ang["nll"] < hmax["nll"]
print("FINAL_JSON",json.dumps({
 "experiment":"BITNET_PERSISTENT_RADIAL_ANGULAR_001_GATE001D",
 "scope":"matched-Hadamard control; activation-geometry pilot only, not BitNet-weight confirmation",
 "gate_rule":"angularH A4 + 4-bit log-radius NLL < Hadamard per-token absmax A4 NLL",
 "gate_pass":gate_pass,"config":{"model":MODEL,"eval_prediction_tokens":N_EVAL*(SEQ-1),"seed":SEED},"results":results
},sort_keys=True),flush=True)
