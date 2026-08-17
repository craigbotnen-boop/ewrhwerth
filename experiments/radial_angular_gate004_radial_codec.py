import json, math, random, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED=26081604
MODEL="microsoft/bitnet-b1.58-2B-4T"
SEQ=128; N_CAL=2; N_EVAL=8; CLIP_Q=0.995; BLOCK=64
BITS=(8,6,5,4)
TOL_NLL=0.10
QLO=0.005; QHI=0.995
EVAL_BATCH=4
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
print("GATE004_START",json.dumps({
    "model":MODEL,"seed":SEED,"seq":SEQ,"n_cal":N_CAL,"n_eval":N_EVAL,
    "calibration_tokens":N_CAL*SEQ,"eval_prediction_tokens":N_EVAL*(SEQ-1),
    "bits":BITS,"tolerance_nll":TOL_NLL,"eval_batch":EVAL_BATCH,
    "rule":"codec passes iff NLL <= exact-radius NLL + 0.10 and NLL < same-run matched Hadamard dynamic A4; minimum passing bitwidth is selected; cascaded ranges are calibrated from already-quantized upstream trajectories"
}),flush=True)

device=torch.device("cpu")
tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=False,dtype=torch.bfloat16,device_map={"":"cpu"}); model.eval()
layers=model.model.layers; D=int(model.config.hidden_size)
if D%BLOCK: raise RuntimeError("hidden size not divisible by Hadamard block")
print("MODEL_INFO",json.dumps({"layers":len(layers),"hidden_size":D,"rms_norm_eps":float(model.config.rms_norm_eps),"quantization_config":str(getattr(model.config,"quantization_config",None))}),flush=True)

def hidden(o): return o if torch.is_tensor(o) else o[0]
def repl(o,h): return h if torch.is_tensor(o) else ((h,)+o[1:] if isinstance(o,tuple) else [h]+list(o[1:]))
def rms(x):
    xf=x.float(); return torch.sqrt(torch.mean(xf*xf,dim=-1,keepdim=True).clamp_min(1e-12))
def had(n):
    H=torch.ones(1,1,dtype=torch.float32)
    while H.shape[0]<n: H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return H/math.sqrt(n)
H=had(BLOCK)
def rot(x):
    shp=x.shape; return torch.matmul(x.float().reshape(*shp[:-1],D//BLOCK,BLOCK),H.T).reshape(shp)
def irot(x):
    shp=x.shape; return torch.matmul(x.float().reshape(*shp[:-1],D//BLOCK,BLOCK),H).reshape(shp)

ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
text="\n\n".join(x for x in ds["text"] if x and not x.isspace())
ids=tok(text,return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
segments=[ids[i*SEQ:(i+1)*SEQ].unsqueeze(0) for i in range(N_CAL+N_EVAL)]
if any(s.shape[1]!=SEQ for s in segments): raise RuntimeError("not enough frozen tokens")
cal,ev=segments[:N_CAL],segments[N_CAL:]
cal_batch=torch.cat(cal,dim=0)

# Native-trajectory calibration for the fixed angular grid and static radial ranges.
zs=[[] for _ in layers]; lrs=[[] for _ in layers]; handles=[]
for li,layer in enumerate(layers):
    def mk(i):
        def hook(_m,_inp,o):
            h=hidden(o).detach(); rr=rms(h); z=rot(h.float()/rr).abs().flatten()
            n=min(8192,z.numel()); idx=torch.linspace(0,z.numel()-1,n).long(); zs[i].append(z[idx].cpu())
            lrs[i].append(torch.log(rr.flatten()).cpu()); return None
        return hook
    handles.append(layer.register_forward_hook(mk(li)))
with torch.inference_mode(): model(cal_batch,use_cache=False)
for h in handles:h.remove()
clips=[]; static_lo=[]; static_hi=[]
for zbank,lbank in zip(zs,lrs):
    z=torch.cat(zbank); lr=torch.cat(lbank)
    clips.append(max(float(torch.quantile(z,CLIP_Q)),1.0))
    static_lo.append(float(torch.quantile(lr,QLO))); static_hi.append(float(torch.quantile(lr,QHI)))
print("NATIVE_CAL",json.dumps({"clip_min":min(clips),"clip_median":float(torch.tensor(clips).median()),"clip_max":max(clips),"logr_width_min":min(h-l for l,h in zip(static_lo,static_hi)),"logr_width_median":float(torch.tensor([h-l for l,h in zip(static_lo,static_hi)]).median()),"logr_width_max":max(h-l for l,h in zip(static_lo,static_hi))}),flush=True)

def sym_a4(z):
    zf=z.float(); qmax=7; a=zf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8); sc=a/qmax
    return torch.round(zf/sc).clamp(-qmax,qmax)*sc

def dynamic_a4(h): return irot(sym_a4(rot(h))).to(h.dtype)
def qdir4(z,c):
    step=float(c)/7.0; return torch.round(torch.clamp(z,-float(c),float(c))/step)*step

def qlog(rr,bits,l,h):
    lr=torch.log(rr.float()).clamp(float(l),float(h)); levels=(1<<bits)-1
    if h<=l+1e-12: return torch.exp(torch.full_like(lr,float(l)))
    step=(float(h)-float(l))/levels
    return torch.exp(torch.round((lr-float(l))/step)*step+float(l))

def angular_quant(h,i,radial_mode="exact",bits=None,los=None,his=None):
    rr=rms(h); u=h.float()/rr; uh=irot(qdir4(rot(u),clips[i])); uh=uh/rms(uh)
    if radial_mode=="exact": rq=rr
    elif radial_mode=="log": rq=qlog(rr,bits,los[i],his[i])
    else: raise ValueError(radial_mode)
    return (rq*uh).to(h.dtype)

# Exact cascaded calibration in one batched trajectory pass per bitwidth.
# At layer i, the hook observes all calibration tokens after layers <i have already
# been quantized, freezes the requested quantiles, then quantizes layer i before
# the same forward pass continues. Because batch elements are independent, this
# is equivalent to layerwise calibration on the frozen calibration set without
# the previous O(L^2) repeated full-model forwards.
def calibrate_cascaded(bits):
    los=[]; his=[]; hs=[]; t0=time.time()
    for li,layer in enumerate(layers):
        def mkq(i):
            def hook(_m,_inp,o):
                h=hidden(o)
                lr=torch.log(rms(h.detach()).flatten()).cpu()
                los.append(float(torch.quantile(lr,QLO))); his.append(float(torch.quantile(lr,QHI)))
                return repl(o,angular_quant(h,i,"log",bits,los,his))
            return hook
        hs.append(layer.register_forward_hook(mkq(li)))
    with torch.inference_mode(): model(cal_batch,use_cache=False)
    for hdl in hs: hdl.remove()
    if len(los)!=len(layers): raise RuntimeError(f"cascaded calibration captured {len(los)} ranges for {len(layers)} layers")
    print("CASCADE_CAL",json.dumps({"bits":bits,"seconds":time.time()-t0,"logr_width_min":min(h-l for l,h in zip(los,his)),"logr_width_median":float(torch.tensor([h-l for l,h in zip(los,his)]).median()),"logr_width_max":max(h-l for l,h in zip(los,his))}),flush=True)
    return los,his

cascade={}
for b in BITS:
    cascade[b]=calibrate_cascaded(b)

def evaluate(name,mode,radial_mode=None,bits=None,los=None,his=None):
    hs=[]
    for li,layer in enumerate(layers):
        def mk(i):
            def hook(_m,_inp,o):
                h=hidden(o)
                if mode=="dynamic": hh=dynamic_a4(h)
                elif mode=="angular": hh=angular_quant(h,i,radial_mode,bits,los,his)
                else: raise ValueError(mode)
                return repl(o,hh)
            return hook
        hs.append(layer.register_forward_hook(mk(li)))
    total=0.; nt=0; t0=time.time()
    with torch.inference_mode():
        for k in range(0,len(ev),EVAL_BATCH):
            b=torch.cat(ev[k:k+EVAL_BATCH],dim=0)
            o=model(b,labels=b,use_cache=False); n=b.shape[0]*(b.shape[1]-1); total+=float(o.loss)*n; nt+=n
    for hdl in hs:hdl.remove()
    nll=total/nt; out={"name":name,"nll":nll,"ppl":math.exp(nll),"seconds":time.time()-t0}; print("RESULT",json.dumps(out),flush=True); return out

results=[]
results.append(evaluate("hadamard_dynamic_A4_absmax","dynamic"))
results.append(evaluate("angularH_A4_radius_exact","angular","exact"))
for b in BITS:
    results.append(evaluate(f"angularH_A4_radius_log{b}_static","angular","log",b,static_lo,static_hi))
    clo,chi=cascade[b]
    results.append(evaluate(f"angularH_A4_radius_log{b}_cascaded","angular","log",b,clo,chi))

by={x["name"]:x for x in results}; exact=by["angularH_A4_radius_exact"]["nll"]; dyn=by["hadamard_dynamic_A4_absmax"]["nll"]
passes={}
for b in BITS:
    for kind in ("static","cascaded"):
        name=f"angularH_A4_radius_log{b}_{kind}"; nll=by[name]["nll"]
        passes[name]={"pass":bool(nll<=exact+TOL_NLL and nll<dyn),"delta_nll_vs_exact":nll-exact,"delta_nll_vs_dynamic":nll-dyn}
passing_bits=[b for b in BITS if passes[f"angularH_A4_radius_log{b}_cascaded"]["pass"]]
minimum=min(passing_bits) if passing_bits else None
print("FINAL_JSON",json.dumps({
    "experiment":"BITNET_PERSISTENT_RADIAL_ANGULAR_001_GATE004_RADIAL_CODEC",
    "config":{"model":MODEL,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,"eval_prediction_tokens":N_EVAL*(SEQ-1),"seed":SEED,"bits":BITS,"tolerance_nll":TOL_NLL,"radius_range_quantiles":[QLO,QHI],"eval_batch":EVAL_BATCH},
    "gate_rule":"codec NLL <= exact-radius NLL + 0.10 and codec NLL < same-run matched Hadamard dynamic A4",
    "passes":passes,"minimum_passing_cascaded_bits":minimum,"results":results
},sort_keys=True),flush=True)
