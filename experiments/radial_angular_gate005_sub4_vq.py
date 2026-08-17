import json, math, random, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED=26081705
MODEL="microsoft/bitnet-b1.58-2B-4T"
SEQ=128; N_CAL=2; N_EVAL=8; EVAL_BATCH=4
BLOCK=64; CLIP_Q=0.995; QLO=0.005; QHI=0.995
RADIAL_BITS=4
GROUP=4; K=256; SAMPLE_PER_LAYER=2048
SYN_N=32768; KMEANS_ITERS=8; NN_CHUNK=16384
SHIFT_MARGIN=0.25; VQ_PREEMPT_MARGIN=0.10
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

print("GATE005_START",json.dumps({
    "model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,
    "eval_prediction_tokens":N_EVAL*(SEQ-1),"radial_bits":RADIAL_BITS,
    "scalar_direction_bits":[4,3,2],"vq_group":GROUP,"vq_K":K,
    "vq_rate_bits_per_element":math.log2(K)/GROUP,"eval_batch":EVAL_BATCH,
    "rules":{
      "sub4_success":"candidate NLL < matched dynamic A4 and shifted-radius NLL - candidate NLL >= 0.25",
      "vq_preemption":"best VQ A2 beats scalar A2 by >= 0.10 NLL"
    }
}),flush=True)

# ----- Model / data -----
tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=False,dtype=torch.bfloat16,device_map={"":"cpu"}); model.eval()
layers=model.model.layers; D=int(model.config.hidden_size)
if D%BLOCK or D%GROUP: raise RuntimeError("hidden size incompatible with frozen block/group sizes")
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

# ----- Native calibration: one shared scalar clip per layer + pooled VQ groups -----
clip_banks=[[] for _ in layers]; group_banks=[]; hs=[]
for li,layer in enumerate(layers):
    def mk(i):
        def hook(_m,_inp,o):
            h=hidden(o).detach(); rr=rms(h); z=rot(h.float()/rr)
            za=z.abs().flatten(); n=min(8192,za.numel()); idx=torch.linspace(0,za.numel()-1,n).long()
            clip_banks[i].append(za[idx].cpu())
            g=z.reshape(-1,GROUP)
            ng=min(SAMPLE_PER_LAYER,g.shape[0]); gi=torch.linspace(0,g.shape[0]-1,ng).long()
            group_banks.append(g[gi].cpu())
            return None
        return hook
    hs.append(layer.register_forward_hook(mk(li)))
with torch.inference_mode(): model(cal_batch,use_cache=False)
for h in hs:h.remove()
clips=[]
for bank in clip_banks:
    v=torch.cat(bank); clips.append(max(float(torch.quantile(v,CLIP_Q)),1.0))
cal_groups=torch.cat(group_banks,dim=0).float().contiguous()
print("NATIVE_CAL",json.dumps({
    "clip_min":min(clips),"clip_median":float(torch.tensor(clips).median()),"clip_max":max(clips),
    "vq_calibration_groups":int(cal_groups.shape[0]),"vq_group_dim":GROUP
}),flush=True)

# ----- Deterministic full 4-D VQ codebooks -----
def nearest_idx(x,c,chunk=NN_CHUNK):
    c=c.float(); c2=(c*c).sum(dim=1)
    outs=[]
    for s in range(0,x.shape[0],chunk):
        xx=x[s:s+chunk].float(); d=(xx*xx).sum(dim=1,keepdim=True)+c2.unsqueeze(0)-2.0*(xx@c.T)
        outs.append(torch.argmin(d,dim=1))
    return torch.cat(outs,dim=0)

def kmeans_pp(data,k,iters,seed):
    data=data.float().contiguous(); gen=torch.Generator(device="cpu").manual_seed(seed)
    first=int(torch.randint(0,data.shape[0],(1,),generator=gen))
    centers=[data[first].clone()]
    min_d=((data-centers[0])**2).sum(dim=1)
    for _ in range(1,k):
        w=min_d.clamp_min(1e-12); idx=int(torch.multinomial(w/w.sum(),1,generator=gen))
        cc=data[idx].clone(); centers.append(cc)
        min_d=torch.minimum(min_d,((data-cc)**2).sum(dim=1))
    c=torch.stack(centers,dim=0)
    for it in range(iters):
        lab=nearest_idx(data,c)
        sums=torch.zeros_like(c); sums.index_add_(0,lab,data)
        counts=torch.bincount(lab,minlength=k).float().unsqueeze(1)
        empty=(counts.squeeze(1)==0)
        c=sums/counts.clamp_min(1.0)
        if bool(empty.any()):
            nempty=int(empty.sum()); ridx=torch.randint(0,data.shape[0],(nempty,),generator=gen); c[empty]=data[ridx]
        inertia=0.0
        for s in range(0,data.shape[0],NN_CHUNK):
            xx=data[s:s+NN_CHUNK]; ll=lab[s:s+NN_CHUNK]; inertia+=float(((xx-c[ll])**2).sum())
        print("KMEANS",json.dumps({"seed":seed,"iter":it+1,"n":int(data.shape[0]),"inertia_per_vector":inertia/data.shape[0]}),flush=True)
    return c.contiguous()

gen=torch.Generator(device="cpu").manual_seed(SEED+101)
syn_data=torch.randn(SYN_N,GROUP,generator=gen)
t0=time.time(); CB_SYN=kmeans_pp(syn_data,K,KMEANS_ITERS,SEED+102); print("CODEBOOK_READY",json.dumps({"name":"synthetic_gaussian","seconds":time.time()-t0}),flush=True)
t0=time.time(); CB_CAL=kmeans_pp(cal_groups,K,KMEANS_ITERS,SEED+103); print("CODEBOOK_READY",json.dumps({"name":"calibration_kmeans","seconds":time.time()-t0}),flush=True)

# ----- Quantizers -----
def sym_dynamic_a4(z):
    zf=z.float(); qmax=7; a=zf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8); sc=a/qmax
    return torch.round(zf/sc).clamp(-qmax,qmax)*sc

def dynamic_a4(h): return irot(sym_dynamic_a4(rot(h))).to(h.dtype)

def qscalar(z,c,bits):
    qmax=(1<<(bits-1))-1
    step=float(c)/qmax
    return torch.round(torch.clamp(z,-float(c),float(c))/step).clamp(-qmax,qmax)*step

def qvq(z,cb):
    shp=z.shape; g=z.float().reshape(-1,GROUP); lab=nearest_idx(g,cb); return cb[lab].reshape(shp)

def qlog(rr,bits,l,h):
    lr=torch.log(rr.float()).clamp(float(l),float(h)); levels=(1<<bits)-1
    if h<=l+1e-12: return torch.exp(torch.full_like(lr,float(l)))
    step=(float(h)-float(l))/levels
    return torch.exp(torch.round((lr-float(l))/step)*step+float(l))

def direction_unit(h,i,kind):
    rr=rms(h); z=rot(h.float()/rr)
    if kind=="scalar4": zq=qscalar(z,clips[i],4)
    elif kind=="scalar3": zq=qscalar(z,clips[i],3)
    elif kind=="scalar2": zq=qscalar(z,clips[i],2)
    elif kind=="vq_syn": zq=qvq(z,CB_SYN)
    elif kind=="vq_cal": zq=qvq(z,CB_CAL)
    else: raise ValueError(kind)
    uh=irot(zq); return uh/rms(uh)

def candidate_quant(h,i,kind,los=None,his=None,radial="log4",shifted=False):
    rr=rms(h); uh=direction_unit(h,i,kind)
    if radial=="exact": rq=rr
    elif radial=="log4": rq=qlog(rr,RADIAL_BITS,los[i],his[i])
    else: raise ValueError(radial)
    if shifted:
        if rq.ndim<3: raise RuntimeError("shifted-radius control requires [batch,token,1]")
        rq=torch.roll(rq,shifts=1,dims=1)
    return (rq*uh).to(h.dtype)

# Candidate-specific cascaded radial range calibration, frozen only on calibration tokens.
def calibrate_cascaded(kind):
    los=[]; his=[]; handles=[]; t0=time.time()
    for li,layer in enumerate(layers):
        def mk(i):
            def hook(_m,_inp,o):
                h=hidden(o); lr=torch.log(rms(h.detach()).flatten()).cpu()
                los.append(float(torch.quantile(lr,QLO))); his.append(float(torch.quantile(lr,QHI)))
                return repl(o,candidate_quant(h,i,kind,los,his,"log4",False))
            return hook
        handles.append(layer.register_forward_hook(mk(li)))
    with torch.inference_mode(): model(cal_batch,use_cache=False)
    for hh in handles:hh.remove()
    if len(los)!=len(layers): raise RuntimeError(f"{kind}: captured {len(los)} ranges")
    widths=[h-l for l,h in zip(los,his)]
    print("CASCADE_CAL",json.dumps({"kind":kind,"seconds":time.time()-t0,"width_min":min(widths),"width_median":float(torch.tensor(widths).median()),"width_max":max(widths)}),flush=True)
    return los,his

kinds=["scalar4","scalar3","scalar2","vq_syn","vq_cal"]
ranges={k:calibrate_cascaded(k) for k in kinds}

# ----- Evaluation -----
def evaluate(name,mode,kind=None,radial="log4",shifted=False):
    handles=[]
    los,his=(ranges[kind] if kind is not None else (None,None))
    for li,layer in enumerate(layers):
        def mk(i):
            def hook(_m,_inp,o):
                h=hidden(o)
                if mode=="dynamic": hh=dynamic_a4(h)
                elif mode=="candidate": hh=candidate_quant(h,i,kind,los,his,radial,shifted)
                else: raise ValueError(mode)
                return repl(o,hh)
            return hook
        handles.append(layer.register_forward_hook(mk(li)))
    total=0.0; nt=0; t0=time.time()
    with torch.inference_mode():
        for k0 in range(0,len(ev),EVAL_BATCH):
            b=torch.cat(ev[k0:k0+EVAL_BATCH],dim=0); o=model(b,labels=b,use_cache=False)
            n=b.shape[0]*(b.shape[1]-1); total+=float(o.loss)*n; nt+=n
    for hh in handles:hh.remove()
    nll=total/nt; out={"name":name,"nll":nll,"ppl":math.exp(nll),"seconds":time.time()-t0}; print("RESULT",json.dumps(out),flush=True); return out

results=[]
results.append(evaluate("hadamard_dynamic_A4_absmax","dynamic"))
results.append(evaluate("scalar_A4_log4_radius","candidate","scalar4"))
results.append(evaluate("scalar_A3_log4_radius","candidate","scalar3"))
results.append(evaluate("scalar_A2_log4_radius","candidate","scalar2"))
results.append(evaluate("vq_syn_A2_log4_radius","candidate","vq_syn"))
results.append(evaluate("vq_cal_A2_log4_radius","candidate","vq_cal"))
results.append(evaluate("vq_syn_A2_exact_radius","candidate","vq_syn","exact"))

by={r["name"]:r for r in results}; dyn=by["hadamard_dynamic_A4_absmax"]["nll"]
a2_names=["scalar_A2_log4_radius","vq_syn_A2_log4_radius","vq_cal_A2_log4_radius"]
best_a2=min(a2_names,key=lambda n:by[n]["nll"])
kind_for_name={"scalar_A2_log4_radius":"scalar2","vq_syn_A2_log4_radius":"vq_syn","vq_cal_A2_log4_radius":"vq_cal","scalar_A3_log4_radius":"scalar3"}
shift_name=best_a2+"__shifted_radius"
shift_res=evaluate(shift_name,"candidate",kind_for_name[best_a2],"log4",True); results.append(shift_res); by[shift_name]=shift_res

sub4_names=["scalar_A3_log4_radius"]+a2_names
better=[n for n in sub4_names if by[n]["nll"]<dyn]
best_sub4=min(better,key=lambda n:by[n]["nll"]) if better else None
if best_sub4 is not None and best_sub4!=best_a2:
    extra=best_sub4+"__shifted_radius"; er=evaluate(extra,"candidate",kind_for_name[best_sub4],"log4",True); results.append(er); by[extra]=er

# ----- Frozen decisions -----
shift_deltas={}
for n in [best_a2]+(([best_sub4] if best_sub4 and best_sub4!=best_a2 else [])):
    sn=n+"__shifted_radius"; shift_deltas[n]=by[sn]["nll"]-by[n]["nll"]
sub4_causal_candidates=[n for n,delta in shift_deltas.items() if by[n]["nll"]<dyn and delta>=SHIFT_MARGIN]
scalar_a2_pass=by["scalar_A2_log4_radius"]["nll"]<dyn
best_vq=min(["vq_syn_A2_log4_radius","vq_cal_A2_log4_radius"],key=lambda n:by[n]["nll"])
vq_a2_pass=by[best_vq]["nll"]<dyn
representation_generalization=bool(scalar_a2_pass and vq_a2_pass)
vq_preemption=bool(by[best_vq]["nll"]<=by["scalar_A2_log4_radius"]["nll"]-VQ_PREEMPT_MARGIN)
sub4_success=bool(sub4_causal_candidates)
stop=not any(by[n]["nll"]<dyn for n in sub4_names)

final={
  "experiment":"BITNET_RESIDUAL_SHAPE_GAIN_RATE_DISTORTION_GATE005",
  "config":{
    "model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,
    "evaluation_prediction_tokens":N_EVAL*(SEQ-1),"radial_bits":RADIAL_BITS,
    "radial_overhead_bits_per_element":RADIAL_BITS/D,
    "scalar_rates_bits_per_element":{"A4":4+RADIAL_BITS/D,"A3":3+RADIAL_BITS/D,"A2":2+RADIAL_BITS/D},
    "vq_rate_bits_per_element":math.log2(K)/GROUP+RADIAL_BITS/D,
    "vq_codebook_bytes_float32":K*GROUP*4,
    "vq_group":GROUP,"vq_K":K,"kmeans_iters":KMEANS_ITERS
  },
  "results":results,
  "decisions":{
    "best_a2":best_a2,"best_vq_a2":best_vq,"best_sub4":best_sub4,
    "shift_deltas_nll":shift_deltas,
    "sub4_causal_candidates":sub4_causal_candidates,
    "sub4_success":sub4_success,
    "a2_scalar_success":scalar_a2_pass,
    "a2_vq_success":vq_a2_pass,
    "representation_generalization":representation_generalization,
    "vq_preemption":vq_preemption,
    "stop_sub4":stop
  }
}
with open("gate005_results.json","w") as f: json.dump(final,f,indent=2,sort_keys=True)
print("FINAL_JSON",json.dumps(final,sort_keys=True),flush=True)
