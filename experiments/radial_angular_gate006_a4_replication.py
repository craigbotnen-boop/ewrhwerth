import json, math, random, time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED=26081706
BOOT_SEED=260817061
MODEL="microsoft/bitnet-b1.58-2B-4T"
SEQ=128; N_CAL=8; N_EVAL=32; WIKI_OFFSET=16
CLIP_Q=0.995; QLO=0.005; QHI=0.995; BLOCK=64
RADIAL_BITS=4; EVAL_BATCH=4; BOOT=2000
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
print("GATE006_START", json.dumps({
    "model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,
    "evaluation_sequences_per_corpus":N_EVAL,"evaluation_prediction_tokens_per_corpus":N_EVAL*(SEQ-1),
    "wiki_eval_offset_sequences":WIKI_OFFSET,"radial_bits":RADIAL_BITS,
    "nominal_bits_per_element":4+RADIAL_BITS/2560,"bootstrap_resamples":BOOT,
    "rule":"robust pass iff on both corpora log4 A4 beats matched dynamic A4 with paired-bootstrap 95% upper CI <0, shifted-radius penalty >=0.25 NLL, and log4 is within +0.10 NLL of exact radius"
}), flush=True)

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

def segments_from_ids(ids,n,offset=0):
    out=[]
    for i in range(offset,offset+n):
        s=ids[i*SEQ:(i+1)*SEQ]
        if s.numel()!=SEQ: raise RuntimeError("insufficient tokens")
        out.append(s.unsqueeze(0))
    return out

# Calibration is from WikiText validation only, and is transferred unchanged to both evaluation corpora.
wval=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
wval_text="\n\n".join(x for x in wval["text"] if x and not x.isspace())
wval_ids=tok(wval_text,return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
cal=segments_from_ids(wval_ids,N_CAL,0); cal_batch=torch.cat(cal,dim=0)

wtest=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
wtest_text="\n\n".join(x for x in wtest["text"] if x and not x.isspace())
wtest_ids=tok(wtest_text,return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
wiki_eval=segments_from_ids(wtest_ids,N_EVAL,WIKI_OFFSET)

# C4 second-corpus evaluation: deterministic head of realnewslike validation, no C4 calibration.
def collect_c4_ids(required):
    ds=load_dataset("allenai/c4","realnewslike",split="validation",streaming=True)
    parts=[]; count=0
    for row in ds:
        t=row.get("text","")
        if t and not t.isspace():
            parts.append(t); count+=1
        if count%16==0:
            ids=tok("\n\n".join(parts),return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
            if ids.numel()>=required: return ids
        if count>=512: break
    ids=tok("\n\n".join(parts),return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
    if ids.numel()<required: raise RuntimeError(f"C4 stream only yielded {ids.numel()} tokens; need {required}")
    return ids
c4_ids=collect_c4_ids(N_EVAL*SEQ)
c4_eval=segments_from_ids(c4_ids,N_EVAL,0)
print("DATA_READY",json.dumps({"wiki_eval_sequences":len(wiki_eval),"c4_eval_sequences":len(c4_eval),"wiki_prediction_tokens":len(wiki_eval)*(SEQ-1),"c4_prediction_tokens":len(c4_eval)*(SEQ-1)}),flush=True)

# Native calibration of fixed angular clips.
zs=[[] for _ in layers]; handles=[]
for li,layer in enumerate(layers):
    def mk(i):
        def hook(_m,_inp,o):
            h=hidden(o).detach(); rr=rms(h); z=rot(h.float()/rr).abs().flatten()
            n=min(32768,z.numel()); idx=torch.linspace(0,z.numel()-1,n).long(); zs[i].append(z[idx].cpu()); return None
        return hook
    handles.append(layer.register_forward_hook(mk(li)))
with torch.inference_mode(): model(cal_batch,use_cache=False)
for h in handles:h.remove()
clips=[]
for zbank in zs:
    z=torch.cat(zbank); clips.append(max(float(torch.quantile(z,CLIP_Q)),1.0))
print("ANGULAR_CAL",json.dumps({"clip_min":min(clips),"clip_median":float(torch.tensor(clips).median()),"clip_max":max(clips)}),flush=True)

def qdir4(z,c):
    step=float(c)/7.0
    return torch.round(torch.clamp(z,-float(c),float(c))/step)*step

def qlog(rr,bits,l,h):
    lr=torch.log(rr.float()).clamp(float(l),float(h)); levels=(1<<bits)-1
    if h<=l+1e-12: return torch.exp(torch.full_like(lr,float(l)))
    step=(float(h)-float(l))/levels
    return torch.exp(torch.round((lr-float(l))/step)*step+float(l))

def angular_direction(h,i):
    rr=rms(h); u=h.float()/rr; uh=irot(qdir4(rot(u),clips[i])); return uh/rms(uh)

def angular_quant(h,i,radial_mode,los=None,his=None):
    rr=rms(h); uh=angular_direction(h,i)
    if radial_mode=="exact": rq=rr
    elif radial_mode=="log4": rq=qlog(rr,RADIAL_BITS,los[i],his[i])
    elif radial_mode=="shifted_log4": rq=torch.roll(qlog(rr,RADIAL_BITS,los[i],his[i]),shifts=1,dims=-2)
    else: raise ValueError(radial_mode)
    return (rq*uh).to(h.dtype)

def sym_a4(z):
    zf=z.float(); a=zf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8); sc=a/7.0
    return torch.round(zf/sc).clamp(-7,7)*sc
def dynamic_a4(h): return irot(sym_a4(rot(h))).to(h.dtype)

# Cascaded 4-bit radial calibration on WikiText validation only.
los=[]; his=[]; hs=[]; t0=time.time()
for li,layer in enumerate(layers):
    def mkq(i):
        def hook(_m,_inp,o):
            h=hidden(o); lr=torch.log(rms(h.detach()).flatten()).cpu()
            los.append(float(torch.quantile(lr,QLO))); his.append(float(torch.quantile(lr,QHI)))
            return repl(o,angular_quant(h,i,"log4",los,his))
        return hook
    hs.append(layer.register_forward_hook(mkq(li)))
with torch.inference_mode(): model(cal_batch,use_cache=False)
for h in hs:h.remove()
if len(los)!=len(layers): raise RuntimeError("cascaded radial calibration incomplete")
print("RADIAL_CAL",json.dumps({"seconds":time.time()-t0,"width_min":min(h-l for l,h in zip(los,his)),"width_median":float(torch.tensor([h-l for l,h in zip(los,his)]).median()),"width_max":max(h-l for l,h in zip(los,his))}),flush=True)

def seq_nll_from_logits(logits,ids):
    # returns one NLL per sequence
    sl=logits[:,:-1,:].float(); y=ids[:,1:]
    losses=F.cross_entropy(sl.reshape(-1,sl.shape[-1]),y.reshape(-1),reduction="none").reshape(ids.shape[0],-1)
    return losses.mean(dim=1).cpu().tolist()

def evaluate(corpus_name,segments,name,mode):
    hs=[]
    if mode!="native":
        for li,layer in enumerate(layers):
            def mk(i):
                def hook(_m,_inp,o):
                    h=hidden(o)
                    if mode=="dynamic": hh=dynamic_a4(h)
                    elif mode=="exact": hh=angular_quant(h,i,"exact")
                    elif mode=="log4": hh=angular_quant(h,i,"log4",los,his)
                    elif mode=="shifted": hh=angular_quant(h,i,"shifted_log4",los,his)
                    else: raise ValueError(mode)
                    return repl(o,hh)
                return hook
            hs.append(layer.register_forward_hook(mk(li)))
    vals=[]; t0=time.time()
    with torch.inference_mode():
        for k in range(0,len(segments),EVAL_BATCH):
            b=torch.cat(segments[k:k+EVAL_BATCH],dim=0)
            o=model(b,use_cache=False)
            vals.extend(seq_nll_from_logits(o.logits,b))
    for h in hs:h.remove()
    nll=sum(vals)/len(vals); out={"corpus":corpus_name,"name":name,"nll":nll,"ppl":math.exp(nll),"sequence_nlls":vals,"seconds":time.time()-t0}
    print("RESULT",json.dumps({k:v for k,v in out.items() if k!="sequence_nlls"}),flush=True); return out

def boot_ci(diff,seed):
    rng=random.Random(seed); n=len(diff); means=[]
    for _ in range(BOOT):
        means.append(sum(diff[rng.randrange(n)] for __ in range(n))/n)
    means.sort(); lo=means[int(0.025*BOOT)]; hi=means[min(BOOT-1,int(0.975*BOOT))]
    return [lo,hi]

corpora={"wikitext2_test":wiki_eval,"c4_realnewslike_validation":c4_eval}
results=[]
for cname,segs in corpora.items():
    results.append(evaluate(cname,segs,"native","native"))
    results.append(evaluate(cname,segs,"hadamard_dynamic_A4_absmax","dynamic"))
    results.append(evaluate(cname,segs,"angular_A4_exact_radius","exact"))
    results.append(evaluate(cname,segs,"angular_A4_log4_radius","log4"))
    results.append(evaluate(cname,segs,"angular_A4_log4_shifted_radius","shifted"))

by={(r["corpus"],r["name"]):r for r in results}; decisions={}
for ci,(cname,segs) in enumerate(corpora.items()):
    dyn=by[(cname,"hadamard_dynamic_A4_absmax")]; exact=by[(cname,"angular_A4_exact_radius")]; log4=by[(cname,"angular_A4_log4_radius")]; shift=by[(cname,"angular_A4_log4_shifted_radius")]
    d=[a-b for a,b in zip(log4["sequence_nlls"],dyn["sequence_nlls"])]
    ci95=boot_ci(d,BOOT_SEED+ci)
    delta=sum(d)/len(d); shift_pen=shift["nll"]-log4["nll"]; codec_delta=log4["nll"]-exact["nll"]
    primary=bool(log4["nll"]<dyn["nll"] and ci95[1]<0)
    radial=bool(shift_pen>=0.25); codec=bool(codec_delta<=0.10)
    decisions[cname]={"delta_nll_log4_vs_dynamic":delta,"paired_bootstrap_95ci":ci95,"primary_replication":primary,"shift_penalty_nll":shift_pen,"radius_specificity":radial,"codec_delta_nll_vs_exact":codec_delta,"codec_fidelity":codec,"all_three":bool(primary and radial and codec)}
robust=all(v["all_three"] for v in decisions.values())
primary_count=sum(v["primary_replication"] for v in decisions.values())
status="GATE006_ROBUST_A4_REPLICATION_PASS" if robust else ("GATE006_PARTIAL_A4_REPLICATION" if primary_count else "GATE006_A4_REPLICATION_FAIL")
final={"experiment":"BITNET_RESIDUAL_SHAPE_GAIN_RATE_DISTORTION_GATE006_A4_REPLICATION","status":status,"config":{"model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,"evaluation_sequences_per_corpus":N_EVAL,"evaluation_prediction_tokens_per_corpus":N_EVAL*(SEQ-1),"wiki_eval_offset_sequences":WIKI_OFFSET,"radial_bits":RADIAL_BITS,"nominal_bits_per_element":4+RADIAL_BITS/D,"bootstrap_resamples":BOOT,"bootstrap_seed":BOOT_SEED,"calibration_transfer":"WikiText validation calibration frozen and used unchanged for C4"},"decisions":decisions,"results":results}
with open("gate006_results.json","w") as f: json.dump(final,f,indent=2,sort_keys=True)
print("FINAL_JSON",json.dumps({"experiment":final["experiment"],"status":status,"config":final["config"],"decisions":decisions,"results":[{k:v for k,v in r.items() if k!="sequence_nlls"} for r in results]},sort_keys=True),flush=True)
