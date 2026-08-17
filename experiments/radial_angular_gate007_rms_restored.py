import json, math, random, time, hashlib
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED=26081707
BOOT_SEED=260817071
MODEL="microsoft/bitnet-b1.58-2B-4T"
SEQ=128; N_CAL=8; N_EVAL=32; WIKI_OFFSET=16
CLIP_Q=0.995; QLO=0.005; QHI=0.995; BLOCK=64
RADIAL_BITS=4; EVAL_BATCH=4; BOOT=2000
PARITY_TOL=1e-5
EXPECTED_DYN={"wikitext2_test":4.8992728888988495,"c4_realnewslike_validation":4.447894737124443}
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
print("GATE007_START",json.dumps({"model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,"evaluation_sequences_per_corpus":N_EVAL,"evaluation_prediction_tokens_per_corpus":N_EVAL*(SEQ-1),"radial_bits":RADIAL_BITS,"bootstrap_resamples":BOOT,"rule":"robust pass iff exact-angular and log4-angular each beat exact-RMS-restored matched dynamic A4 with paired-bootstrap 95% upper CI <0 on both corpora, log4 is within +0.10 NLL of exact angular, and raw dynamic baseline reproduces Gate006 within tolerance"}),flush=True)

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

def ids_hash(segments):
    x=torch.cat(segments,dim=0).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(x).hexdigest()

# Exactly the Gate006 data construction: WikiText validation calibration transferred unchanged to both corpora.
wval=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
wval_text="\n\n".join(x for x in wval["text"] if x and not x.isspace())
wval_ids=tok(wval_text,return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
cal=segments_from_ids(wval_ids,N_CAL,0); cal_batch=torch.cat(cal,dim=0)

wtest=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
wtest_text="\n\n".join(x for x in wtest["text"] if x and not x.isspace())
wtest_ids=tok(wtest_text,return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
wiki_eval=segments_from_ids(wtest_ids,N_EVAL,WIKI_OFFSET)

def collect_c4_ids(required):
    ds=load_dataset("allenai/c4","realnewslike",split="validation",streaming=True)
    parts=[]; count=0
    for row in ds:
        t=row.get("text","")
        if t and not t.isspace(): parts.append(t); count+=1
        if count%16==0:
            ids=tok("\n\n".join(parts),return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
            if ids.numel()>=required: return ids
        if count>=512: break
    ids=tok("\n\n".join(parts),return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
    if ids.numel()<required: raise RuntimeError("insufficient C4 tokens")
    return ids
c4_ids=collect_c4_ids(N_EVAL*SEQ); c4_eval=segments_from_ids(c4_ids,N_EVAL,0)
DATA_HASHES={"cal_wikitext_validation":ids_hash(cal),"wikitext2_test":ids_hash(wiki_eval),"c4_realnewslike_validation":ids_hash(c4_eval)}
print("DATA_READY",json.dumps({"hashes":DATA_HASHES,"prediction_tokens_per_corpus":N_EVAL*(SEQ-1)}),flush=True)

# Fixed/shared angular-grid calibration.
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
clips=[max(float(torch.quantile(torch.cat(zbank),CLIP_Q)),1.0) for zbank in zs]
print("ANGULAR_CAL",json.dumps({"clip_min":min(clips),"clip_median":float(torch.tensor(clips).median()),"clip_max":max(clips)}),flush=True)

def qdir4(z,c):
    step=float(c)/7.0; return torch.round(torch.clamp(z,-float(c),float(c))/step)*step

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
    else: raise ValueError(radial_mode)
    return (rq*uh).to(h.dtype)

def sym_a4(z):
    zf=z.float(); a=zf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8); sc=a/7.0
    return torch.round(zf/sc).clamp(-7,7)*sc

def dynamic_a4(h): return irot(sym_a4(rot(h))).to(h.dtype)
def dynamic_a4_rms_restored(h):
    rr=rms(h); q=irot(sym_a4(rot(h))).float(); q=q/rms(q)
    return (rr*q).to(h.dtype)

# Cascaded 4-bit radial calibration from WikiText validation only.
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
    sl=logits[:,:-1,:].float(); y=ids[:,1:]
    losses=F.cross_entropy(sl.reshape(-1,sl.shape[-1]),y.reshape(-1),reduction="none").reshape(ids.shape[0],-1)
    return losses.mean(dim=1).cpu().tolist()

def evaluate(corpus_name,segments,name,mode):
    hs=[]
    for li,layer in enumerate(layers):
        def mk(i):
            def hook(_m,_inp,o):
                h=hidden(o)
                if mode=="dynamic": hh=dynamic_a4(h)
                elif mode=="dynamic_rms": hh=dynamic_a4_rms_restored(h)
                elif mode=="exact": hh=angular_quant(h,i,"exact")
                elif mode=="log4": hh=angular_quant(h,i,"log4",los,his)
                else: raise ValueError(mode)
                return repl(o,hh)
            return hook
        hs.append(layer.register_forward_hook(mk(li)))
    vals=[]; t0=time.time()
    with torch.inference_mode():
        for k in range(0,len(segments),EVAL_BATCH):
            b=torch.cat(segments[k:k+EVAL_BATCH],dim=0); o=model(b,use_cache=False)
            vals.extend(seq_nll_from_logits(o.logits,b))
    for h in hs:h.remove()
    nll=sum(vals)/len(vals); out={"corpus":corpus_name,"name":name,"nll":nll,"ppl":math.exp(nll),"sequence_nlls":vals,"seconds":time.time()-t0}
    print("RESULT",json.dumps({k:v for k,v in out.items() if k!="sequence_nlls"}),flush=True); return out

def boot_ci(diff,seed):
    rng=random.Random(seed); n=len(diff); means=[]
    for _ in range(BOOT): means.append(sum(diff[rng.randrange(n)] for __ in range(n))/n)
    means.sort(); return [means[int(0.025*BOOT)],means[min(BOOT-1,int(0.975*BOOT))]]

corpora={"wikitext2_test":wiki_eval,"c4_realnewslike_validation":c4_eval}; results=[]
for cname,segs in corpora.items():
    results.append(evaluate(cname,segs,"hadamard_dynamic_A4_absmax","dynamic"))
    results.append(evaluate(cname,segs,"hadamard_dynamic_A4_exact_RMS_restored","dynamic_rms"))
    results.append(evaluate(cname,segs,"angular_A4_exact_radius","exact"))
    results.append(evaluate(cname,segs,"angular_A4_log4_radius","log4"))

by={(r["corpus"],r["name"]):r for r in results}; decisions={}
for ci,cname in enumerate(corpora):
    raw=by[(cname,"hadamard_dynamic_A4_absmax")]; dr=by[(cname,"hadamard_dynamic_A4_exact_RMS_restored")]; ex=by[(cname,"angular_A4_exact_radius")]; lg=by[(cname,"angular_A4_log4_radius")]
    dex=[a-b for a,b in zip(ex["sequence_nlls"],dr["sequence_nlls"])]
    dlg=[a-b for a,b in zip(lg["sequence_nlls"],dr["sequence_nlls"])]
    ciex=boot_ci(dex,BOOT_SEED+2*ci); cilg=boot_ci(dlg,BOOT_SEED+2*ci+1)
    exact_pass=bool(ex["nll"]<dr["nll"] and ciex[1]<0)
    log4_pass=bool(lg["nll"]<dr["nll"] and cilg[1]<0)
    codec=bool(lg["nll"]<=ex["nll"]+0.10)
    parity=bool(abs(raw["nll"]-EXPECTED_DYN[cname])<=PARITY_TOL)
    decisions[cname]={"raw_dynamic_parity_delta":raw["nll"]-EXPECTED_DYN[cname],"raw_dynamic_parity":parity,"delta_exact_angular_vs_rms_restored_dynamic":ex["nll"]-dr["nll"],"exact_angular_bootstrap_95ci":ciex,"exact_angular_independent_value":exact_pass,"delta_log4_angular_vs_rms_restored_dynamic":lg["nll"]-dr["nll"],"log4_angular_bootstrap_95ci":cilg,"practical_log4_independent_value":log4_pass,"codec_delta_log4_vs_exact":lg["nll"]-ex["nll"],"codec_fidelity":codec,"all":bool(parity and exact_pass and log4_pass and codec)}
robust=all(v["all"] for v in decisions.values())
status="GATE007_RMS_RESTORED_HOSTILE_CONTROL_PASS" if robust else "GATE007_RMS_RESTORED_HOSTILE_CONTROL_FAIL"
final={"experiment":"BITNET_RESIDUAL_SHAPE_GAIN_RATE_DISTORTION_GATE007_RMS_RESTORED","status":status,"config":{"model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,"evaluation_prediction_tokens_per_corpus":N_EVAL*(SEQ-1),"radial_bits":RADIAL_BITS,"bootstrap_resamples":BOOT,"data_hashes":DATA_HASHES,"calibration_transfer":"WikiText validation calibration frozen and transferred unchanged to C4"},"decisions":decisions,"results":results}
with open("gate007_results.json","w") as f: json.dump(final,f,indent=2,sort_keys=True)
print("FINAL_JSON",json.dumps({"experiment":final["experiment"],"status":status,"config":final["config"],"decisions":decisions,"results":[{k:v for k,v in r.items() if k!="sequence_nlls"} for r in results]},sort_keys=True),flush=True)
