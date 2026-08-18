import json, math, random, time, hashlib
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED=26081709
BOOT_SEED=260817091
MODEL="microsoft/bitnet-b1.58-2B-4T"
SEQ=128; N_CAL=8; N_EVAL=32; WIKI_OFFSET=16
QLO=0.005; QHI=0.995; BLOCK=64
RADIAL_BITS=4; SCALE_BITS=4; CODEBOOK=16; EVAL_BATCH=4; BOOT=2000
FRACS=[0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]
PARITY_TOL=1e-5; RETAIN=0.70
EXPECTED={
 "wikitext2_test":{"dynamic_rms":4.638513311743736,"fixed_log4":4.1521482691168785,"oracle_log4":3.9999716952443123},
 "c4_realnewslike_validation":{"dynamic_rms":3.998234026134014,"fixed_log4":3.4902998358011246,"oracle_log4":3.297784671187401},
}
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
print("GATE009_START",json.dumps({
 "model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,
 "evaluation_sequences_per_corpus":N_EVAL,"evaluation_prediction_tokens_per_corpus":N_EVAL*(SEQ-1),
 "radial_bits":RADIAL_BITS,"scale_index_bits":SCALE_BITS,"codebook_entries":CODEBOOK,
 "shape_gain_nominal_bits_per_element":4+(RADIAL_BITS+SCALE_BITS)/2560,
 "raw_adaptive_nominal_bits_per_element":4+SCALE_BITS/2560,
 "bootstrap_resamples":BOOT,"oracle_gain_retention_threshold":RETAIN,
 "rule":"terminal pass iff encodable shape-gain log4 beats exact-RMS dynamic, raw adaptive clipping, and fixed-grid log4 with paired-bootstrap upper95<0 on both corpora; codec fidelity <=+0.10; >=70% Gate008 oracle gain retained; baseline parity holds"
}),flush=True)

tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=False,dtype=torch.bfloat16,device_map={"":"cpu"}); model.eval()
layers=model.model.layers; D=int(model.config.hidden_size)
if D%BLOCK: raise RuntimeError("hidden size not divisible by Hadamard block")
if CODEBOOK!=(1<<SCALE_BITS): raise RuntimeError("codebook size/index bits mismatch")
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
EXPECTED_HASHES={
 "cal_wikitext_validation":"b025c5b984588d46b3c5e6d64144ad90a487073be8357b50d2814abac12ac908",
 "wikitext2_test":"f3645edea43250be6d5c485b2e5de4cb7e376e5f55234e8449b7237d4395ad8e",
 "c4_realnewslike_validation":"bfd2c9c2df055ea0572189e6993dd7f7fdb65ac8b841af526fcd1761af203ec4",
}
if DATA_HASHES!=EXPECTED_HASHES: raise RuntimeError(f"data hash mismatch: {DATA_HASHES}")
print("DATA_READY",json.dumps({"hashes":DATA_HASHES,"prediction_tokens_per_corpus":N_EVAL*(SEQ-1)}),flush=True)

def q4_clip(z,c):
    c=torch.as_tensor(c,dtype=z.dtype,device=z.device)
    step=c/7.0
    return torch.round(torch.clamp(z,-c,c)/step).clamp(-7,7)*step

def oracle_clip_targets(z):
    a=z.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8)
    best_mse=torch.full_like(a,float("inf")); best_c=a.clone()
    for f in FRACS:
        c=a*float(f); q=q4_clip(z,c)
        mse=torch.mean((q-z)*(q-z),dim=-1,keepdim=True)
        take=mse<best_mse
        best_mse=torch.where(take,mse,best_mse)
        best_c=torch.where(take,c,best_c)
    return best_c.squeeze(-1)

def build_codebook(targets):
    t=targets.float().flatten()
    qs=torch.linspace(QLO,QHI,CODEBOOK)
    cb=torch.quantile(t,qs).clamp_min(1e-6)
    for j in range(1,cb.numel()):
        if cb[j]<=cb[j-1]: cb[j]=cb[j-1]+1e-6
    return cb.cpu()

def q4_codebook(z,cb):
    best_mse=torch.full(z.shape[:-1]+(1,),float("inf"),dtype=z.dtype,device=z.device)
    best_q=torch.zeros_like(z)
    for c0 in cb:
        c=float(c0); q=q4_clip(z,c)
        mse=torch.mean((q-z)*(q-z),dim=-1,keepdim=True)
        take=mse<best_mse
        best_mse=torch.where(take,mse,best_mse)
        best_q=torch.where(take,q,best_q)
    return best_q

fixed_samples=[[] for _ in layers]
norm_targets=[[] for _ in layers]
raw_targets=[[] for _ in layers]
handles=[]
for li,layer in enumerate(layers):
    def mk(i):
        def hook(_m,_inp,o):
            h=hidden(o).detach().float(); rr=rms(h)
            zn=rot(h/rr)
            zr=rot(h)
            flat=zn.abs().flatten()
            n=min(32768,flat.numel()); idx=torch.linspace(0,flat.numel()-1,n).long()
            fixed_samples[i].append(flat[idx].cpu())
            norm_targets[i].append(oracle_clip_targets(zn).cpu())
            raw_targets[i].append(oracle_clip_targets(zr).cpu())
            return None
        return hook
    handles.append(layer.register_forward_hook(mk(li)))
t0=time.time()
with torch.inference_mode(): model(cal_batch,use_cache=False)
for h in handles:h.remove()
fixed_clips=[max(float(torch.quantile(torch.cat(bank),0.995)),1.0) for bank in fixed_samples]
norm_codebooks=[build_codebook(torch.cat(bank)) for bank in norm_targets]
raw_codebooks=[build_codebook(torch.cat(bank)) for bank in raw_targets]
print("ANGULAR_CODEBOOK_CAL",json.dumps({
 "seconds":time.time()-t0,
 "fixed_clip_min":min(fixed_clips),"fixed_clip_median":float(torch.tensor(fixed_clips).median()),"fixed_clip_max":max(fixed_clips),
 "norm_cb_min":min(float(cb.min()) for cb in norm_codebooks),"norm_cb_max":max(float(cb.max()) for cb in norm_codebooks),
 "raw_cb_min":min(float(cb.min()) for cb in raw_codebooks),"raw_cb_max":max(float(cb.max()) for cb in raw_codebooks)
}),flush=True)

def qlog(rr,bits,l,h):
    lr=torch.log(rr.float()).clamp(float(l),float(h)); levels=(1<<bits)-1
    if h<=l+1e-12: return torch.exp(torch.full_like(lr,float(l)))
    step=(float(h)-float(l))/levels
    return torch.exp(torch.round((lr-float(l))/step)*step+float(l))

def direction_fixed(h,i):
    rr=rms(h); u=h.float()/rr; uh=irot(q4_clip(rot(u),fixed_clips[i])); return uh/rms(uh)
def direction_shape_cb(h,i):
    rr=rms(h); u=h.float()/rr; uh=irot(q4_codebook(rot(u),norm_codebooks[i])); return uh/rms(uh)
def raw_adaptive_cb(h,i):
    return irot(q4_codebook(rot(h),raw_codebooks[i])).to(h.dtype)

def fixed_quant(h,i,los,his):
    rr=rms(h); rq=qlog(rr,RADIAL_BITS,los[i],his[i]); return (rq*direction_fixed(h,i)).to(h.dtype)
def shape_quant(h,i,radial_mode,los=None,his=None):
    rr=rms(h); uhat=direction_shape_cb(h,i)
    if radial_mode=="exact": rq=rr
    elif radial_mode=="log4": rq=qlog(rr,RADIAL_BITS,los[i],his[i])
    else: raise ValueError(radial_mode)
    return (rq*uhat).to(h.dtype)

def sym_a4(z):
    zf=z.float(); a=zf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8); sc=a/7.0
    return torch.round(zf/sc).clamp(-7,7)*sc
def dynamic_a4_rms_restored(h):
    rr=rms(h); q=irot(sym_a4(rot(h))).float(); q=q/rms(q)
    return (rr*q).to(h.dtype)

fixed_los=[]; fixed_his=[]; hs=[]; t0=time.time()
for li,layer in enumerate(layers):
    def mk_fixed(i):
        def hook(_m,_inp,o):
            h=hidden(o); lr=torch.log(rms(h.detach()).flatten()).cpu()
            fixed_los.append(float(torch.quantile(lr,QLO))); fixed_his.append(float(torch.quantile(lr,QHI)))
            return repl(o,fixed_quant(h,i,fixed_los,fixed_his))
        return hook
    hs.append(layer.register_forward_hook(mk_fixed(li)))
with torch.inference_mode(): model(cal_batch,use_cache=False)
for h in hs:h.remove()
if len(fixed_los)!=len(layers): raise RuntimeError("fixed radial calibration incomplete")
print("FIXED_RADIAL_CAL",json.dumps({"seconds":time.time()-t0}),flush=True)

shape_los=[]; shape_his=[]; hs=[]; t0=time.time()
for li,layer in enumerate(layers):
    def mk_shape(i):
        def hook(_m,_inp,o):
            h=hidden(o); lr=torch.log(rms(h.detach()).flatten()).cpu()
            shape_los.append(float(torch.quantile(lr,QLO))); shape_his.append(float(torch.quantile(lr,QHI)))
            return repl(o,shape_quant(h,i,"log4",shape_los,shape_his))
        return hook
    hs.append(layer.register_forward_hook(mk_shape(li)))
with torch.inference_mode(): model(cal_batch,use_cache=False)
for h in hs:h.remove()
if len(shape_los)!=len(layers): raise RuntimeError("shape radial calibration incomplete")
print("SHAPE_RADIAL_CAL",json.dumps({
 "seconds":time.time()-t0,
 "width_min":min(h-l for l,h in zip(shape_los,shape_his)),
 "width_median":float(torch.tensor([h-l for l,h in zip(shape_los,shape_his)]).median()),
 "width_max":max(h-l for l,h in zip(shape_los,shape_his))
}),flush=True)

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
                if mode=="dynamic_rms": hh=dynamic_a4_rms_restored(h)
                elif mode=="fixed_log4": hh=fixed_quant(h,i,fixed_los,fixed_his)
                elif mode=="raw_cb": hh=raw_adaptive_cb(h,i)
                elif mode=="shape_exact": hh=shape_quant(h,i,"exact")
                elif mode=="shape_log4": hh=shape_quant(h,i,"log4",shape_los,shape_his)
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
    nll=sum(vals)/len(vals)
    out={"corpus":corpus_name,"name":name,"nll":nll,"ppl":math.exp(nll),"sequence_nlls":vals,"seconds":time.time()-t0}
    print("RESULT",json.dumps({k:v for k,v in out.items() if k!="sequence_nlls"}),flush=True); return out

def boot_ci(diff,seed):
    rng=random.Random(seed); n=len(diff); means=[]
    for _ in range(BOOT): means.append(sum(diff[rng.randrange(n)] for __ in range(n))/n)
    means.sort(); return [means[int(0.025*BOOT)],means[min(BOOT-1,int(0.975*BOOT))]]

corpora={"wikitext2_test":wiki_eval,"c4_realnewslike_validation":c4_eval}; results=[]
for cname,segs in corpora.items():
    results.append(evaluate(cname,segs,"hadamard_dynamic_A4_exact_RMS_restored","dynamic_rms"))
    results.append(evaluate(cname,segs,"fixed_shared_angular_A4_log4_radius","fixed_log4"))
    results.append(evaluate(cname,segs,"raw_adaptive_clip_codebook16_A4","raw_cb"))
    results.append(evaluate(cname,segs,"shape_gain_codebook16_A4_exact_radius","shape_exact"))
    results.append(evaluate(cname,segs,"shape_gain_codebook16_A4_log4_radius","shape_log4"))

by={(r["corpus"],r["name"]):r for r in results}; decisions={}
for ci,cname in enumerate(corpora):
    dr=by[(cname,"hadamard_dynamic_A4_exact_RMS_restored")]
    fx=by[(cname,"fixed_shared_angular_A4_log4_radius")]
    rw=by[(cname,"raw_adaptive_clip_codebook16_A4")]
    sx=by[(cname,"shape_gain_codebook16_A4_exact_radius")]
    sl=by[(cname,"shape_gain_codebook16_A4_log4_radius")]
    dd=[a-b for a,b in zip(sl["sequence_nlls"],dr["sequence_nlls"])]
    df=[a-b for a,b in zip(sl["sequence_nlls"],fx["sequence_nlls"])]
    drw=[a-b for a,b in zip(sl["sequence_nlls"],rw["sequence_nlls"])]
    cid=boot_ci(dd,BOOT_SEED+3*ci)
    cif=boot_ci(df,BOOT_SEED+3*ci+1)
    cir=boot_ci(drw,BOOT_SEED+3*ci+2)
    parity=bool(abs(dr["nll"]-EXPECTED[cname]["dynamic_rms"])<=PARITY_TOL and abs(fx["nll"]-EXPECTED[cname]["fixed_log4"])<=PARITY_TOL)
    pass_dyn=bool(sl["nll"]<dr["nll"] and cid[1]<0)
    pass_fix=bool(sl["nll"]<fx["nll"] and cif[1]<0)
    pass_raw=bool(sl["nll"]<rw["nll"] and cir[1]<0)
    codec=bool(sl["nll"]<=sx["nll"]+0.10)
    denom=EXPECTED[cname]["dynamic_rms"]-EXPECTED[cname]["oracle_log4"]
    retention=(EXPECTED[cname]["dynamic_rms"]-sl["nll"])/denom if denom>0 else float("nan")
    retain_pass=bool(retention>=RETAIN)
    decisions[cname]={
      "dynamic_rms_parity_delta":dr["nll"]-EXPECTED[cname]["dynamic_rms"],
      "fixed_log4_parity_delta":fx["nll"]-EXPECTED[cname]["fixed_log4"],
      "baseline_parity":parity,
      "delta_shape_log4_vs_dynamic_rms":sl["nll"]-dr["nll"],"shape_vs_dynamic_bootstrap_95ci":cid,"shape_gain_vs_dynamic":pass_dyn,
      "delta_shape_log4_vs_fixed_log4":sl["nll"]-fx["nll"],"shape_vs_fixed_bootstrap_95ci":cif,"shape_gain_vs_fixed":pass_fix,
      "delta_shape_log4_vs_raw_adaptive":sl["nll"]-rw["nll"],"shape_vs_raw_bootstrap_95ci":cir,"shape_gain_vs_raw_adaptive":pass_raw,
      "codec_delta_log4_vs_exact":sl["nll"]-sx["nll"],"codec_fidelity":codec,
      "oracle_gain_retention_fraction":retention,"oracle_gain_retention":retain_pass,
      "all":bool(parity and pass_dyn and pass_fix and pass_raw and codec and retain_pass)
    }

robust=all(v["all"] for v in decisions.values())
status="GATE009_TERMINAL_METHOD_PASS" if robust else "GATE009_TERMINAL_METHOD_FAIL_STOP_COMPRESSION_METHOD_LANE"
final={
 "experiment":"BITNET_RESIDUAL_SHAPE_GAIN_RATE_DISTORTION_GATE009_RATE_MATCHED_ADAPTIVE",
 "status":status,
 "config":{
   "model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,
   "evaluation_prediction_tokens_per_corpus":N_EVAL*(SEQ-1),"bootstrap_resamples":BOOT,
   "radial_bits":RADIAL_BITS,"scale_index_bits":SCALE_BITS,"codebook_entries":CODEBOOK,
   "shape_gain_nominal_bits_per_element":4+(RADIAL_BITS+SCALE_BITS)/D,
   "raw_adaptive_nominal_bits_per_element":4+SCALE_BITS/D,
   "codebook_quantile_endpoints":[QLO,QHI],"oracle_gain_retention_threshold":RETAIN,
   "data_hashes":DATA_HASHES,"calibration_transfer":"WikiText validation calibration frozen and transferred unchanged to C4",
   "runtime_scale_selection":"per-token reconstruction MSE only; no labels/logits/loss"
 },
 "decisions":decisions,"results":results
}
with open("gate009_results.json","w") as f: json.dump(final,f,indent=2,sort_keys=True)
print("FINAL_JSON",json.dumps({
 "experiment":final["experiment"],"status":status,"config":final["config"],"decisions":decisions,
 "results":[{k:v for k,v in r.items() if k!="sequence_nlls"} for r in results]
},sort_keys=True),flush=True)
