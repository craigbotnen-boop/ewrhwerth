import json, math, random, time, hashlib, types
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.bitnet import BitLinear

SEED=26081810
BOOT_SEED=260818101
MODEL="microsoft/bitnet-b1.58-2B-4T"
SEQ=128; N_CAL=8; N_EVAL=16; N_HASH=32; WIKI_OFFSET=16
QLO=0.005; QHI=0.995; CODEBOOK=16; SCALE_BITS=4
MSE_CH=128; EVAL_BATCH=4; BOOT=2000
FRACS=[0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]
EXPECTED_HASHES={
 "cal_wikitext_validation":"b025c5b984588d46b3c5e6d64144ad90a487073be8357b50d2814abac12ac908",
 "wikitext2_test":"f3645edea43250be6d5c485b2e5de4cb7e376e5f55234e8449b7237d4395ad8e",
 "c4_realnewslike_validation":"bfd2c9c2df055ea0572189e6993dd7f7fdb65ac8b841af526fcd1761af203ec4",
}
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
print("GATE010_START",json.dumps({
 "model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,
 "evaluation_sequences_per_corpus":N_EVAL,
 "evaluation_prediction_tokens_per_corpus":N_EVAL*(SEQ-1),
 "adaptive_codebook_entries_per_BitLinear":CODEBOOK,"scale_index_bits":SCALE_BITS,
 "selection_mse_channel_sample":MSE_CH,"bootstrap_resamples":BOOT,
 "rule":"pass iff adaptive codebook A4 beats native-site dynamic absmax A4 with paired-bootstrap upper95<0 and remains within +0.50 NLL of native A8 on both corpora; data/model parity holds"
}),flush=True)

tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(
    MODEL,trust_remote_code=False,dtype=torch.bfloat16,device_map={"":"cpu"}
)
model.eval()
bitmods=[(n,m) for n,m in model.named_modules() if isinstance(m,BitLinear)]
if not bitmods:
    raise RuntimeError("No BitLinear modules found")
if CODEBOOK != (1<<SCALE_BITS):
    raise RuntimeError("codebook size/index bits mismatch")
print("MODEL_INFO",json.dumps({
 "bitlinear_modules":len(bitmods),
 "hidden_size":int(model.config.hidden_size),
 "rms_norm_eps":float(model.config.rms_norm_eps),
 "quantization_config":str(getattr(model.config,"quantization_config",None)),
 "first_modules":[n for n,_ in bitmods[:8]]
}),flush=True)

def segments_from_ids(ids,n,offset=0):
    out=[]
    for i in range(offset,offset+n):
        s=ids[i*SEQ:(i+1)*SEQ]
        if s.numel()!=SEQ:
            raise RuntimeError("insufficient tokens")
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
wiki_hash_segments=segments_from_ids(wtest_ids,N_HASH,WIKI_OFFSET)
wiki_eval=wiki_hash_segments[:N_EVAL]

def collect_c4_ids(required):
    ds=load_dataset("allenai/c4","realnewslike",split="validation",streaming=True)
    parts=[]; count=0
    for row in ds:
        t=row.get("text","")
        if t and not t.isspace():
            parts.append(t); count+=1
        if count%16==0:
            ids=tok("\n\n".join(parts),return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
            if ids.numel()>=required:
                return ids
        if count>=512:
            break
    ids=tok("\n\n".join(parts),return_tensors="pt",add_special_tokens=False,truncation=False)["input_ids"][0]
    if ids.numel()<required:
        raise RuntimeError("insufficient C4 tokens")
    return ids

c4_ids=collect_c4_ids(N_HASH*SEQ)
c4_hash_segments=segments_from_ids(c4_ids,N_HASH,0)
c4_eval=c4_hash_segments[:N_EVAL]
DATA_HASHES={
 "cal_wikitext_validation":ids_hash(cal),
 "wikitext2_test":ids_hash(wiki_hash_segments),
 "c4_realnewslike_validation":ids_hash(c4_hash_segments),
}
if DATA_HASHES != EXPECTED_HASHES:
    raise RuntimeError(f"data hash mismatch: {DATA_HASHES}")
print("DATA_READY",json.dumps({
 "full32_hashes":DATA_HASHES,
 "evaluated_prefix_sequences":N_EVAL,
 "prediction_tokens_per_corpus":N_EVAL*(SEQ-1)
}),flush=True)

def sample_channels(x):
    d=x.shape[-1]
    n=min(MSE_CH,d)
    if n==d:
        return x.float()
    idx=torch.linspace(0,d-1,n,device=x.device).long()
    return x.float().index_select(-1,idx)

def oracle_target_clip(x):
    xf=x.detach().float()
    a=xf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-6)
    xs=sample_channels(xf)
    best_mse=torch.full(a.shape,float("inf"),dtype=torch.float32,device=xf.device)
    best_c=a.clone()
    for f in FRACS:
        c=a*float(f)
        scale=7.0/c
        q=torch.round(xs*scale).clamp(-8,7)
        recon=q/scale
        mse=torch.mean((recon-xs)**2,dim=-1,keepdim=True)
        take=mse<best_mse
        best_mse=torch.where(take,mse,best_mse)
        best_c=torch.where(take,c,best_c)
    return best_c.squeeze(-1).cpu()

def build_codebook(target):
    t=target.float().flatten()
    qs=torch.linspace(QLO,QHI,CODEBOOK)
    cb=torch.quantile(t,qs).clamp_min(1e-6)
    for j in range(1,cb.numel()):
        if cb[j] <= cb[j-1]:
            cb[j]=cb[j-1]+1e-6
    return cb.cpu()

# Calibrate a separate absolute clipping codebook for every actual BitLinear quantization site.
orig_methods=[m.activation_quant for _,m in bitmods]
target_banks=[[] for _ in bitmods]
for i,(_,m) in enumerate(bitmods):
    orig=orig_methods[i]
    def make_collect(ii,orig_bound):
        def aq(self,x,num_bits=8):
            target_banks[ii].append(oracle_target_clip(x))
            return orig_bound(x,num_bits=8)
        return aq
    m.activation_quant=types.MethodType(make_collect(i,orig),m)

t0=time.time()
with torch.inference_mode():
    model(cal_batch,use_cache=False)
for (_,m),orig in zip(bitmods,orig_methods):
    m.activation_quant=orig
if any(len(b)==0 for b in target_banks):
    missing=[bitmods[i][0] for i,b in enumerate(target_banks) if not b]
    raise RuntimeError(f"calibration missing modules: {missing[:8]}")
codebooks=[build_codebook(torch.cat(bank)) for bank in target_banks]
print("CODEBOOK_CAL",json.dumps({
 "seconds":time.time()-t0,
 "modules":len(codebooks),
 "cb_min":min(float(cb.min()) for cb in codebooks),
 "cb_median":float(torch.tensor([float(cb.median()) for cb in codebooks]).median()),
 "cb_max":max(float(cb.max()) for cb in codebooks),
}),flush=True)

def dynamic_a4_method(self,x,num_bits=8):
    xf=x.float()
    a=xf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-6)
    scale=7.0/a
    q=torch.round(xf*scale).clamp(-8,7)
    return q.to(torch.int8),scale

def make_adaptive_method(cb_cpu):
    cb=[float(v) for v in cb_cpu.tolist()]
    def aq(self,x,num_bits=8):
        xf=x.float()
        xs=sample_channels(xf)
        best_mse=torch.full(xf.shape[:-1]+(1,),float("inf"),dtype=torch.float32,device=xf.device)
        best_c=torch.full_like(best_mse,cb[0])
        for c0 in cb:
            c=float(c0)
            scale=7.0/c
            q=torch.round(xs*scale).clamp(-8,7)
            recon=q/scale
            mse=torch.mean((recon-xs)**2,dim=-1,keepdim=True)
            take=mse<best_mse
            best_mse=torch.where(take,mse,best_mse)
            best_c=torch.where(take,torch.full_like(best_c,c),best_c)
        scale=7.0/best_c.clamp_min(1e-6)
        q=torch.round(xf*scale).clamp(-8,7)
        return q.to(torch.int8),scale
    return aq

def seq_nll_from_logits(logits,ids):
    sl=logits[:,:-1,:].float(); y=ids[:,1:]
    losses=F.cross_entropy(
        sl.reshape(-1,sl.shape[-1]),y.reshape(-1),reduction="none"
    ).reshape(ids.shape[0],-1)
    return losses.mean(dim=1).cpu().tolist()

def patch_mode(mode):
    if mode=="native":
        return
    if mode=="dynamic_a4":
        for _,m in bitmods:
            m.activation_quant=types.MethodType(dynamic_a4_method,m)
        return
    if mode=="adaptive_a4":
        for (_,m),cb in zip(bitmods,codebooks):
            m.activation_quant=types.MethodType(make_adaptive_method(cb),m)
        return
    raise ValueError(mode)

def restore_native():
    for (_,m),orig in zip(bitmods,orig_methods):
        m.activation_quant=orig

def evaluate(corpus_name,segments,name,mode):
    restore_native(); patch_mode(mode)
    vals=[]; t0=time.time()
    try:
        with torch.inference_mode():
            for k in range(0,len(segments),EVAL_BATCH):
                b=torch.cat(segments[k:k+EVAL_BATCH],dim=0)
                out=model(b,use_cache=False)
                vals.extend(seq_nll_from_logits(out.logits,b))
    finally:
        restore_native()
    nll=sum(vals)/len(vals)
    result={
      "corpus":corpus_name,"name":name,"nll":nll,"ppl":math.exp(nll),
      "sequence_nlls":vals,"seconds":time.time()-t0
    }
    print("RESULT",json.dumps({k:v for k,v in result.items() if k!="sequence_nlls"}),flush=True)
    return result

def boot_ci(diff,seed):
    rng=random.Random(seed); n=len(diff); means=[]
    for _ in range(BOOT):
        means.append(sum(diff[rng.randrange(n)] for __ in range(n))/n)
    means.sort()
    return [means[int(0.025*BOOT)],means[min(BOOT-1,int(0.975*BOOT))]]

corpora={"wikitext2_test":wiki_eval,"c4_realnewslike_validation":c4_eval}
results=[]
for cname,segs in corpora.items():
    results.append(evaluate(cname,segs,"native_W1.58A8","native"))
    results.append(evaluate(cname,segs,"native_site_dynamic_absmax_A4","dynamic_a4"))
    results.append(evaluate(cname,segs,"native_site_adaptive_codebook16_A4","adaptive_a4"))

by={(r["corpus"],r["name"]):r for r in results}
decisions={}
for ci,cname in enumerate(corpora):
    nat=by[(cname,"native_W1.58A8")]
    dyn=by[(cname,"native_site_dynamic_absmax_A4")]
    ada=by[(cname,"native_site_adaptive_codebook16_A4")]
    diff=[a-b for a,b in zip(ada["sequence_nlls"],dyn["sequence_nlls"])]
    ci95=boot_ci(diff,BOOT_SEED+ci)
    beats=bool(ada["nll"]<dyn["nll"] and ci95[1]<0)
    viable=bool(ada["nll"]<=nat["nll"]+0.50)
    decisions[cname]={
      "delta_adaptive_vs_dynamic":ada["nll"]-dyn["nll"],
      "adaptive_vs_dynamic_bootstrap_95ci":ci95,
      "adaptive_beats_dynamic":beats,
      "delta_adaptive_vs_native_A8":ada["nll"]-nat["nll"],
      "absolute_viability_within_plus_0p50":viable,
      "all":bool(beats and viable)
    }
robust=all(v["all"] for v in decisions.values())
status="GATE010_NATIVE_BITLINEAR_ADAPTIVE_A4_PASS" if robust else "GATE010_NATIVE_BITLINEAR_ADAPTIVE_A4_FAIL_STOP_ADAPTIVE_METHOD_LANE"
final={
 "experiment":"BITNET_GATE010_NATIVE_BITLINEAR_ADAPTIVE_A4_TRANSFER",
 "status":status,
 "config":{
   "model":MODEL,"seed":SEED,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,
   "evaluation_sequences_per_corpus":N_EVAL,
   "evaluation_prediction_tokens_per_corpus":N_EVAL*(SEQ-1),
   "bitlinear_modules":len(bitmods),"codebook_entries":CODEBOOK,
   "scale_index_bits":SCALE_BITS,"selection_mse_channel_sample":MSE_CH,
   "candidate_clip_fractions":FRACS,"codebook_quantile_endpoints":[QLO,QHI],
   "bootstrap_resamples":BOOT,"data_hashes_full32":DATA_HASHES,
   "calibration_transfer":"WikiText validation calibration frozen and transferred unchanged to C4"
 },
 "decisions":decisions,
 "results":results,
}
with open("gate010_results.json","w") as f:
    json.dump(final,f,indent=2,sort_keys=True)
print("FINAL_JSON",json.dumps(final),flush=True)
