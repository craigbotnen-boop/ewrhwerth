import json, math, random, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED=26081603
MODEL="microsoft/bitnet-b1.58-2B-4T"
SEQ=128; N_CAL=2; N_EVAL=8; CLIP_Q=0.995; BLOCK=64
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
device=torch.device("cpu")
print("GATE003E_START",json.dumps({"model":MODEL,"seq":SEQ,"n_cal":N_CAL,"n_eval":N_EVAL,"eval_prediction_tokens":N_EVAL*(SEQ-1),"rule":"|delta NLL|<=0.10 => radial restoration explains gap; angular+0.10<restored => fixed angular code adds independent value; restored+0.10<angular => dynamic angular code preferred"}),flush=True)

tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=False,dtype=torch.bfloat16,device_map={"":"cpu"}); model.eval()
layers=model.model.layers; D=int(model.config.hidden_size)
if D%BLOCK: raise RuntimeError("hidden size not divisible by block")

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
cal,ev=segments[:N_CAL],segments[N_CAL:]

# Calibrate the same fixed angular grid as Gate003.
zs=[[] for _ in layers]; handles=[]
for li,layer in enumerate(layers):
    def mk(i):
        def hook(_m,_inp,o):
            h=hidden(o).detach(); rr=rms(h); z=rot(h.float()/rr).abs().flatten()
            n=min(8192,z.numel()); idx=torch.linspace(0,z.numel()-1,n).long(); zs[i].append(z[idx].cpu()); return None
        return hook
    handles.append(layer.register_forward_hook(mk(li)))
with torch.inference_mode():
    for s in cal: model(s,use_cache=False)
for h in handles:h.remove()
clips=[]
for bank in zs:
    z=torch.cat(bank); clips.append(max(float(torch.quantile(z,CLIP_Q)),1.0))
print("CAL",json.dumps({"calibration_tokens":N_CAL*SEQ,"clip_min":min(clips),"clip_median":float(torch.tensor(clips).median()),"clip_max":max(clips)}),flush=True)

def sym_a4(z):
    zf=z.float(); qmax=7; a=zf.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8); sc=a/qmax
    return torch.round(zf/sc).clamp(-qmax,qmax)*sc

def dynamic_restored(h):
    rr=rms(h); hh=irot(sym_a4(rot(h))); hh=hh/rms(hh)
    return (rr*hh).to(h.dtype)

def qdir4(z,c):
    step=float(c)/7.0; return torch.round(torch.clamp(z,-float(c),float(c))/step)*step

def angular_true(h,i):
    rr=rms(h); u=h.float()/rr; uh=irot(qdir4(rot(u),clips[i])); uh=uh/rms(uh)
    return (rr*uh).to(h.dtype)

def evaluate(name,mode):
    hs=[]
    for li,layer in enumerate(layers):
        def mk(i):
            def hook(_m,_inp,o):
                h=hidden(o); hh=dynamic_restored(h) if mode=="restored" else angular_true(h,i)
                return repl(o,hh)
            return hook
        hs.append(layer.register_forward_hook(mk(li)))
    total=0.; nt=0; t0=time.time()
    with torch.inference_mode():
        for s in ev:
            o=model(s,labels=s,use_cache=False); n=s.shape[1]-1; total+=float(o.loss)*n; nt+=n
    for h in hs:h.remove()
    nll=total/nt; out={"name":name,"nll":nll,"ppl":math.exp(nll),"seconds":time.time()-t0}; print("RESULT",json.dumps(out),flush=True); return out

angular=evaluate("angularH_A4_radius_true","angular")
restored=evaluate("hadamard_dynamic_A4_absmax_radius_restored","restored")
d=angular["nll"]-restored["nll"]
if abs(d)<=0.10: verdict="RADIAL_RESTORATION_EXPLAINS_GAP_WITHIN_0P10_NLL"
elif d < -0.10: verdict="FIXED_ANGULAR_GRID_ADDS_INDEPENDENT_VALUE"
else: verdict="DYNAMIC_ANGULAR_CODE_WITH_RADIUS_RESTORATION_PREFERRED"
print("FINAL_JSON",json.dumps({"experiment":"BITNET_PERSISTENT_RADIAL_ANGULAR_001_GATE003E_HOSTILE_CONTROL","config":{"model":MODEL,"seq":SEQ,"calibration_tokens":N_CAL*SEQ,"eval_prediction_tokens":N_EVAL*(SEQ-1),"seed":SEED},"angular_true":angular,"dynamic_radius_restored":restored,"delta_nll_angular_minus_restored":d,"threshold_nll":0.10,"verdict":verdict},sort_keys=True),flush=True)
