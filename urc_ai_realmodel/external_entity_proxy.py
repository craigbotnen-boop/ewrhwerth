#!/usr/bin/env python3
"""
External-data replication using ICML-2026 entity-tracking generator output.

Input JSONL comes from PootieT/entity-tracking-mi generator and includes:
  sentence
  counterfactual_rand_obj_rand_query_id

We do NOT claim or recompute their task labels here. This replication tests
only the mechanistic proxy result:
  raw target-row attention vs contribution-weighted target-row routing
  as rankings of actual residual-stream activation-patching influence.

Pairs are filtered BEFORE analysis to identical tokenizer length and bounded
sequence length so source positions align exactly.
"""
from __future__ import annotations
import argparse,json,gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import recoverability_gap as base

MODELS={
 "qwen05":("Qwen/Qwen2.5-0.5B-Instruct","qwen"),
 "qwen15":("Qwen/Qwen2.5-1.5B-Instruct","qwen"),
}

class Cap:
    def __init__(self,model):
        self.values={};self.handles=[];self.layers=model.model.layers
        for i,l in enumerate(self.layers):
            def mk(ii):
                def h(mod,inp,out):self.values[ii]=out.detach().float().cpu()
                return h
            self.handles.append(l.self_attn.v_proj.register_forward_hook(mk(i)))
    def close(self):
        for h in self.handles:h.remove()

def load(mid):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    t=AutoTokenizer.from_pretrained(mid,use_fast=True)
    m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation="eager")
    m.eval();return t,m

def contribution(Ah,norms):
    return np.einsum("hi,ih->i",Ah[:,-1,:],norms,optimize=True)

def patch_batch(model,layer,ids,donor_h,positions,batch=12):
    N=ids.shape[1];out=np.zeros(N)
    # baseline final hidden
    with torch.inference_mode():
        b=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
    base=b.hidden_states[-1][0,-1,:].detach().float().cpu().numpy();del b
    for s in range(0,N,batch):
        poss=list(range(s,min(N,s+batch)));xx=ids.repeat(len(poss),1)
        def hook(mod,inp,o):
            if isinstance(o,tuple):
                z=o[0].clone()
                for bi,p in enumerate(poss):z[bi,p,:]=donor_h[p,:].to(z.device,z.dtype)
                return (z,)+tuple(o[1:])
            z=o.clone()
            for bi,p in enumerate(poss):z[bi,p,:]=donor_h[p,:].to(z.device,z.dtype)
            return z
        h=layer.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                q=model(input_ids=xx,output_hidden_states=True,use_cache=False,return_dict=True)
        finally:h.remove()
        hf=q.hidden_states[-1][:,-1,:].detach().float().cpu().numpy();del q
        out[poss]=np.linalg.norm(hf-base[None,:],axis=1)
    return out

def corr(a,b):
    x=spearmanr(a,b,nan_policy="omit").statistic
    return float(x) if np.isfinite(x) else np.nan

def boot(v,n=2000):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    rng=np.random.default_rng(20260927)
    x=[float(np.median(rng.choice(v,size=len(v),replace=True))) for _ in range(n)]
    return np.quantile(x,[.025,.5,.975]).tolist()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-key",choices=MODELS,required=True)
    ap.add_argument("--data",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--n",type=int,default=20)
    ap.add_argument("--max-tokens",type=int,default=180)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    mid,fam=MODELS[a.model_key];tok,model=load(mid);cap=Cap(model);middle=len(cap.layers)//2

    rows=[]
    candidates=[]
    with open(a.data,encoding="utf-8") as f:
        for line in f:
            d=json.loads(line)
            if "counterfactual_rand_obj_rand_query_id" not in d:continue
            s=d["sentence"];ct=d["counterfactual_rand_obj_rand_query_id"]
            x=tok.encode(s,add_special_tokens=False);y=tok.encode(ct,add_special_tokens=False)
            if len(x)==len(y) and 8<len(x)<=a.max_tokens and x!=y:
                candidates.append((s,ct,d.get("sample_id"),d.get("numops")))
            if len(candidates)>=a.n:break
    if len(candidates)<max(8,a.n//2):
        raise RuntimeError(f"Too few aligned external pairs: {len(candidates)}")

    for i,(s,ct,sid,numops) in enumerate(candidates):
        ids=torch.tensor([tok.encode(s,add_special_tokens=False)],dtype=torch.long)
        dids=torch.tensor([tok.encode(ct,add_special_tokens=False)],dtype=torch.long)
        cap.values.clear()
        with torch.inference_mode():
            o=model(input_ids=ids,output_attentions=True,output_hidden_states=True,use_cache=False,return_dict=True)
        basecap=cap.values[middle].clone()
        Ah=o.attentions[middle][0].detach().float().cpu().numpy()
        raw=Ah[:,-1,:].mean(0)
        hn=base.head_residual_norms(model,"qwen",middle,basecap,0)
        con=contribution(Ah,hn)
        base_mid=o.hidden_states[middle+1][0].detach().float().cpu();del o

        cap.values.clear()
        with torch.inference_mode():
            d=model(input_ids=dids,output_hidden_states=True,use_cache=False,return_dict=True)
        donor=d.hidden_states[middle+1][0].detach().float().cpu();del d
        patch=patch_batch(model,cap.layers[middle],ids,donor,list(range(ids.shape[1])))
        patchnorm=torch.norm(donor-base_mid,dim=1).numpy()+1e-9
        normpatch=patch/patchnorm
        valid=np.arange(ids.shape[1]-1)
        rr=corr(raw[valid],normpatch[valid]);rc=corr(con[valid],normpatch[valid])
        rows.append({"pair":i,"sample_id":sid,"numops":numops,"N":ids.shape[1],
                     "rho_raw_patch":rr,"rho_contribution_patch":rc,"paired_gain":rc-rr})
        print(a.model_key,i+1,"/",len(candidates),"N",ids.shape[1],"raw",round(rr,3),"con",round(rc,3),flush=True)
        gc.collect()
    cap.close()
    df=pd.DataFrame(rows);df.to_csv(out/"external_entity_proxy.csv",index=False)
    summary={
      "model":a.model_key,"model_id":mid,"middle_layer":middle,"n_pairs":len(df),
      "median_rho_raw":float(df.rho_raw_patch.median()),
      "median_rho_contribution":float(df.rho_contribution_patch.median()),
      "median_paired_gain":float(df.paired_gain.median()),
      "raw_bootstrap95":boot(df.rho_raw_patch),
      "contribution_bootstrap95":boot(df.rho_contribution_patch),
      "paired_gain_bootstrap95":boot(df.paired_gain),
      "external_source":"PootieT/entity-tracking-mi (ICML 2026), generator-produced 2-put entity tracking sentences; built-in rand_obj_rand_query_id counterfactuals",
      "scope":"Mechanistic proxy replication only; no claim about benchmark task accuracy or original paper labels."
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
