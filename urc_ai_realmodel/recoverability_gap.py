#!/usr/bin/env python3
"""
URC-AI S5A: time-aligned latent/readout recoverability gap.

Task: running parity over random bit strings.
At each step t:
  H_t = final hidden state immediately BEFORE emitting parity bit t.
  Y_t = ground-truth running parity.
  T_t = model's constrained serialized parity bit (0/1).

The generation vocabulary is constrained to the two single-token labels only.
This guarantees alignment and avoids free-form parsing artifacts.

Primary operational quantity:
  Delta R = Acc(linear probe H_t -> Y_t) - Acc(model serialized bit -> Y_t)

The probe split is by whole examples, not by individual steps.

This is recoverability, not evidence the model causally used Y_t.
"""
from __future__ import annotations
import argparse, json, os, random, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score

torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.set_num_interop_threads(1)

MODELS={"qwen":"Qwen/Qwen2.5-0.5B-Instruct","pythia":"EleutherAI/pythia-160m"}
SEED=20260927

def label_ids(tok):
    for a,b in [(" 0"," 1"),("0","1"),("\n0","\n1")]:
        ia=tok.encode(a,add_special_tokens=False); ib=tok.encode(b,add_special_tokens=False)
        if len(ia)==1 and len(ib)==1 and ia[0]!=ib[0]:
            return ia[0],ib[0],a,b
    raise RuntimeError("Could not find single-token binary labels.")

def make_examples(n_per_len=192):
    rng=np.random.default_rng(SEED)
    ex=[]; eid=0
    for K in [16,32]:
      for _ in range(n_per_len):
        bits=rng.integers(0,2,size=K,dtype=np.int8)
        parity=np.bitwise_xor.accumulate(bits)
        prompt=(
          "Input bits: "+" ".join(map(str,bits.tolist()))+
          ". Compute the running parity from left to right. "
          f"Output exactly {K} parity bits, using only 0 or 1, in order."
          "\nRunning parity:"
        )
        ex.append(dict(example_id=eid,K=K,bits=bits,targets=parity,prompt=prompt)); eid+=1
    return ex

class Capture:
    def __init__(self,model,key):
        self.key=key; self.handles=[]; self.values={}
        if key=="qwen":
            self.layers=model.model.layers
            for i,layer in enumerate(self.layers):
                def mk(ii):
                    def h(mod,inp,out): self.values[ii]=out.detach().float().cpu()
                    return h
                self.handles.append(layer.self_attn.v_proj.register_forward_hook(mk(i)))
        else:
            self.layers=model.gpt_neox.layers
            for i,layer in enumerate(self.layers):
                def mk(ii):
                    def h(mod,inp,out): self.values[ii]=out.detach().float().cpu()
                    return h
                self.handles.append(layer.attention.query_key_value.register_forward_hook(mk(i)))
    def close(self):
        for h in self.handles:h.remove()

def head_residual_norms(model,key,l,captured,bidx):
    if key=="qwen":
        att=model.model.layers[l].self_attn; Hq=model.config.num_attention_heads; Hkv=model.config.num_key_value_heads
        D=captured.shape[-1]//Hkv
        V=captured[bidx].numpy().reshape(captured.shape[1],Hkv,D)
        V=np.repeat(V,Hq//Hkv,axis=1)
        Wo=att.o_proj.weight.detach().float().cpu().numpy().reshape(model.config.hidden_size,Hq,D)
    else:
        att=model.gpt_neox.layers[l].attention; Hq=model.config.num_attention_heads; D=model.config.hidden_size//Hq
        qkv=captured[bidx].numpy().reshape(captured.shape[1],Hq,3,D); V=qkv[:,:,2,:]
        Wo=att.dense.weight.detach().float().cpu().numpy().reshape(model.config.hidden_size,Hq,D)
    G=np.einsum("ohd,ohe->hde",Wo,Wo,optimize=True)
    sq=np.einsum("jhd,hde,jhe->jh",V,G,V,optimize=True)
    return np.sqrt(np.maximum(sq,0))

def contrib(Ah,norms):
    C=np.einsum("hij,jh->ij",Ah,norms,optimize=True); s=C.sum(1,keepdims=True)
    return np.divide(C,s,out=np.zeros_like(C),where=s>1e-15)

def graph_metrics(A):
    W=(A+A.T)/2; deg=W.sum(1); keep=deg>1e-14; W=W[np.ix_(keep,keep)]; deg=W.sum(1)
    if len(W)<3:return (np.nan,)*4
    mean=deg.mean(); hub=deg.max()/mean
    inv=1/np.sqrt(np.maximum(deg,1e-300)); S=(inv[:,None]*W)*inv[None,:]
    Ln=np.eye(len(W))-S; Ln=(Ln+Ln.T)/2
    en=scipy.linalg.eigvalsh(Ln,check_finite=False); pos=en[en>1e-10]
    gap=float(pos[0]) if len(pos) else np.nan; trel=1/gap if np.isfinite(gap) and gap>0 else np.inf
    Lc=np.diag(deg)-W; ec=scipy.linalg.eigvalsh(Lc,check_finite=False); ep=ec[ec>1e-10]
    reff=float(2/(len(W)-1)*np.sum(1/ep)) if len(ep)==len(W)-1 else np.nan
    return gap,trel,float(hub),reff

def load(mid):
    tok=AutoTokenizer.from_pretrained(mid,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    try:model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation="eager")
    except TypeError:
        model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,low_cpu_mem_usage=True)
        try:model.config._attn_implementation="eager"
        except:pass
    model.eval(); return tok,model

def bootstrap_gap(rows,nboot=1000):
    rng=np.random.default_rng(SEED+1)
    ids=np.unique(rows.example_id)
    vals=[]
    for _ in range(nboot):
        samp=rng.choice(ids,size=len(ids),replace=True)
        rr=pd.concat([rows[rows.example_id==i] for i in samp],ignore_index=True)
        vals.append(rr.probe_correct.mean()-rr.model_correct.mean())
    return np.quantile(vals,[.025,.5,.975]).tolist()

def cv_r2(X,y):
    if len(y)<30:return np.nan
    kf=KFold(n_splits=5,shuffle=True,random_state=SEED)
    model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    return float(np.mean(cross_val_score(model,X,y,cv=kf,scoring="r2")))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-key",choices=MODELS,required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--n-per-length",type=int,default=192); ap.add_argument("--batch-size",type=int,default=16)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tok,model=load(MODELS[a.model_key]); id0,id1,s0,s1=label_ids(tok); cap=Capture(model,a.model_key)
    ex=make_examples(a.n_per_length); selected=[0,len(cap.layers)//2,len(cap.layers)-1]
    step_rows=[]; example_transport=[]

    # deterministic grouped batches by K for equal generation horizons
    for K in [16,32]:
      group=[e for e in ex if e["K"]==K]
      for start in range(0,len(group),a.batch_size):
        batch=group[start:start+a.batch_size]; prompts=[e["prompt"] for e in batch]
        enc=tok(prompts,return_tensors="pt",padding=True)
        cap.values.clear()
        with torch.inference_mode():
            o=model(**enc,output_attentions=True,output_hidden_states=True,use_cache=True,return_dict=True)
        B=len(batch); seqN=enc.input_ids.shape[1]
        # Prompt transport metrics, raw and contribution, selected layers.
        for bi,e in enumerate(batch):
            rec={"example_id":e["example_id"],"K":K}
            for l in selected:
                Ah=o.attentions[l][bi].detach().float().cpu().numpy()
                A=Ah.mean(0); g=graph_metrics(A)
                for name,val in zip(["gap","trel","hub","reff"],g):rec[f"raw_L{l}_{name}"]=val
                hn=head_residual_norms(model,a.model_key,l,cap.values[l],bi)
                C=contrib(Ah,hn); cg=graph_metrics(C)
                for name,val in zip(["gap","trel","hub","reff"],cg):rec[f"contrib_L{l}_{name}"]=val
            example_transport.append(rec)

        past=o.past_key_values
        h=o.hidden_states[-1][:,-1,:].detach().float().cpu().numpy()
        logits=o.logits[:,-1,:]
        attention_mask=enc.attention_mask
        del o
        emitted=np.zeros((B,K),dtype=np.int8)
        hidden_steps=[]
        for t in range(K):
            hidden_steps.append(h.copy())
            pair=torch.stack([logits[:,id0],logits[:,id1]],dim=1)
            pred=torch.argmax(pair,dim=1)
            emitted[:,t]=pred.detach().cpu().numpy().astype(np.int8)
            next_ids=torch.where(pred==0,torch.tensor(id0),torch.tensor(id1)).view(B,1)
            if t==K-1:break
            attention_mask=torch.cat([attention_mask,torch.ones((B,1),dtype=attention_mask.dtype)],dim=1)
            with torch.inference_mode():
                oo=model(input_ids=next_ids,attention_mask=attention_mask,past_key_values=past,
                         output_hidden_states=True,use_cache=True,return_dict=True)
            past=oo.past_key_values; h=oo.hidden_states[-1][:,-1,:].detach().float().cpu().numpy(); logits=oo.logits[:,-1,:]
            del oo
        H=np.stack(hidden_steps,axis=1) # B,K,d
        for bi,e in enumerate(batch):
            for t in range(K):
                step_rows.append({"example_id":e["example_id"],"K":K,"step":t+1,"target":int(e["targets"][t]),
                                  "emitted":int(emitted[bi,t]),"model_correct":int(emitted[bi,t]==e["targets"][t]),
                                  "hidden":H[bi,t]})
        gc.collect()
        print(a.model_key,K,start,"done",flush=True)
    cap.close()

    # grouped train/test split by example
    rng=np.random.default_rng(SEED); ids=np.array([e["example_id"] for e in ex]); rng.shuffle(ids)
    cut=int(.7*len(ids)); train=set(ids[:cut]); test=set(ids[cut:])
    train_rows=[r for r in step_rows if r["example_id"] in train]; test_rows=[r for r in step_rows if r["example_id"] in test]
    Xtr=np.stack([r["hidden"] for r in train_rows]); ytr=np.array([r["target"] for r in train_rows])
    Xte=np.stack([r["hidden"] for r in test_rows]); yte=np.array([r["target"] for r in test_rows])
    probe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0,random_state=SEED))
    probe.fit(Xtr,ytr); pp=probe.predict(Xte)

    clean=[]
    for r,p in zip(test_rows,pp):
        q={k:v for k,v in r.items() if k!="hidden"}; q["probe_pred"]=int(p); q["probe_correct"]=int(p==r["target"]); clean.append(q)
    testdf=pd.DataFrame(clean); testdf.to_csv(out/"heldout_steps.csv",index=False)
    trans=pd.DataFrame(example_transport); trans.to_csv(out/"example_transport.csv",index=False)
    exacc=testdf.groupby(["example_id","K"]).agg(model_acc=("model_correct","mean"),probe_acc=("probe_correct","mean")).reset_index()
    exacc["gap"]=exacc.probe_acc-exacc.model_acc
    merged=exacc.merge(trans,on=["example_id","K"],how="left"); merged.to_csv(out/"heldout_example_gap.csv",index=False)

    wrong=testdf[testdf.model_correct==0]
    primary={
      "model":a.model_key,"n_examples":len(ex),"test_examples":len(test),
      "model_bit_accuracy":float(testdf.model_correct.mean()),
      "probe_bit_accuracy":float(testdf.probe_correct.mean()),
      "recoverability_gap":float(testdf.probe_correct.mean()-testdf.model_correct.mean()),
      "gap_bootstrap_95":bootstrap_gap(testdf),
      "probe_accuracy_on_model_wrong_bits":float(wrong.probe_correct.mean()) if len(wrong) else None,
      "n_model_wrong_bits":int(len(wrong)),
      "label_tokens":{"zero":id0,"one":id1,"zero_form":s0,"one_form":s1}
    }

    # S6 exploratory held-out prediction of per-example gap.
    controls=merged[["K"]].to_numpy(float)
    raw_cols=[c for c in merged.columns if c.startswith("raw_")]
    con_cols=[c for c in merged.columns if c.startswith("contrib_")]
    def finite_matrix(cols):
        X=merged[cols].replace([np.inf,-np.inf],np.nan)
        X=X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(float)
        return X
    y=merged.gap.to_numpy(float)
    base=cv_r2(controls,y)
    raw=cv_r2(np.column_stack([controls,finite_matrix(raw_cols)]),y)
    con=cv_r2(np.column_stack([controls,finite_matrix(con_cols)]),y)
    both=cv_r2(np.column_stack([controls,finite_matrix(raw_cols),finite_matrix(con_cols)]),y)
    primary["s6_pilot_cv_r2"]={"controls_K":base,"plus_raw_transport":raw,"plus_contribution_transport":con,"plus_both":both,
                              "delta_raw":raw-base,"delta_contribution":con-base,"delta_both":both-base}
    (out/"summary.json").write_text(json.dumps(primary,indent=2))
    print(json.dumps(primary,indent=2),flush=True)

if __name__=="__main__":main()
