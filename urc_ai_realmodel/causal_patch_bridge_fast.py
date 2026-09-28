#!/usr/bin/env python3
"""
URC-AI S7A causal bridge.

Controlled task: step-4 running parity.
For each held-out example:
  1. teacher-force the first 3 correct parity bits;
  2. read the step-4 target from the last hidden state/logits;
  3. compute contribution-weight importance from source positions at the middle layer;
  4. perform matched activation patching one source position at a time:
       h_mid[source] <- h_mid_donor[source]
     with a same-length donor sequence;
  5. measure downstream changes at the final target hidden state and binary logit margin.

Primary causal-proxy question:
  Does contribution importance rank actual intervention influence?

Secondary exploratory question:
  Do causal influence-profile metrics distinguish probe-correct/model-wrong
  readout mismatches?

All probe training uses separate examples.
"""
from __future__ import annotations
import argparse, json, gc, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import recoverability_gap as base

SEED=20260927
STEP=4

def build_ids(tok,e,id0,id1):
    ids=tok.encode(e["prompt"],add_special_tokens=False)
    # Teacher force first STEP-1 correct parity labels using the same single-token labels.
    for y in e["targets"][:STEP-1]:
        ids.append(id1 if int(y)==1 else id0)
    return torch.tensor([ids],dtype=torch.long)

def projected_head_norms_from_capture(model,key,l,captured):
    return base.head_residual_norms(model,key,l,captured,0)

def contribution_target(Ah,norms):
    # Ah [H,N,N], norms [N,H]; target is last query row.
    return np.einsum("hi,ih->i",Ah[:,-1,:],norms,optimize=True)

def entropy_metrics(x):
    x=np.maximum(np.asarray(x,float),0)
    s=x.sum()
    if s<=1e-15:return dict(eff_support=np.nan,top1_share=np.nan,entropy=np.nan)
    p=x/s; nz=p[p>0]
    H=-float(np.sum(nz*np.log(nz)))
    return dict(eff_support=float(np.exp(H)),top1_share=float(p.max()),entropy=H)

def patch_run(model,layer,batch_ids,donor_h,source_positions,id0,id1,probe=None):
    """
    batch_ids [B,N], donor_h [N,d], source_positions length B.
    Patches layer output separately per batch item.
    Returns final target hidden, binary margin, probe margin.
    """
    B=batch_ids.shape[0]
    def hook(mod,inp,out):
        if isinstance(out,tuple):
            h=out[0].clone()
            for b,pos in enumerate(source_positions):
                h[b,pos,:]=donor_h[pos,:].to(h.device,h.dtype)
            return (h,)+tuple(out[1:])
        h=out.clone()
        for b,pos in enumerate(source_positions):
            h[b,pos,:]=donor_h[pos,:].to(h.device,h.dtype)
        return h

    handle=layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            o=model(input_ids=batch_ids,output_hidden_states=True,use_cache=False,return_dict=True)
    finally:
        handle.remove()
    hf=o.hidden_states[-1][:,-1,:].detach().float().cpu().numpy()
    lg=o.logits[:,-1,:].detach().float().cpu().numpy()
    margin=lg[:,id1]-lg[:,id0]
    pm=None
    if probe is not None:
        # decision_function sign corresponds to class 1 for binary logistic regression
        pm=probe.decision_function(hf)
    del o
    return hf,margin,pm

def cv_auc(X,y):
    if len(np.unique(y))<2 or min(np.bincount(y.astype(int)))<5:return np.nan
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0,random_state=SEED))
    return float(np.mean(cross_val_score(m,X,y,cv=cv,scoring="roc_auc")))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-key",choices=base.MODELS,required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--train-n",type=int,default=192)
    ap.add_argument("--test-n",type=int,default=48); ap.add_argument("--patch-batch",type=int,default=16)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tok,model=base.load(base.MODELS[a.model_key]); id0,id1,s0,s1=base.label_ids(tok)
    cap=base.Capture(model,a.model_key); L=len(cap.layers); mid=L//2
    examples=[e for e in base.make_examples(256) if e["K"]==16]
    rng=np.random.default_rng(SEED); rng.shuffle(examples)
    train=examples[:a.train_n]; test=examples[a.train_n:a.train_n+a.test_n]
    donors=examples[a.train_n+a.test_n:a.train_n+2*a.test_n]
    if len(donors)<len(test): donors=(examples[:len(test)])

    # train the identical clean step-4 probe, but batch same-length examples.
    # This changes execution efficiency only; examples, labels and probe are unchanged.
    X=[];Y=[]
    train_batch=16
    for start in range(0,len(train),train_batch):
        chunk=train[start:start+train_batch]
        parts=[build_ids(tok,e,id0,id1) for e in chunk]
        widths={int(x.shape[1]) for x in parts}
        if len(widths)!=1:
            raise RuntimeError("Unexpected token-length mismatch in fixed-format training examples")
        ids=torch.cat(parts,dim=0); cap.values.clear()
        with torch.inference_mode():
            o=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
        X.extend(o.hidden_states[-1][:,-1,:].detach().float().cpu().numpy())
        Y.extend(int(e["targets"][STEP-1]) for e in chunk)
        del o
    probe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0,random_state=SEED))
    probe.fit(np.stack(X),np.array(Y))

    rows=[]; profiles=[]
    layer=cap.layers[mid]
    for idx,(e,d) in enumerate(zip(test,donors)):
        ids=build_ids(tok,e,id0,id1); dids=build_ids(tok,d,id0,id1)
        if ids.shape[1]!=dids.shape[1]:
            # same task format should make this impossible; skip rather than retokenize post hoc
            continue
        N=ids.shape[1]
        cap.values.clear()
        with torch.inference_mode():
            o=model(input_ids=ids,output_attentions=True,output_hidden_states=True,use_cache=False,return_dict=True)
        base_h=o.hidden_states[-1][0,-1,:].detach().float().cpu().numpy()
        base_logits=o.logits[0,-1,:].detach().float().cpu().numpy()
        base_margin=float(base_logits[id1]-base_logits[id0])
        target=int(e["targets"][STEP-1]); model_pred=int(base_margin>0)
        probe_pred=int(probe.predict(base_h[None,:])[0]); probe_margin=float(probe.decision_function(base_h[None,:])[0])
        Ah=o.attentions[mid][0].detach().float().cpu().numpy()
        hn=projected_head_norms_from_capture(model,a.model_key,mid,cap.values[mid])
        contrib=contribution_target(Ah,hn)
        base_mid=o.hidden_states[mid+1][0].detach().float().cpu()
        del o

        cap.values.clear()
        with torch.inference_mode():
            od=model(input_ids=dids,output_hidden_states=True,use_cache=False,return_dict=True)
        donor_mid=od.hidden_states[mid+1][0].detach().float().cpu()
        del od
        patch_delta=torch.norm(donor_mid-base_mid,dim=1).numpy()+1e-9

        causal=np.zeros(N); dmargin=np.zeros(N); dprobe=np.zeros(N)
        for start in range(0,N,a.patch_batch):
            poss=list(range(start,min(N,start+a.patch_batch))); B=len(poss)
            batch_ids=ids.repeat(B,1)
            hf,mg,pm=patch_run(model,layer,batch_ids,donor_mid,poss,id0,id1,probe=probe)
            causal[poss]=np.linalg.norm(hf-base_h[None,:],axis=1)/patch_delta[poss]
            dmargin[poss]=np.abs(mg-base_margin)
            dprobe[poss]=np.abs(pm-probe_margin)

        # Exclude final target position for source-ranking sensitivity check; it trivially has direct carry.
        valid=np.arange(max(1,N-1))
        rho=float(spearmanr(contrib[valid],causal[valid],nan_policy="omit").statistic)
        rho_m=float(spearmanr(contrib[valid],dmargin[valid],nan_policy="omit").statistic)
        rho_p=float(spearmanr(contrib[valid],dprobe[valid],nan_policy="omit").statistic)
        em=entropy_metrics(causal[valid])
        dist=(N-1-valid).astype(float)
        cs=causal[valid].sum()
        mean_dist=float(np.sum(dist*causal[valid])/cs) if cs>0 else np.nan
        mismatch=int(probe_pred==target and model_pred!=target)
        rec=dict(example_id=int(e["example_id"]),N=int(N),target=target,model_pred=model_pred,probe_pred=probe_pred,
                 model_correct=int(model_pred==target),probe_correct=int(probe_pred==target),mismatch=mismatch,
                 base_model_margin=base_margin,base_probe_margin=probe_margin,
                 rho_contrib_causal=rho,rho_contrib_modelmargin=rho_m,rho_contrib_probemargin=rho_p,
                 causal_mean_distance=mean_dist,**em)
        rows.append(rec)
        profiles.append(pd.DataFrame({"example_id":e["example_id"],"source":np.arange(N),
                                      "contribution":contrib,"causal_hidden":causal,
                                      "delta_model_margin":dmargin,"delta_probe_margin":dprobe}))
        print(a.model_key,idx+1,"/",len(test),"rho",round(rho,3),"mismatch",mismatch,flush=True)
        gc.collect()
    cap.close()

    df=pd.DataFrame(rows); df.to_csv(out/"causal_example_metrics.csv",index=False)
    pd.concat(profiles,ignore_index=True).to_csv(out/"causal_profiles.csv",index=False)
    metrics=["eff_support","top1_share","entropy","causal_mean_distance"]
    X=df[metrics].replace([np.inf,-np.inf],np.nan).fillna(df[metrics].median()).fillna(0).to_numpy(float)
    y=df.mismatch.to_numpy(int)
    summary={
      "model":a.model_key,"middle_layer":mid,"step":STEP,"n_test":len(df),
      "probe_accuracy":float(df.probe_correct.mean()),"model_accuracy":float(df.model_correct.mean()),
      "mismatch_rate":float(df.mismatch.mean()),"n_mismatch":int(df.mismatch.sum()),
      "median_rho_contribution_vs_causal_hidden":float(df.rho_contrib_causal.median()),
      "mean_rho_contribution_vs_causal_hidden":float(df.rho_contrib_causal.mean()),
      "median_rho_contribution_vs_model_margin_effect":float(df.rho_contrib_modelmargin.median()),
      "median_rho_contribution_vs_probe_margin_effect":float(df.rho_contrib_probemargin.median()),
      "causal_metrics_mismatch_auc_cv":cv_auc(X,y),
      "interpretation":"Activation patching is a finite intervention on the residual stream. Donor swaps are matched by task/sequence length. Correlation evaluates contribution weights as a causal-ranking proxy; it does not identify natural causal effects of changing input tokens."
    }
    # nonparametric bootstrap of median rho
    rr=np.random.default_rng(SEED+4); vals=df.rho_contrib_causal.to_numpy(float)
    boots=[float(np.median(rr.choice(vals,size=len(vals),replace=True))) for _ in range(2000)]
    summary["median_rho_bootstrap_95"]=np.quantile(boots,[.025,.5,.975]).tolist()
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
