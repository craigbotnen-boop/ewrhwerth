#!/usr/bin/env python3
"""
URC-AI S5A audit:
- same deterministic running-parity forced-readout assay as recoverability_gap.py
- probes early/mid/final hidden layers
- raw-input linear baseline
- random-label/selectivity controls
- same example-level train/test split
"""
from __future__ import annotations
import argparse, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import recoverability_gap as base

def pipe(seed=base.SEED):
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0,random_state=seed))

def raw_feat(e,step):
    bits=np.zeros(32,dtype=float); bits[:e["K"]]=e["bits"]
    pos=np.zeros(32,dtype=float); pos[step]=1.0
    kval=np.array([float(e["K"]==16),float(e["K"]==32)])
    return np.concatenate([bits,pos,kval])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-key",choices=base.MODELS,required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--n-per-length",type=int,default=192); ap.add_argument("--batch-size",type=int,default=16)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tok,model=base.load(base.MODELS[a.model_key]); id0,id1,s0,s1=base.label_ids(tok)
    cap=base.Capture(model,a.model_key); ex=base.make_examples(a.n_per_length)
    L=len(cap.layers); probe_layers=[0,L//2,L-1]; transport_layers=probe_layers
    step_rows=[]; example_transport=[]

    for K in [16,32]:
      group=[e for e in ex if e["K"]==K]
      for start in range(0,len(group),a.batch_size):
        batch=group[start:start+a.batch_size]; enc=tok([e["prompt"] for e in batch],return_tensors="pt",padding=True)
        cap.values.clear()
        with torch.inference_mode():
            o=model(**enc,output_attentions=True,output_hidden_states=True,use_cache=True,return_dict=True)
        B=len(batch)
        for bi,e in enumerate(batch):
            rec={"example_id":e["example_id"],"K":K}
            for l in transport_layers:
                Ah=o.attentions[l][bi].detach().float().cpu().numpy(); A=Ah.mean(0)
                for name,val in zip(["gap","trel","hub","reff"],base.graph_metrics(A)):rec[f"raw_L{l}_{name}"]=val
                hn=base.head_residual_norms(model,a.model_key,l,cap.values[l],bi); C=base.contrib(Ah,hn)
                for name,val in zip(["gap","trel","hub","reff"],base.graph_metrics(C)):rec[f"contrib_L{l}_{name}"]=val
            example_transport.append(rec)

        past=o.past_key_values; attention_mask=enc.attention_mask; logits=o.logits[:,-1,:]
        hcur={l:o.hidden_states[l+1][:,-1,:].detach().float().cpu().numpy() for l in probe_layers}
        del o
        emitted=np.zeros((B,K),dtype=np.int8); hsteps={l:[] for l in probe_layers}
        for t in range(K):
            for l in probe_layers:hsteps[l].append(hcur[l].copy())
            pair=torch.stack([logits[:,id0],logits[:,id1]],dim=1); pred=torch.argmax(pair,dim=1)
            emitted[:,t]=pred.detach().cpu().numpy().astype(np.int8)
            next_ids=torch.where(pred==0,torch.tensor(id0),torch.tensor(id1)).view(B,1)
            if t==K-1:break
            attention_mask=torch.cat([attention_mask,torch.ones((B,1),dtype=attention_mask.dtype)],dim=1)
            with torch.inference_mode():
                oo=model(input_ids=next_ids,attention_mask=attention_mask,past_key_values=past,output_hidden_states=True,use_cache=True,return_dict=True)
            past=oo.past_key_values; logits=oo.logits[:,-1,:]
            hcur={l:oo.hidden_states[l+1][:,-1,:].detach().float().cpu().numpy() for l in probe_layers}
            del oo
        H={l:np.stack(hsteps[l],axis=1) for l in probe_layers}
        for bi,e in enumerate(batch):
          for t in range(K):
            step_rows.append(dict(example_id=e["example_id"],K=K,step=t+1,target=int(e["targets"][t]),emitted=int(emitted[bi,t]),
                                  model_correct=int(emitted[bi,t]==e["targets"][t]),raw=raw_feat(e,t),
                                  **{f"h{l}":H[l][bi,t] for l in probe_layers}))
        gc.collect(); print(a.model_key,K,start,"done",flush=True)
    cap.close()

    rng=np.random.default_rng(base.SEED); ids=np.array([e["example_id"] for e in ex]); rng.shuffle(ids); cut=int(.7*len(ids))
    train=set(ids[:cut]); test=set(ids[cut:])
    tr=[r for r in step_rows if r["example_id"] in train]; te=[r for r in step_rows if r["example_id"] in test]
    ytr=np.array([r["target"] for r in tr]); yte=np.array([r["target"] for r in te])
    outrows=[{k:v for k,v in r.items() if k not in ["raw"]+[f"h{l}" for l in probe_layers]} for r in te]
    summary={"model":a.model_key,"n_examples":len(ex),"test_examples":len(test),"model_bit_accuracy":float(np.mean([r["model_correct"] for r in te])),
             "probe_layers":probe_layers,"label_tokens":{"zero":id0,"one":id1,"zero_form":s0,"one_form":s1}}

    probe_results={}
    final_pred=None
    for l in probe_layers:
        Xtr=np.stack([r[f"h{l}"] for r in tr]); Xte=np.stack([r[f"h{l}"] for r in te])
        p=pipe(); p.fit(Xtr,ytr); pred=p.predict(Xte); acc=float(np.mean(pred==yte))
        wrong=np.array([r["model_correct"]==0 for r in te])
        wrong_acc=float(np.mean(pred[wrong]==yte[wrong])) if wrong.any() else None
        probe_results[str(l)]={"accuracy":acc,"gap_vs_model":acc-summary["model_bit_accuracy"],"accuracy_on_model_wrong":wrong_acc}
        for q,pr in zip(outrows,pred):q[f"probe_L{l}_pred"]=int(pr); q[f"probe_L{l}_correct"]=int(pr==q["target"])
        if l==probe_layers[-1]:final_pred=pred

    # raw input linear baseline
    Xrtr=np.stack([r["raw"] for r in tr]); Xrte=np.stack([r["raw"] for r in te]); rp=pipe(); rp.fit(Xrtr,ytr); rpred=rp.predict(Xrte)
    summary["raw_input_linear_accuracy"]=float(np.mean(rpred==yte))

    # Random-label selectivity control: same final hidden states, shuffled train labels, evaluated on true held-out labels.
    Xftr=np.stack([r[f"h{probe_layers[-1]}"] for r in tr]); Xfte=np.stack([r[f"h{probe_layers[-1]}"] for r in te])
    rnd=[]
    for q in range(10):
        rr=np.random.default_rng(base.SEED+100+q); ys=ytr.copy(); rr.shuffle(ys)
        pp=pipe(base.SEED+q); pp.fit(Xftr,ys); rnd.append(float(np.mean(pp.predict(Xfte)==yte)))
    summary["random_label_control_accuracy_mean"]=float(np.mean(rnd)); summary["random_label_control_accuracy_std"]=float(np.std(rnd)); summary["random_label_runs"]=rnd
    summary["layerwise_probe"]=probe_results

    testdf=pd.DataFrame(outrows); testdf.to_csv(out/"heldout_steps_audit.csv",index=False)
    finalcol=f"probe_L{probe_layers[-1]}_correct"
    bdf=testdf.rename(columns={finalcol:"probe_correct"})
    summary["final_recoverability_gap"]=float(bdf.probe_correct.mean()-bdf.model_correct.mean())
    summary["final_gap_bootstrap_95"]=base.bootstrap_gap(bdf)

    trans=pd.DataFrame(example_transport); trans.to_csv(out/"example_transport.csv",index=False)
    exacc=bdf.groupby(["example_id","K"]).agg(model_acc=("model_correct","mean"),probe_acc=("probe_correct","mean")).reset_index(); exacc["gap"]=exacc.probe_acc-exacc.model_acc
    merged=exacc.merge(trans,on=["example_id","K"],how="left"); merged.to_csv(out/"heldout_example_gap.csv",index=False)
    controls=merged[["K"]].to_numpy(float); rawcols=[c for c in merged if c.startswith("raw_")]; concols=[c for c in merged if c.startswith("contrib_")]
    def mx(cols):
        X=merged[cols].replace([np.inf,-np.inf],np.nan); return X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(float)
    y=merged.gap.to_numpy(float); b=base.cv_r2(controls,y); r=base.cv_r2(np.column_stack([controls,mx(rawcols)]),y); c=base.cv_r2(np.column_stack([controls,mx(concols)]),y)
    summary["s6_pilot_cv_r2"]={"controls_K":b,"plus_raw":r,"plus_contribution":c,"delta_raw":r-b,"delta_contribution":c-b}

    (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
