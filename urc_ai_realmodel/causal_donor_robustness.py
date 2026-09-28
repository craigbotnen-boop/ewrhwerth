#!/usr/bin/env python3
"""
URC-AI S7A donor-robustness replicate.

Uses the exact same 48 base examples as causal_patch_bridge.py, but swaps the
original donor set (first 48 shuffled training examples, due to the fallback in
the original script) for the next 48 training examples.

No probe is trained: this replicate isolates the causal source-ranking result.
All source positions are patched in one vectorized batch per example.
"""
from __future__ import annotations
import argparse, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, rankdata
import recoverability_gap as base
import causal_patch_bridge_fast as cb

SEED=cb.SEED
STEP=cb.STEP

def partial_rank_position(contrib, causal):
    n=len(contrib)
    pos=np.arange(n)
    rp,rc,ri=rankdata(pos),rankdata(contrib),rankdata(causal)
    X=np.column_stack([np.ones(n),rp])
    ec=rc-X@np.linalg.lstsq(X,rc,rcond=None)[0]
    ei=ri-X@np.linalg.lstsq(X,ri,rcond=None)[0]
    return float(np.corrcoef(ec,ei)[0,1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-key",choices=base.MODELS,required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    tok,model=base.load(base.MODELS[a.model_key])
    id0,id1,_,_=base.label_ids(tok)
    cap=base.Capture(model,a.model_key)
    L=len(cap.layers); mid=L//2; layer=cap.layers[mid]

    examples=[e for e in base.make_examples(256) if e["K"]==16]
    rng=np.random.default_rng(SEED); rng.shuffle(examples)
    test=examples[192:240]            # identical bases to original run
    alt_donors=examples[48:96]        # disjoint alternate donors

    rows=[]; profiles=[]
    for idx,(e,d) in enumerate(zip(test,alt_donors)):
        ids=cb.build_ids(tok,e,id0,id1); dids=cb.build_ids(tok,d,id0,id1)
        if ids.shape[1]!=dids.shape[1]:
            raise RuntimeError("Matched fixed-format sequences changed token length")
        N=ids.shape[1]
        cap.values.clear()
        with torch.inference_mode():
            o=model(input_ids=ids,output_attentions=True,output_hidden_states=True,use_cache=False,return_dict=True)
        base_h=o.hidden_states[-1][0,-1,:].detach().float().cpu().numpy()
        base_logits=o.logits[0,-1,:].detach().float().cpu().numpy()
        base_margin=float(base_logits[id1]-base_logits[id0])
        Ah=o.attentions[mid][0].detach().float().cpu().numpy()
        hn=base.head_residual_norms(model,a.model_key,mid,cap.values[mid],0)
        contrib=cb.contribution_target(Ah,hn)
        base_mid=o.hidden_states[mid+1][0].detach().float().cpu()
        del o

        cap.values.clear()
        with torch.inference_mode():
            od=model(input_ids=dids,output_hidden_states=True,use_cache=False,return_dict=True)
        donor_mid=od.hidden_states[mid+1][0].detach().float().cpu()
        del od
        patch_delta=torch.norm(donor_mid-base_mid,dim=1).numpy()+1e-9

        poss=list(range(N))
        batch_ids=ids.repeat(N,1)
        hf,mg,_=cb.patch_run(model,layer,batch_ids,donor_mid,poss,id0,id1,probe=None)
        causal=np.linalg.norm(hf-base_h[None,:],axis=1)/patch_delta
        dmargin=np.abs(mg-base_margin)

        valid=np.arange(max(1,N-1))
        rho=float(spearmanr(contrib[valid],causal[valid],nan_policy="omit").statistic)
        partial=partial_rank_position(contrib[valid],causal[valid])
        rho_margin=float(spearmanr(contrib[valid],dmargin[valid],nan_policy="omit").statistic)
        rows.append(dict(example_id=int(e["example_id"]),N=int(N),
                         rho_contrib_causal=rho,partial_rho_position=partial,
                         rho_contrib_modelmargin=rho_margin))
        profiles.append(pd.DataFrame({
            "example_id":int(e["example_id"]),
            "source":np.arange(N),
            "contribution":contrib,
            "causal_hidden_alt":causal,
            "delta_model_margin_alt":dmargin,
        }))
        print(a.model_key,idx+1,"/",len(test),"rho",round(rho,3),"partial",round(partial,3),flush=True)
        gc.collect()

    cap.close()
    df=pd.DataFrame(rows); df.to_csv(out/"donor2_example_metrics.csv",index=False)
    pd.concat(profiles,ignore_index=True).to_csv(out/"donor2_profiles.csv",index=False)

    rr=np.random.default_rng(SEED+99)
    vals=df.rho_contrib_causal.to_numpy(float)
    pvals=df.partial_rho_position.to_numpy(float)
    boots=[float(np.median(rr.choice(vals,size=len(vals),replace=True))) for _ in range(3000)]
    pboots=[float(np.median(rr.choice(pvals,size=len(pvals),replace=True))) for _ in range(3000)]
    summary={
      "model":a.model_key,"middle_layer":mid,"n_test":len(df),
      "donor_definition":"same original base examples; alternate donors are shuffled training examples 48:96",
      "median_rho_contribution_vs_causal_hidden":float(df.rho_contrib_causal.median()),
      "median_rho_bootstrap_95":np.quantile(boots,[.025,.5,.975]).tolist(),
      "median_partial_rho_controlling_position":float(df.partial_rho_position.median()),
      "median_partial_rho_bootstrap_95":np.quantile(pboots,[.025,.5,.975]).tolist(),
      "median_rho_contribution_vs_model_margin_effect":float(df.rho_contrib_modelmargin.median())
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
