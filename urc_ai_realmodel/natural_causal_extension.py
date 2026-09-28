#!/usr/bin/env python3
"""
URC-AI natural-language causal replication + scaling extension.

Task: counterfactual entity tracking in natural language.

Each matched pair has the same A/B unigram counts:
  - one CAUSAL room label is swapped (this changes the answer)
  - one DISTRACTOR room label is swapped oppositely (does not change the answer)

Example logic:
  lantern -> room X
  key -> room containing lantern
  lantern moves again
  coin -> opposite(X)  [counterbalances A/B count]
  ask: where is key?  Answer = X

Measurements:
  1) zero-shot constrained A/B readout
  2) early/mid/final linear recoverability probes
  3) middle-layer one-position activation patching profile
  4) raw-attention vs value/output-aware contribution vs patching ranks
  5) scalar-logit Jacobian gradient norms vs finite patching
  6) causal-token vs matched distractor-token intervention effect
  7) early/mid/late causal-token patch effect (layer dependence)
  8) counterfactual directional test: does patching the causal token move
     the A/B logit margin toward the donor answer?

Models:
  qwen05  Qwen2.5-0.5B-Instruct
  qwen15  Qwen2.5-1.5B-Instruct (larger-model replication)
  pythia  Pythia-160M (cross-architecture replication)
"""
from __future__ import annotations
import argparse, gc, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
import recoverability_gap as base

SEED=20260927
MODELS={
 "qwen05":("Qwen/Qwen2.5-0.5B-Instruct","qwen"),
 "qwen15":("Qwen/Qwen2.5-1.5B-Instruct","qwen"),
 "pythia":("EleutherAI/pythia-160m","pythia"),
}

SOURCES=["lantern","marker","vase","parcel","medal","token"]
TARGETS=["key","ring","note","badge","ticket","card"]
DISTRACTORS=["coin","book","cup","map","rope","stone"]
VERB1=["move","carry","place","transfer"]
COPY_PHRASES=[
 "Then move the {target} into whichever room currently contains the {source}.",
 "Then place the {target} in the same room as the {source}.",
 "Next transfer the {target} to the room that currently holds the {source}.",
 "Next put the {target} wherever the {source} is at that moment.",
]

def label_ids(tok):
    for a,b in [(" A"," B"),("A","B"),("\nA","\nB")]:
        ia=tok.encode(a,add_special_tokens=False); ib=tok.encode(b,add_special_tokens=False)
        if len(ia)==1 and len(ib)==1 and ia[0]!=ib[0]:
            return ia[0],ib[0],a,b
    raise RuntimeError("Could not find single-token A/B labels")

def story(source,target,distractor,causal,later,verb,copy_phrase,variant=0):
    opp="B" if causal=="A" else "A"
    text=(
      "Two rooms are labeled A and B. Track the objects exactly. "
      f"Initially, the {source} is in room A, the {target} is in room B, "
      f"and the {distractor} is in room A. "
      f"First, {verb} the {source} to room {causal}. "
      + copy_phrase.format(target=target,source=source) + " "
      f"After that, move the {source} to room {later}. "
      f"Finally, move the {distractor} to room {opp}. "
      f"The {target} does not move again. "
      f"Which room contains the {target}? Answer only A or B."
    )
    return text

def make_pairs(n_pairs=160):
    rng=random.Random(SEED); pairs=[]
    for pid in range(n_pairs):
        source=SOURCES[pid%len(SOURCES)]
        target=TARGETS[(pid*3+1)%len(TARGETS)]
        distractor=DISTRACTORS[(pid*5+2)%len(DISTRACTORS)]
        later="A" if rng.random()<.5 else "B"
        verb=VERB1[pid%len(VERB1)]
        cp=COPY_PHRASES[(pid//2)%len(COPY_PHRASES)]
        a=story(source,target,distractor,"A",later,verb,cp,0)
        b=story(source,target,distractor,"B",later,verb,cp,1)
        pairs.append({"pair_id":pid,"A":a,"B":b,"source":source,"target":target,"distractor":distractor})
    return pairs

def probe():
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200,C=1.0,random_state=SEED))

class Capture:
    def __init__(self,model,family):
        self.family=family; self.values={}; self.handles=[]
        if family=="qwen":
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

def head_norms(model,family,l,captured):
    return base.head_residual_norms(model,family,l,captured,0)

def contribution_target(Ah,norms):
    return np.einsum("hi,ih->i",Ah[:,-1,:],norms,optimize=True)

def build_single(model,tok,cap,text,idA,idB,need_attn=True):
    ids=tok.encode(text,add_special_tokens=False)
    x=torch.tensor([ids],dtype=torch.long)
    cap.values.clear()
    with torch.inference_mode():
        o=model(input_ids=x,output_attentions=need_attn,output_hidden_states=True,use_cache=False,return_dict=True)
    hf=o.hidden_states[-1][0,-1,:].detach().float().cpu().numpy()
    lg=o.logits[0,-1,:].detach().float().cpu().numpy()
    margin=float(lg[idB]-lg[idA]); pred="B" if margin>0 else "A"
    return x,o,hf,margin,pred

def patch_batch(model,layer,ids,donor_h,positions,idA,idB):
    B=len(positions); xx=ids.repeat(B,1); captured={}
    def h(mod,inp,out):
        if isinstance(out,tuple):
            z=out[0].clone()
            for bi,p in enumerate(positions):z[bi,p,:]=donor_h[p,:].to(z.device,z.dtype)
            return (z,)+tuple(out[1:])
        z=out.clone()
        for bi,p in enumerate(positions):z[bi,p,:]=donor_h[p,:].to(z.device,z.dtype)
        return z
    handle=layer.register_forward_hook(h)
    try:
        with torch.inference_mode():
            o=model(input_ids=xx,output_hidden_states=True,use_cache=False,return_dict=True)
    finally:handle.remove()
    hf=o.hidden_states[-1][:,-1,:].detach().float().cpu().numpy()
    lg=o.logits[:,-1,:].detach().float().cpu().numpy()
    mg=lg[:,idB]-lg[:,idA]
    del o
    return hf,mg

def patch_multi_layer_single_token(model,layers,ids,donors,pos,idA,idB):
    out=[]
    for li in layers:
        donor=donors[li]
        _,mg=patch_batch(model,li,ids,donor,[pos],idA,idB)
        out.append(float(mg[0]))
    return out

def jacobian_margin_profile(model,layer,ids,idA,idB):
    # Params need no gradients; detach at selected layer and make that activation leaf-like.
    old=[p.requires_grad for p in model.parameters()]
    for p in model.parameters():p.requires_grad_(False)
    box={}
    def h(mod,inp,out):
        if isinstance(out,tuple):
            z=out[0].detach().requires_grad_(True); z.retain_grad(); box["z"]=z
            return (z,)+tuple(out[1:])
        z=out.detach().requires_grad_(True); z.retain_grad(); box["z"]=z
        return z
    handle=layer.register_forward_hook(h)
    try:
        o=model(input_ids=ids,use_cache=False,return_dict=True)
        margin=o.logits[0,-1,idB]-o.logits[0,-1,idA]
        margin.backward()
        g=box["z"].grad[0].detach().float().cpu().numpy()
        prof=np.linalg.norm(g,axis=1)
    finally:
        handle.remove()
        for p,r in zip(model.parameters(),old):p.requires_grad_(r)
        model.zero_grad(set_to_none=True)
    return prof

def rank_corr(x,y):
    r=spearmanr(x,y,nan_policy="omit").statistic
    return float(r) if np.isfinite(r) else np.nan

def load(mid):
    tok=torch.hub # sentinel to avoid accidental shadowing
    from transformers import AutoTokenizer,AutoModelForCausalLM
    t=AutoTokenizer.from_pretrained(mid,use_fast=True)
    try:m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation="eager")
    except TypeError:
        m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,low_cpu_mem_usage=True)
        try:m.config._attn_implementation="eager"
        except:pass
    m.eval(); return t,m

def bootstrap_median(v,n=2000):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    rng=np.random.default_rng(SEED+55)
    b=[float(np.median(rng.choice(v,size=len(v),replace=True))) for _ in range(n)]
    return np.quantile(b,[.025,.5,.975]).tolist()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-key",choices=MODELS,required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--train-pairs",type=int,default=80)
    ap.add_argument("--test-pairs",type=int,default=None); ap.add_argument("--patch-batch",type=int,default=16)
    a=ap.parse_args()
    if a.test_pairs is None:a.test_pairs=16 if a.model_key=="qwen15" else 24
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    mid,family=MODELS[a.model_key]; tok,model=load(mid); idA,idB,sA,sB=label_ids(tok)
    cap=Capture(model,family); L=len(cap.layers); layers=[0,L//2,L-1]; middle=layers[1]
    pairs=make_pairs(a.train_pairs+a.test_pairs+8)

    # Train probes on both variants of training pairs.
    X={l:[] for l in layers};Y=[];texts=[]
    for p in pairs[:a.train_pairs]:
      for lab in ["A","B"]:
        ids,o,hf,mg,pred=build_single(model,tok,cap,p[lab],idA,idB,need_attn=False)
        for l in layers:X[l].append(o.hidden_states[l+1][0,-1,:].detach().float().cpu().numpy())
        Y.append(0 if lab=="A" else 1);texts.append(p[lab]);del o
    probes={}
    for l in layers:
        probes[l]=probe();probes[l].fit(np.stack(X[l]),np.array(Y))
    # unigram count baseline
    vec=CountVectorizer(lowercase=False,ngram_range=(1,1),token_pattern=r"(?u)\b\w+\b")
    Z=vec.fit_transform(texts); rawclf=LogisticRegression(max_iter=1200,C=1.0,random_state=SEED).fit(Z,Y)

    exrows=[];profiles=[]
    for qi,p in enumerate(pairs[a.train_pairs:a.train_pairs+a.test_pairs]):
        # base A, donor B; exact matched counterfactual
        text=p["A"]; donor_text=p["B"]; target=0; donor_target=1
        ids,o,base_h,base_margin,base_pred=build_single(model,tok,cap,text,idA,idB,need_attn=True)
        dids,od,donor_final,donor_margin,donor_pred=build_single(model,tok,cap,donor_text,idA,idB,need_attn=False)
        if ids.shape!=dids.shape:
            print("skip length mismatch",qi,ids.shape,dids.shape,flush=True);del o,od;continue
        arrA=ids[0].numpy();arrB=dids[0].numpy();diff=np.flatnonzero(arrA!=arrB)
        if len(diff)!=2:
            print("skip diff count",qi,len(diff),flush=True);del o,od;continue
        causal_pos=int(diff[0]);distractor_pos=int(diff[1])
        # verify textual order: causal counterfactual appears before counterbalancing distractor
        # contribution/raw ranking at middle
        Ah=o.attentions[middle][0].detach().float().cpu().numpy()
        raw=Ah[:,-1,:].mean(0)
        hn=head_norms(model,family,middle,cap.values[middle])
        contrib=contribution_target(Ah,hn)
        base_mid=o.hidden_states[middle+1][0].detach().float().cpu()
        donor_h_by_layer={li:od.hidden_states[li+1][0].detach().float().cpu() for li in layers}
        donor_mid=donor_h_by_layer[middle]
        patch_norm=torch.norm(donor_mid-base_mid,dim=1).numpy()+1e-9

        causal=np.zeros(ids.shape[1]);dmargin=np.zeros(ids.shape[1])
        for s in range(0,ids.shape[1],a.patch_batch):
            poss=list(range(s,min(ids.shape[1],s+a.patch_batch)))
            hf,mg=patch_batch(model,cap.layers[middle],ids,donor_mid,poss,idA,idB)
            causal[poss]=np.linalg.norm(hf-base_h[None,:],axis=1)/patch_norm[poss]
            dmargin[poss]=mg-base_margin
        jac=jacobian_margin_profile(model,cap.layers[middle],ids,idA,idB)

        # Layer dependence of causal-token intervention.
        layer_margins=patch_multi_layer_single_token(model,[cap.layers[l] for l in layers],ids,
                                                     {cap.layers[l]:donor_h_by_layer[l] for l in []},causal_pos,idA,idB) if False else None
        lm=[]
        for li in layers:
            _,mg=patch_batch(model,cap.layers[li],ids,donor_h_by_layer[li],[causal_pos],idA,idB)
            lm.append(float(mg[0]))

        # probe and readout controls
        probe_preds={};probe_probs={}
        for li in layers:
            h=o.hidden_states[li+1][0,-1,:].detach().float().cpu().numpy()
            probe_preds[li]=int(probes[li].predict(h[None,:])[0])
            probe_probs[li]=float(probes[li].predict_proba(h[None,:])[0,1])
        raw_pred=int(rawclf.predict(vec.transform([text]))[0])

        # donor direction is B => increasing margin is toward donor answer
        causal_shift=float(dmargin[causal_pos]); distractor_shift=float(dmargin[distractor_pos])
        prof=pd.DataFrame({"pair_id":p["pair_id"],"source":np.arange(len(raw)),"raw_attention":raw,
                           "contribution":contrib,"patch_hidden":causal,"patch_margin_signed":dmargin,"jacobian_margin_grad":jac})
        profiles.append(prof)
        valid=np.arange(len(raw)-1)
        exrows.append({
          "pair_id":p["pair_id"],"N":len(raw),"model_pred":1 if base_pred=="B" else 0,"model_correct":int(base_pred=="A"),
          "raw_unigram_pred":raw_pred,"raw_unigram_correct":int(raw_pred==0),
          **{f"probe_L{li}_pred":probe_preds[li] for li in layers},
          **{f"probe_L{li}_correct":int(probe_preds[li]==0) for li in layers},
          "rho_raw_patch":rank_corr(raw[valid],causal[valid]),
          "rho_contrib_patch":rank_corr(contrib[valid],causal[valid]),
          "rho_jac_patch":rank_corr(jac[valid],np.abs(dmargin[valid])),
          "rho_jac_hiddenpatch":rank_corr(jac[valid],causal[valid]),
          "rho_contrib_jac":rank_corr(contrib[valid],jac[valid]),
          "causal_pos":causal_pos,"distractor_pos":distractor_pos,
          "causal_patch_hidden":float(causal[causal_pos]),"distractor_patch_hidden":float(causal[distractor_pos]),
          "causal_patch_margin_shift":causal_shift,"distractor_patch_margin_shift":distractor_shift,
          "causal_moves_toward_donor":int(causal_shift>0),
          "distractor_moves_toward_donor":int(distractor_shift>0),
          **{f"causal_shift_L{li}":float(m-lm[0]+(lm[0]-base_margin)) if False else float(m-base_margin) for li,m in zip(layers,lm)},
        })
        del o,od;gc.collect()
        print(a.model_key,qi+1,"/",a.test_pairs,"rhoC",round(exrows[-1]["rho_contrib_patch"],3),
              "rhoJ",round(exrows[-1]["rho_jac_patch"],3),"causal_shift",round(causal_shift,3),flush=True)
    cap.close()

    df=pd.DataFrame(exrows);df.to_csv(out/"natural_causal_examples.csv",index=False)
    pd.concat(profiles,ignore_index=True).to_csv(out/"natural_causal_profiles.csv",index=False)
    summary={
      "model":a.model_key,"model_id":mid,"family":family,"layers":layers,"n_test":len(df),
      "model_accuracy":float(df.model_correct.mean()),"raw_unigram_accuracy":float(df.raw_unigram_correct.mean()),
      "probe_accuracy":{str(li):float(df[f"probe_L{li}_correct"].mean()) for li in layers},
      "median_rho_raw_vs_patch":float(df.rho_raw_patch.median()),
      "median_rho_contribution_vs_patch":float(df.rho_contrib_patch.median()),
      "median_rho_jacobian_vs_margin_patch":float(df.rho_jac_patch.median()),
      "median_rho_jacobian_vs_hidden_patch":float(df.rho_jac_hiddenpatch.median()),
      "median_rho_contribution_vs_jacobian":float(df.rho_contrib_jac.median()),
      "rho_contribution_patch_bootstrap95":bootstrap_median(df.rho_contrib_patch),
      "rho_jac_patch_bootstrap95":bootstrap_median(df.rho_jac_patch),
      "causal_patch_toward_donor_fraction":float(df.causal_moves_toward_donor.mean()),
      "distractor_patch_toward_donor_fraction":float(df.distractor_moves_toward_donor.mean()),
      "median_causal_hidden_effect":float(df.causal_patch_hidden.median()),
      "median_distractor_hidden_effect":float(df.distractor_patch_hidden.median()),
      "median_causal_margin_shift":float(df.causal_patch_margin_shift.median()),
      "median_distractor_margin_shift":float(df.distractor_patch_margin_shift.median()),
      "layer_causal_shift_median":{str(li):float(df[f"causal_shift_L{li}"].median()) for li in layers},
      "task_definition":"Matched natural-language state-copying stories; causal A/B swap flips key location; distractor A/B token is counter-swapped to preserve unigram counts.",
      "claim_boundary":"Activation patching and scalar-logit Jacobian gradients are intervention/local-sensitivity diagnostics. They do not by themselves establish a complete causal graph of the model."
    }
    # paired relevant-vs-distractor effect tests
    try:
        summary["wilcoxon_causal_gt_distractor_hidden_p"]=float(wilcoxon(df.causal_patch_hidden,df.distractor_patch_hidden,alternative="greater").pvalue)
        summary["wilcoxon_causal_gt_distractor_abs_margin_p"]=float(wilcoxon(np.abs(df.causal_patch_margin_shift),np.abs(df.distractor_patch_margin_shift),alternative="greater").pvalue)
    except Exception:
        summary["wilcoxon_causal_gt_distractor_hidden_p"]=None;summary["wilcoxon_causal_gt_distractor_abs_margin_p"]=None
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
