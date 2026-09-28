#!/usr/bin/env python3
"""
URC-AI S4 bridge: residual-contribution transport.

Positive edge weight:
  C_ij = sum_h A_ij^h || W_O^(h) v_j^(h) ||_2.

By triangle inequality this upper-bounds the norm of the summed per-edge
attention contribution in residual coordinates while retaining a positive
graph suitable for diffusion analysis. Rows are normalized after construction.

The exact frozen Gate-02 reject-option criteria are reused unchanged.
"""
from __future__ import annotations
import argparse, gc, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.set_num_interop_threads(1)

MODELS={"qwen":"Qwen/Qwen2.5-0.5B-Instruct","pythia":"EleutherAI/pythia-160m"}
TASKS={
"arithmetic":"Solve carefully. A warehouse starts with 4,120 pallets. On Monday 14 shipments of 65 pallets arrive and 380 pallets are dispatched. On Tuesday 22 shipments of 45 pallets arrive and 515 pallets are dispatched. On Wednesday one-fourth of the current stock is transferred out. How many pallets remain? Show the arithmetic logic explicitly.",
"retrieval":"Use this reference log. [001 Alpha=Operational] [002 Beta=Degraded] [003 Gamma=Offline] [004 Delta=Operational] [005 Epsilon=Standby] [006 Zeta=Maintenance] [007 Eta=Operational] [008 Theta=Offline]. Identify which systems are either Degraded or Maintenance and explain how the answer follows from the log."
}
FILLER="Archived neutral context. This sentence is deliberately irrelevant to the task, contains no answer-bearing quantities or system labels, and exists only to control context length. "

def exact_ids(tok,core,N):
    c=tok.encode(core,add_special_tokens=False); f=tok.encode(FILLER,add_special_tokens=False)
    if len(c)>=N: ids=c[-N:]
    else:
        need=N-len(c); ids=(f*((need+len(f)-1)//len(f)))[:need]+c
    return torch.tensor([ids],dtype=torch.long)

def top_p(A,p):
    B=np.zeros_like(A,dtype=float)
    for i in range(len(A)):
        x=A[i,:i+1]; order=np.argsort(-x); k=np.searchsorted(np.cumsum(x[order]),p,side="left")+1
        keep=order[:max(1,int(k))]; B[i,keep]=x[keep]; s=B[i].sum()
        if s>0:B[i]/=s
    return B

def puncture(A,idx):
    k=np.ones(len(A),bool); k[idx]=False
    return A[np.ix_(k,k)]

def normalize_rows(C):
    s=C.sum(1,keepdims=True)
    return np.divide(C,s,out=np.zeros_like(C),where=s>1e-15)

def graph(A):
    W=(A+A.T)/2
    deg=W.sum(1); keep=deg>1e-14; W=W[np.ix_(keep,keep)]; deg=W.sum(1)
    if len(W)<3:return None
    mean=deg.mean(); top=int(np.argmax(deg)); pi=deg/deg.sum()
    inv=1/np.sqrt(np.maximum(deg,1e-300)); S=(inv[:,None]*W)*inv[None,:]
    L=np.eye(len(W))-S; L=(L+L.T)/2
    e=scipy.linalg.eigvalsh(L,overwrite_a=True,check_finite=False)
    e[np.abs(e)<1e-11]=0; e=np.maximum(e,0)
    pos=e[e>1e-11]; lam=float(pos[0]) if len(pos) else np.nan
    return dict(eig=e,lambda1=lam,t_relax=(1/lam if np.isfinite(lam) and lam>0 else np.inf),
                hub_index=float(deg[top]/mean),max_degree_node=top,
                stationary_pi_max=float(pi[top]),stationary_ipr=float(np.sum(pi*pi)),
                stationary_effective_nodes=float(1/np.sum(pi*pi)))

def fit(x,y):
    b,a=np.polyfit(x,y,1); yh=a+b*x; den=np.sum((y-y.mean())**2)
    return float(-2*b),float(1-np.sum((y-yh)**2)/den) if den>0 else np.nan

def plateau(e,min_dec=1.5,min_r2=.995,max_range=.15):
    pos=e[e>1e-11]
    if not len(pos):return dict(status="NO_STABLE_PLATEAU",d_s=None,r2=None,t1=None,t2=None)
    lam=float(pos[0]); ir=.25/lam; t=np.geomspace(5,1e4,220); c0=np.sum(e<=1e-11)
    p=np.exp(-np.outer(t,e)).mean(1)-c0/len(e); floor=max(1e-15,float(p.max())*1e-12)
    m=(p>floor)&(t<=ir); t,p=t[m],p[m]
    if len(t)<12 or t[-1]/t[0]<10**min_dec:
        return dict(status="INSUFFICIENT_DYNAMIC_RANGE",d_s=None,r2=None,t1=None,t2=None)
    lx,ly=np.log10(t),np.log10(p); best=None
    for i in range(len(t)):
      for j in range(i+8,len(t)):
        if lx[j]-lx[i]<min_dec:continue
        ds,r2=fit(lx[i:j+1],ly[i:j+1])
        if not np.isfinite(r2) or r2<min_r2 or ds<=0:continue
        loc=[]
        for k in range(i,j-7):
            d,r=fit(lx[k:k+9],ly[k:k+9])
            if np.isfinite(r) and r>=.98:loc.append(d)
        if len(loc)<3 or max(loc)-min(loc)>max_range:continue
        score=(lx[j]-lx[i],r2,-(max(loc)-min(loc)))
        rec=dict(status="PLATEAU_ACCEPTED",d_s=ds,r2=r2,t1=float(t[i]),t2=float(t[j]))
        if best is None or score>best[0]:best=(score,rec)
    return best[1] if best else dict(status="NO_STABLE_PLATEAU",d_s=None,r2=None,t1=None,t2=None)

class Capture:
    def __init__(self,model,key):
        self.key=key; self.handles=[]; self.values={}
        if key=="qwen":
            self.layers=model.model.layers
            for i,layer in enumerate(self.layers):
                def mk(ii):
                    def h(mod,inp,out):
                        self.values[ii]=out.detach().float().cpu()
                    return h
                self.handles.append(layer.self_attn.v_proj.register_forward_hook(mk(i)))
        else:
            self.layers=model.gpt_neox.layers
            for i,layer in enumerate(self.layers):
                def mk(ii):
                    def h(mod,inp,out):
                        self.values[ii]=out.detach().float().cpu()
                    return h
                self.handles.append(layer.attention.query_key_value.register_forward_hook(mk(i)))
    def close(self):
        for h in self.handles:h.remove()

def projected_head_norms(model,key,layer_i,captured):
    if key=="qwen":
        att=model.model.layers[layer_i].self_attn
        Hq=model.config.num_attention_heads; Hkv=model.config.num_key_value_heads
        D=captured.shape[-1]//Hkv
        V=captured[0].numpy().reshape(captured.shape[1],Hkv,D)
        V=np.repeat(V,Hq//Hkv,axis=1) # [N,Hq,D]
        Wo=att.o_proj.weight.detach().float().cpu().numpy().reshape(model.config.hidden_size,Hq,D)
    else:
        att=model.gpt_neox.layers[layer_i].attention
        Hq=model.config.num_attention_heads; D=model.config.hidden_size//Hq
        qkv=captured[0].numpy().reshape(captured.shape[1],Hq,3,D)
        V=qkv[:,:,2,:]
        Wo=att.dense.weight.detach().float().cpu().numpy().reshape(model.config.hidden_size,Hq,D)
    # residual-coordinate norm for each source token and head
    # || W_h v_jh ||^2 = v^T (W_h^T W_h) v
    G=np.einsum("ohd,ohe->hde",Wo,Wo,optimize=True)
    sq=np.einsum("jhd,hde,jhe->jh",V,G,V,optimize=True)
    return np.sqrt(np.maximum(sq,0))

def contribution_graph(Aheads,source_head_norm):
    # Aheads [H,N,N], norms [N,H]
    C=np.einsum("hij,jh->ij",Aheads,source_head_norm,optimize=True)
    return normalize_rows(C)

def load(mid):
    tok=AutoTokenizer.from_pretrained(mid,use_fast=True)
    try:model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation="eager")
    except TypeError:
        model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,low_cpu_mem_usage=True)
        try:model.config._attn_implementation="eager"
        except:pass
    model.eval(); return tok,model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-key",choices=MODELS,required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tok,model=load(MODELS[a.model_key]); cap=Capture(model,a.model_key); rows=[]
    meta={"model":MODELS[a.model_key],"graph_definition":"C_ij=sum_h A_ij^h ||W_O^h v_j^h||_2; row normalized",
          "lengths":[256,512,1024],"top_p":[.85,.90,.95],"plateau":{"decades":1.5,"r2":.995,"local_range":.15}}
    for task,core in TASKS.items():
      for N in [256,512,1024]:
        ids=exact_ids(tok,core,N); mask=torch.ones_like(ids); cap.values.clear(); t0=time.time()
        with torch.inference_mode():
            o=model(input_ids=ids,attention_mask=mask,output_attentions=True,use_cache=False,return_dict=True)
        print(a.model_key,task,N,"forward",round(time.time()-t0,2),flush=True)
        for l,at in enumerate(o.attentions):
            Ah=at[0].detach().float().cpu().numpy()
            hn=projected_head_norms(model,a.model_key,l,cap.values[l])
            C=contribution_graph(Ah,hn)
            for fname,B in {"RAW":C,"TOPP_085":top_p(C,.85),"TOPP_090":top_p(C,.90),"TOPP_095":top_p(C,.95)}.items():
                variants={"NATIVE":B,"PUNCTURE_TOKEN0":puncture(B,0)}
                top=int(np.argmax(((B+B.T)/2).sum(1)))
                if top!=0:variants["PUNCTURE_TOPDEG"]=puncture(B,top)
                for vname,X in variants.items():
                    g=graph(X)
                    if g is None:continue
                    e=g.pop("eig"); p=plateau(e)
                    rows.append(dict(model=a.model_key,task=task,N=N,layer=l,filter=fname,variant=vname,
                                     mean_raw_contribution=float((C.sum(1)).mean()),**g,**p))
        del o; gc.collect()
    cap.close()
    df=pd.DataFrame(rows); df.to_csv(out/"contribution_results.csv",index=False)
    # threshold-stability and cross-N candidates
    st=[]
    for v in ["NATIVE","PUNCTURE_TOKEN0","PUNCTURE_TOPDEG"]:
      dv=df[(df.variant==v)&df["filter"].isin(["TOPP_085","TOPP_090","TOPP_095"])]
      for (task,N,l),g in dv.groupby(["task","N","layer"]):
        ok=len(g)==3 and (g.status=="PLATEAU_ACCEPTED").all()
        span=float(g.d_s.max()-g.d_s.min()) if ok else np.nan
        st.append(dict(model=a.model_key,task=task,N=int(N),layer=int(l),variant=v,threshold_stable=bool(ok and span<=.15),
                       ds_span=span,ds_mean=float(g.d_s.mean()) if ok else np.nan,t_relax_mean=float(g.t_relax.mean())))
    st=pd.DataFrame(st); st.to_csv(out/"contribution_stability.csv",index=False)
    candidates=[]
    for v in ["NATIVE","PUNCTURE_TOKEN0","PUNCTURE_TOPDEG"]:
      sv=st[st.variant==v]
      for (task,l),g in sv.groupby(["task","layer"]):
        g=g.sort_values("N")
        if set(g.N)=={256,512,1024} and g.threshold_stable.all():
            spread=float(g.ds_mean.max()-g.ds_mean.min()); tr=g.t_relax_mean.to_numpy()
            if spread<=.15 and tr[0]<tr[1]<tr[2]:
                candidates.append(dict(variant=v,task=task,layer=int(l),ds_mean=float(g.ds_mean.mean()),crossN_span=spread,
                                       t256=float(tr[0]),t512=float(tr[1]),t1024=float(tr[2])))
    anyp=bool((df.status=="PLATEAU_ACCEPTED").any())
    if any(x["variant"]=="NATIVE" for x in candidates):branch="A_CONTRIBUTION_BACKBONE"
    elif candidates:branch="H_CONTRIBUTION_HUB_CONTAMINATED"
    elif anyp:branch="B_CONTRIBUTION_LOCAL_ARTIFACT"
    else:branch="C_C_CONTRIBUTION_NONMANIFOLD"
    summary={"model":a.model_key,"branch":branch,"accepted_plateaux":int((df.status=="PLATEAU_ACCEPTED").sum()),
             "threshold_stable_cases":int(st.threshold_stable.sum()),"scale_candidates":candidates,"meta":meta}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
