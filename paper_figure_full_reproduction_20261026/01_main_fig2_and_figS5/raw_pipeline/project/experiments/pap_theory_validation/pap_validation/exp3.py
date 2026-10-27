from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    AREAS9,
    apply_fln_delta,
    edge_to_delta_j,
    frequency_grid,
    is_stable,
    load_model,
    network_features,
    prediction_metrics,
    save_figure,
    save_json,
    transfer_power,
)
from .io import prepare_output


Z0=np.array([10.,6.,-.5,1.]); ZS=np.array([3.,4.,.8,.7])


def _input_trials(n,seed,negative=False):
    rng=np.random.default_rng(seed); cov=np.eye(4)
    if not negative: cov[0,2]=cov[2,0]=.6
    u=rng.multivariate_normal(np.zeros(4),cov,size=n)*(.08 if negative else .2)
    z=Z0+u*ZS; z=np.clip(z,[5.2,1,-1.8,.15],[19.5,15,1,3]); return z,(z-Z0)/ZS


def _features_for_transfer(j,transfer,zs,freqs,inp,out):
    vals=np.empty((len(zs),2)); valid=np.zeros(len(zs),bool)
    for i,z in enumerate(zs):
        f,_=network_features(j,z,inp,out,freqs,transfer)
        vals[i]=f["peak_omega"],f["exponent"]; valid[i]=f["peak_valid"]
    return vals,valid


def _direct_stats(j,zs,freqs,inp,out):
    if not is_stable(j): return {"stable":False,"C_exact":np.nan,"rho_exact":np.nan,"p_value":np.nan,"n_valid":0},None,None
    transfer=transfer_power(j,inp,out,freqs); vals,valid=_features_for_transfer(j,transfer,zs,freqs,inp,out)
    use=valid & np.isfinite(vals).all(1)
    if use.sum()<5: return {"stable":True,"C_exact":np.nan,"rho_exact":np.nan,"p_value":np.nan,"n_valid":int(use.sum())},transfer,vals
    pr=stats.pearsonr(vals[use,0],vals[use,1])
    return {"stable":True,"C_exact":float(np.cov(vals[use].T,ddof=1)[0,1]),"rho_exact":float(pr.statistic),"p_value":float(pr.pvalue),"n_valid":int(use.sum())},transfer,vals


def _linear_stats(j,sigma_u,freqs,inp,out):
    if not is_stable(j): return np.array([np.nan,np.nan])
    tr=transfer_power(j,inp,out,freqs); q=4; grad=np.empty((2,q)); h=2e-3
    for k in range(q):
        up=np.zeros(q); up[k]=h
        yp=[]
        for u in (up,-up):
            f,_=network_features(j,Z0+ZS*u,inp,out,freqs,tr); yp.append([f["peak_omega"],f["exponent"]])
        grad[:,k]=(np.asarray(yp[0])-np.asarray(yp[1]))/(2*h)
    c=float(grad[0]@sigma_u@grad[1]); vp=float(grad[0]@sigma_u@grad[0]); va=float(grad[1]@sigma_u@grad[1])
    return np.array([c,c/np.sqrt(max(vp*va,1e-30))])


def _resolvent_perturbation_norm(j,dj):
    vals=[]; eye=np.eye(len(j))
    for f in np.geomspace(5,20,12):
        r=np.linalg.inv(1j*2*np.pi*f/1000*eye-j); vals.append(np.linalg.norm(r@dj,2))
    return float(max(vals))


def _edge_kernel(model,sigma,freqs,inp,out,edges,h=.02):
    base=_linear_stats(model.J,sigma,freqs,inp,out); rows=[]
    for target,source in edges:
        dj=edge_to_delta_j(model,target,source,model.fln[target,source])
        plus=_linear_stats(model.J+h*dj,sigma,freqs,inp,out); minus=_linear_stats(model.J-h*dj,sigma,freqs,inp,out)
        der=(plus-minus)/(2*h)
        rows.append({"target":model.areas[target],"source":model.areas[source],"target_idx":target,"source_idx":source,
            "weight":model.fln[target,source],"kernel_C_per_multiplier":der[0],"kernel_rho_per_multiplier":der[1]})
    return base,pd.DataFrame(rows)


def _evaluate_delta(model,delta_w,coeffs,kind,pid,zs,sigma,freqs,inp,out,base_exact,base_lin,hdir=.05):
    dj=apply_fln_delta(model,delta_w)-model.J; jnew=model.J+dj
    direct,_,_= _direct_stats(jnew,zs,freqs,inp,out)
    nz=list(zip(*np.nonzero(delta_w))); first=np.zeros(2)
    for t,s in nz:
        row=coeffs[(coeffs.target_idx==t)&(coeffs.source_idx==s)].iloc[0]
        multiplier=delta_w[t,s]/model.fln[t,s]
        first += multiplier*np.array([row.kernel_C_per_multiplier,row.kernel_rho_per_multiplier])
    plus=_linear_stats(model.J+hdir*dj,sigma,freqs,inp,out); minus=_linear_stats(model.J-hdir*dj,sigma,freqs,inp,out)
    second=(plus-2*base_lin+minus)/hdir**2
    pred1=base_lin+first; pred2=pred1+.5*second
    return {"perturbation_id":pid,"kind":kind,"n_edges":len(nz),"edges_json":json.dumps([(model.areas[s],model.areas[t],float(model.fln[t,s])) for t,s in nz]),
        "delta_j_norm":float(np.linalg.norm(dj)),"resolvent_delta_norm":_resolvent_perturbation_norm(model.J,dj),
        **direct,"delta_C_observed":direct["C_exact"]-base_exact["C_exact"],"delta_rho_observed":direct["rho_exact"]-base_exact["rho_exact"],
        "C1_perturbed":float(_linear_stats(jnew,sigma,freqs,inp,out)[0]) if direct["stable"] else np.nan,
        "C_first":float(pred1[0]),"rho_first":float(pred1[1]),"C_second":float(pred2[0]),"rho_second":float(pred2[1]),
        "delta_C_first":float(pred1[0]-base_lin[0]),"delta_rho_first":float(pred1[1]-base_lin[1]),
        "delta_C_second":float(pred2[0]-base_lin[0]),"delta_rho_second":float(pred2[1]-base_lin[1]),
        "first_order_cancellation":float(abs(first[0])/(np.sum([abs((delta_w[t,s]/model.fln[t,s])*coeffs[(coeffs.target_idx==t)&(coeffs.source_idx==s)].iloc[0].kernel_C_per_multiplier) for t,s in nz])+1e-15))}


def run(cfg: dict) -> Path:
    started=time.time(); out=prepare_output(cfg,"exp3_structural_kernel"); ecfg=cfg["exp3"]; seed=int(cfg["seed"])
    model=load_model(Path(cfg["model_dir"]),AREAS9); freqs=frequency_grid(); inp,outidx=0,5
    zs,u=_input_trials(int(ecfg["n_input_trials"]),seed+300); sigma=np.cov(u,rowvar=False,ddof=1)
    base_exact,base_transfer,base_vals=_direct_stats(model.J,zs,freqs,inp,outidx); base_lin,coeffs=_edge_kernel(model,sigma,freqs,inp,outidx,list(zip(*np.nonzero(model.fln))))
    coeffs.to_csv(out/"anatomical_kernel.csv",index=False)
    rows=[]; edges=list(zip(*np.nonzero(model.fln)))
    # Single-edge deletion and doubling.
    for action,mult in (("single_delete",-1.),("single_double",1.)):
      for eid,(t,s) in enumerate(edges):
        dw=np.zeros_like(model.fln); dw[t,s]=mult*model.fln[t,s]
        rows.append(_evaluate_delta(model,dw,coeffs,action,f"{action}_{eid:03d}",zs,sigma,freqs,inp,outidx,base_exact,base_lin))
    # Whole incoming and outgoing edge sets for each area.
    for area in range(len(model.areas)):
      for direction in ("incoming","outgoing"):
        dw=np.zeros_like(model.fln)
        if direction=="incoming": dw[area,:]=-model.fln[area,:]
        else: dw[:,area]=-model.fln[:,area]
        rows.append(_evaluate_delta(model,dw,coeffs,f"area_{direction}",f"{direction}_{model.areas[area]}",zs,sigma,freqs,inp,outidx,base_exact,base_lin))
    # Random 15-edge masks, sampled uniformly without replacement.
    rng=np.random.default_rng(seed+301); nm=int(ecfg["n_random_masks"]); masks=[]
    for mid in range(nm):
        ids=rng.choice(len(edges),15,replace=False); masks.append(ids)
        dw=np.zeros_like(model.fln)
        for k in ids:
            t,s=edges[k]; dw[t,s]=-model.fln[t,s]
        r=_evaluate_delta(model,dw,coeffs,"random15_delete",f"random15_{mid:04d}",zs,sigma,freqs,inp,outidx,base_exact,base_lin)
        r["split"]="pilot" if mid < int(.2*nm) else "heldout"; rows.append(r)
    pert=pd.DataFrame(rows); pert.to_csv(out/"trials.csv.gz",index=False,compression="gzip")
    theory_cols=[c for c in pert.columns if c.startswith("C") or c.startswith("rho") or c.startswith("delta_")]
    pert[["perturbation_id","kind","split","stable"]+[c for c in theory_cols if c not in ("kind","split","stable")]].to_csv(out/"theory_predictions.csv",index=False)
    pert.groupby("kind").agg(n=("perturbation_id","size"),stable_rate=("stable","mean"),mean_delta_C=("delta_C_observed","mean"),mean_abs_delta_C=("delta_C_observed",lambda x:np.mean(abs(x)))).reset_index().to_csv(out/"condition_summary.csv",index=False)
    pd.DataFrame({"trial_id":np.arange(len(zs)),"seed":seed+300,**{f"z{i}":zs[:,i] for i in range(4)}}).to_csv(out/"seeds.csv",index=False)
    # Explicit pullback identity checks.
    pb=[]
    for k in range(int(ecfg["derivative_directions"])):
        x=rng.normal(size=model.fln.shape)*(model.fln!=0)*1e-4; dj=apply_fln_delta(model,x)-model.J
        lhs=sum(coeffs.iloc[i].kernel_C_per_multiplier*(x[t,s]/model.fln[t,s]) for i,(t,s) in enumerate(edges))
        # The RHS is the same J-directional derivative evaluated directly.
        h=1.; rhs=(_linear_stats(model.J+h*dj,sigma,freqs,inp,outidx)[0]-_linear_stats(model.J-h*dj,sigma,freqs,inp,outidx)[0])/(2*h)
        pb.append({"direction":k,"anatomical_inner_product":lhs,"jacobian_directional":rhs,"abs_error":abs(lhs-rhs)})
    pd.DataFrame(pb).to_csv(out/"pullback_validation.csv",index=False)
    neg_z,neg_u=_input_trials(len(zs),seed+302,True); neg_exact,_,_=_direct_stats(model.J,neg_z,freqs,inp,outidx)
    held=pert[(pert.kind=="random15_delete")&(pert.split=="heldout")&pert.stable].copy()
    metrics={"baseline":{"exact":base_exact,"C1":base_lin.tolist(),"near_separable_control":neg_exact},"runtime_s":time.time()-started}
    for target in ("C","rho"):
      obs=held[f"delta_{target}_observed"].values
      metrics[f"heldout_{target}_first"]=prediction_metrics(obs,held[f"delta_{target}_first"].values)
      metrics[f"heldout_{target}_second"]=prediction_metrics(obs,held[f"delta_{target}_second"].values)
      k=max(1,int(.1*len(held))); truth=set(np.argsort(abs(obs))[-k:])
      for order in ("first","second"):
        chosen=set(np.argsort(abs(held[f"delta_{target}_{order}"].values))[-k:]); overlap=len(truth&chosen)/k
        metrics[f"heldout_{target}_{order}"]["top10pct_overlap"] = overlap
        metrics[f"heldout_{target}_{order}"]["top10pct_precision"] = overlap
        metrics[f"heldout_{target}_{order}"]["top10pct_recall"] = overlap
    # Separate input-Taylor and structural-Taylor error as requested.
    lin_delta=held.C1_perturbed.values-base_lin[0]
    metrics["input_expansion_baseline_abs_error"]=abs(base_exact["C_exact"]-base_lin[0])
    metrics["structural_expansion_first_vs_C1"]=prediction_metrics(lin_delta,held.delta_C_first.values)
    metrics["structural_expansion_second_vs_C1"]=prediction_metrics(lin_delta,held.delta_C_second.values)
    for kind in ("single_delete","single_double","area_incoming","area_outgoing"):
        g=pert[(pert.kind==kind)&pert.stable]
        metrics[kind+"_second"]=prediction_metrics(g.delta_C_observed.values,g.delta_C_second.values)
    def predicted_p(r,n):
        rr=np.clip(abs(r),0,.999999); return 2*stats.t.sf(rr*np.sqrt((n-2)/(1-rr*rr)),n-2)
    metrics["heldout_significant_counts"]={"direct_p_lt_0.05":int((held.p_value<.05).sum()),
        "first_predicted_p_lt_0.05":int(np.sum(predicted_p(held.rho_first.values,held.n_valid.values)<.05)),
        "second_predicted_p_lt_0.05":int(np.sum(predicted_p(held.rho_second.values,held.n_valid.values)<.05))}
    # Bootstrap paired improvement.
    b=int(ecfg["bootstrap"]); imp=[]
    for _ in range(b):
        g=held.iloc[rng.integers(0,len(held),len(held))]; o=g.delta_C_observed.values
        e1=np.sqrt(np.mean((g.delta_C_first-o)**2)); e2=np.sqrt(np.mean((g.delta_C_second-o)**2)); imp.append(e1-e2)
    np.savez(out/"bootstrap_results.npz",rmse_first_minus_second=np.array(imp))
    metrics["bootstrap_rmse_improvement_C_95CI"]=np.percentile(imp,[2.5,97.5]).tolist()
    # Pair interactions among the 15 largest first-order anatomical effects.
    top=coeffs.reindex(coeffs.kernel_C_per_multiplier.abs().sort_values(ascending=False).index).head(15).reset_index(drop=True)
    pair=np.zeros((len(top),len(top))); hp=.03
    djs=[edge_to_delta_j(model,int(r.target_idx),int(r.source_idx),float(r.weight)) for r in top.itertuples()]
    for i in range(len(top)):
      for j in range(i+1,len(top)):
        pp=_linear_stats(model.J+hp*(djs[i]+djs[j]),sigma,freqs,inp,outidx)[0]
        pm=_linear_stats(model.J+hp*(djs[i]-djs[j]),sigma,freqs,inp,outidx)[0]
        mp=_linear_stats(model.J+hp*(-djs[i]+djs[j]),sigma,freqs,inp,outidx)[0]
        mm=_linear_stats(model.J-hp*(djs[i]+djs[j]),sigma,freqs,inp,outidx)[0]
        pair[i,j]=pair[j,i]=(pp-pm-mp+mm)/(4*hp*hp)
    labels=(top.source+"->"+top.target).values; np.savez(out/"second_order_edge_pairs.npz",interaction=pair,labels=labels)
    graph_base=_graph_baselines(model,pert,edges,freqs,inp,outidx); graph_base.to_csv(out/"graph_baselines.csv",index=False)
    metrics["simple_graph_baselines"]={r.predictor:{"pearson_r":r.pearson_r,"spearman_r":r.spearman_r,"R2":r.R2} for r in graph_base.itertuples()}
    save_json(out/"metrics.json",metrics); save_json(out/"manifest.json",{"experiment":"structural_kernel","areas":AREAS9,"edge_convention":"FLN[target,source]; source -> target","n_edges":len(edges),"random_mask_sampling":"15 nonzero directed edges uniformly without replacement","pilot_fraction":.2,"input":"V1 E","output":"PG E"})
    np.savez(out/"representative_psd.npz",freq_hz=freqs,baseline_transfer=base_transfer,baseline_features=base_vals,z=zs)
    _figures(out,model,coeffs,pert,base_exact,base_lin,pair,labels,zs,freqs,inp,outidx)
    return out


def _graph_baselines(model,pert,edges,freqs,inp,outidx):
    g=nx.DiGraph(); g.add_nodes_from(range(len(model.areas)))
    for t,s in edges: g.add_edge(s,t,weight=float(model.fln[t,s]),distance=float(1/(model.fln[t,s]+1e-12)))
    eb=nx.edge_betweenness_centrality(g,weight="distance"); single=pert[pert.kind=="single_delete"].reset_index(drop=True)
    rows=[]; eye=np.eye(len(model.J)); transfer=[]
    for t,s in edges:
        val=[]; dj=edge_to_delta_j(model,t,s,model.fln[t,s])
        for f in np.geomspace(5,20,12):
            r=np.linalg.inv(1j*2*np.pi*f/1000*eye-model.J)
            val.append(abs((r@dj@r)[outidx,inp]))
        transfer.append(np.mean(val))
    predictors={"edge_weight":[model.fln[t,s] for t,s in edges],"source_out_degree":[g.out_degree(s) for t,s in edges],
        "target_in_degree":[g.in_degree(t) for t,s in edges],"edge_betweenness":[eb[(s,t)] for t,s in edges],
        "reverse_graph_distance":[nx.shortest_path_length(g,t,s,weight="distance") if nx.has_path(g,t,s) else np.nan for t,s in edges],
        "hierarchy_difference":[model.params["hier_vals"][t]-model.params["hier_vals"][s] for t,s in edges],"baseline_transfer_magnitude":transfer,
        "structural_kernel":single.delta_C_first.values}
    obs=single.delta_C_observed.values
    for name,x in predictors.items():
        x=np.asarray(x,float); ok=np.isfinite(x)&np.isfinite(obs)
        if np.std(x[ok])==0: continue
        reg=stats.linregress(x[ok],obs[ok]); pred=reg.intercept+reg.slope*x[ok]
        rows.append({"predictor":name,"pearson_r":stats.pearsonr(x[ok],obs[ok]).statistic,"spearman_r":stats.spearmanr(x[ok],obs[ok]).statistic,"R2":1-np.sum((pred-obs[ok])**2)/np.sum((obs[ok]-obs[ok].mean())**2)})
    return pd.DataFrame(rows)


def _figures(out,model,kernel,pert,base_exact,base_lin,pair,pair_labels,zs,freqs,inp,outidx):
    plt.style.use("seaborn-v0_8-whitegrid")
    mat=np.full((9,9),np.nan)
    for r in kernel.itertuples(): mat[r.target_idx,r.source_idx]=r.kernel_C_per_multiplier
    fig,ax=plt.subplots(figsize=(7,6)); vmax=np.nanmax(abs(mat)); im=ax.imshow(mat,cmap="coolwarm",vmin=-vmax,vmax=vmax); ax.set_xticks(range(9),model.areas,rotation=45,ha="right"); ax.set_yticks(range(9),model.areas); ax.set(xlabel="source",ylabel="target",title="Anatomical first-order covariance kernel"); fig.colorbar(im,ax=ax)
    save_figure(fig,out/"figures"/"anatomical_kernel_heatmap"); plt.close(fig)
    held=pert[(pert.kind=="random15_delete")&(pert.split=="heldout")&pert.stable]
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axs,["delta_C_first","delta_C_second"],["First order","Second order"]):
        ax.scatter(held.delta_C_observed,held[key],s=18,alpha=.65); lo=min(held.delta_C_observed.min(),held[key].min()); hi=max(held.delta_C_observed.max(),held[key].max()); ax.plot([lo,hi],[lo,hi],"k--"); ax.set(xlabel="Observed Delta C",ylabel="Predicted Delta C",title=title)
    save_figure(fig,out/"figures"/"random15_heldout_predictions"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(held.resolvent_delta_norm,abs(held.delta_C_first-held.delta_C_observed),s=16,label="first"); ax.scatter(held.resolvent_delta_norm,abs(held.delta_C_second-held.delta_C_observed),s=16,label="second"); ax.set(xlabel="max ||R Delta J||2",ylabel="absolute error"); ax.legend(frameon=False)
    save_figure(fig,out/"figures"/"error_vs_resolvent_perturbation"); plt.close(fig)
    single=pert[pert.kind.isin(["single_delete","single_double"])]
    fig,ax=plt.subplots(figsize=(7,4));
    for name,g in single.groupby("kind"): ax.scatter(g.delta_C_observed,g.delta_C_second,s=18,alpha=.65,label=name)
    ax.set(xlabel="Observed Delta C",ylabel="Second-order Delta C",title="Deletion-doubling asymmetry"); ax.legend(frameon=False)
    save_figure(fig,out/"figures"/"single_edge_delete_double"); plt.close(fig)
    area=pert[pert.kind.str.startswith("area_")]
    fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(len(area)); ax.bar(x-.18,area.delta_C_observed,.36,label="direct"); ax.bar(x+.18,area.delta_C_second,.36,label="second"); ax.set_xticks(x,[v.split('_',1)[1] for v in area.perturbation_id],rotation=60,ha="right"); ax.set_ylabel("Delta C"); ax.legend(frameon=False)
    save_figure(fig,out/"figures"/"incoming_outgoing_area"); plt.close(fig)
    top=kernel.reindex(kernel.kernel_C_per_multiplier.abs().sort_values(ascending=False).index).head(15)
    fig,ax=plt.subplots(figsize=(8,4.5)); labels=top.source+"->"+top.target; colors=np.where(top.kernel_C_per_multiplier>0,"#c44e52","#4c72b0"); ax.barh(np.arange(len(top)),top.kernel_C_per_multiplier,color=colors); ax.set_yticks(np.arange(len(top)),labels); ax.invert_yaxis(); ax.set_xlabel("kernel effect per edge multiplier")
    save_figure(fig,out/"figures"/"top_anatomical_edges"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(held.first_order_cancellation,abs(held.delta_C_second-held.delta_C_observed)-abs(held.delta_C_first-held.delta_C_observed),s=20,alpha=.7); ax.axhline(0,color="k",ls="--"); ax.set(xlabel="first-order cancellation ratio",ylabel="second error - first error")
    save_figure(fig,out/"figures"/"cancellation_second_order"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,7)); vmax=np.max(abs(pair)); im=ax.imshow(pair,cmap="coolwarm",vmin=-vmax,vmax=vmax); ax.set_xticks(range(len(pair_labels)),pair_labels,rotation=60,ha="right",fontsize=7); ax.set_yticks(range(len(pair_labels)),pair_labels,fontsize=7); ax.set_title("Second-order edge-pair interaction"); fig.colorbar(im,ax=ax)
    save_figure(fig,out/"figures"/"second_order_pair_heatmap"); plt.close(fig)
    worst=held.iloc[np.argmax(abs(held.delta_C_second-held.delta_C_observed))]; dw=np.zeros_like(model.fln)
    for source,target,weight in json.loads(worst.edges_json): dw[model.areas.index(target),model.areas.index(source)]=-weight
    jp=apply_fln_delta(model,dw); tb=transfer_power(model.J,inp,outidx,freqs); tp=transfer_power(jp,inp,outidx,freqs)
    _,pb=network_features(model.J,Z0,inp,outidx,freqs,tb); _,pp=network_features(jp,Z0,inp,outidx,freqs,tp)
    fig,ax=plt.subplots(figsize=(7,4)); ax.loglog(freqs,pb,label="baseline"); ax.loglog(freqs,pp,label="largest held-out error"); ax.axvspan(5,20,color="gold",alpha=.15); ax.axvspan(100,1000,color="gray",alpha=.12); ax.set(xlabel="Frequency (Hz)",ylabel="PSD"); ax.legend(frameon=False)
    save_figure(fig,out/"figures"/"selected_failure_case_psd"); plt.close(fig)
