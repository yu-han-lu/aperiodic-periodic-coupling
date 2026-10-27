from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = lambda f: f

from .core import feature_from_spectrum, load_model, prediction_metrics, save_figure, save_json
from .io import prepare_output


@njit
def _sl_path(mu, omega0, cr, ci, d, dt, n, seed):
    np.random.seed(seed)
    z = np.empty(n, np.complex128); z[0] = 0.05 + 0.0j
    noise = np.sqrt(2*d*dt)
    for k in range(1,n):
        zz=z[k-1]; r2=zz.real*zz.real+zz.imag*zz.imag
        radial=zz+(mu-cr*r2)*zz*dt+noise*(np.random.randn()+1j*np.random.randn())
        z[k]=radial*np.exp(1j*(omega0-ci*r2)*dt)
    return z


@njit
def _ei_path(w, tau, noise, dt, n, seed):
    np.random.seed(seed)
    x=np.empty((n,2)); x[0]=np.array([0.02,-0.01])
    s=np.sqrt(2*noise*dt)
    for k in range(1,n):
        y=x[k-1]; eta=s*np.random.randn(2); f0=(-y+np.tanh(w@y))/tau
        pred=y+dt*f0+eta; f1=(-pred+np.tanh(w@pred))/tau
        x[k]=y+.5*dt*(f0+f1)+eta
    return x


@njit
def _ei_linear_path(j, noise, dt, n, seed):
    np.random.seed(seed); x=np.empty((n,2)); x[0]=np.array([0.02,-0.01]); s=np.sqrt(2*noise*dt)
    for k in range(1,n):
        y=x[k-1]; eta=s*np.random.randn(2); pred=y+dt*(j@y)+eta
        x[k]=y+.5*dt*(j@y+j@pred)+eta
    return x


def radial_density(r: np.ndarray, mu: float, cr: float, d: float) -> np.ndarray:
    logp=np.log(np.maximum(r,1e-300))+mu*r*r/(2*d)-cr*r**4/(4*d)
    p=np.exp(logp-logp.max()); return p/np.trapezoid(p,r)


def radial_moments(mu: float, cr: float, d: float) -> tuple[float,float,float]:
    scale=max(2.0,np.sqrt(max(mu,0)/cr)*4,np.power(d/cr,.25)*8)
    r=np.linspace(1e-8,scale,20000); p=radial_density(r,mu,cr,d)
    m2=np.trapezoid(r**2*p,r); m4=np.trapezoid(r**4*p,r)
    return float(m2),float(m4),float(m4-m2*m2)


def _windows(z: np.ndarray, dt: float, burn: float, window: float, seed: int, condition: str, layer: str):
    start=int(burn/dt); nw=int(window/dt); fs=1/dt; rows=[]
    z=z[start:]
    for wi,lo in enumerate(range(0,len(z)-nw+1,nw)):
        zz=z[lo:lo+nw]; r2=np.abs(zz)**2; phase=np.unwrap(np.angle(zz)); dot=np.diff(phase)/dt
        f,p=signal.welch(zz.real,fs=fs,nperseg=min(nw,4096),noverlap=min(nw,4096)//2)
        keep=(f>=.5)&(f<=min(1200,fs/2*.999))
        # Fixed smoothing suppresses Welch-bin ripple without adapting to a trial.
        logp=ndimage.gaussian_filter1d(np.log(np.maximum(p[keep],1e-300)),2.0)
        feat=feature_from_spectrum(f[keep],np.exp(logp))
        rows.append({"layer":layer,"condition":condition,"seed":seed,"window_id":wi,"mean_R2":r2.mean(),
            "var_R2":r2.var(ddof=1),"mean_dot_phi":dot.mean(),"peak_hz":feat["peak_hz"],"peak_omega":feat["peak_omega"],
            "exponent":feat["exponent"],"peak_valid":feat["peak_valid"],"feature_failure_reason":feat["feature_failure_reason"]})
    return rows


def _predict_by_condition(df: pd.DataFrame, density_vars: dict[str,float]) -> pd.DataFrame:
    rows=[]
    for cond,g in df.groupby("condition"):
        tr=g[g.seed%2==0]; te=g[g.seed%2==1]
        if len(tr)<3 or len(te)<3: continue
        sf=stats.linregress(tr.mean_R2,tr.mean_dot_phi).slope
        sa=stats.linregress(tr.mean_R2,tr.exponent).slope
        c_obs=np.cov(te.mean_dot_phi,te.exponent,ddof=1)[0,1]
        vemp=te.mean_R2.var(ddof=1)
        rows.append({"condition":cond,"layer":g.layer.iloc[0],"n_train":len(tr),"n_test":len(te),
            "c_i_train":-sf,"kappa_R_train":sa,"var_R2_test":vemp,"var_R2_density":density_vars.get(cond,np.nan),
            "C_obs":c_obs,"C_reg":sf*sa*vemp,"C_semi":-(-sf)*sa*vemp,
            "C_full":-(-sf)*sa*density_vars.get(cond,np.nan),
            "peak_phi_r":stats.pearsonr(te.peak_omega,te.mean_dot_phi).statistic})
    return pd.DataFrame(rows)


def _linear_fit_summary(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """OLS summary with a two-sided 95% t interval for the slope."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0:
        return {k: np.nan for k in ("slope", "intercept", "r2", "p_value", "ci_low", "ci_high")} | {"n": len(x)}
    fit = stats.linregress(x, y)
    delta = stats.t.ppf(0.975, len(x) - 2) * fit.stderr
    return {
        "n": int(len(x)),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r2": float(fit.rvalue**2),
        "p_value": float(fit.pvalue),
        "ci_low": float(fit.slope - delta),
        "ci_high": float(fit.slope + delta),
    }


def condition_regression_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition training fits used to estimate c_i and kappa_R."""
    rows = []
    for (layer, condition), group in df.groupby(["layer", "condition"], sort=True):
        train = group[group.seed % 2 == 0]
        phi = _linear_fit_summary(train.mean_R2, train.mean_dot_phi)
        amp = _linear_fit_summary(train.mean_R2, train.exponent)
        rows.append({
            "layer": layer,
            "condition": condition,
            "n_train": phi["n"],
            "phidot_R2_slope": phi["slope"],
            "phidot_R2_R2": phi["r2"],
            "phidot_R2_p": phi["p_value"],
            "c_i_estimate": -phi["slope"],
            "c_i_ci_low": -phi["ci_high"],
            "c_i_ci_high": -phi["ci_low"],
            "A_R2_slope": amp["slope"],
            "A_R2_R2": amp["r2"],
            "A_R2_p": amp["p_value"],
            "kappa_R_estimate": amp["slope"],
            "kappa_R_ci_low": amp["ci_low"],
            "kappa_R_ci_high": amp["ci_high"],
            "ci_method": "window-level OLS t interval (95%)",
        })
    return pd.DataFrame(rows)


def _annotated_regression(ax, x, y, color="#1f6f8b"):
    fit = _linear_fit_summary(pd.Series(x), pd.Series(y))
    if np.isfinite(fit["slope"]):
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, fit["intercept"] + fit["slope"] * xx, color=color, lw=2, label="OLS fit")
        p_text = f"{fit['p_value']:.2e}" if fit["p_value"] < 0.001 else f"{fit['p_value']:.3f}"
        ax.text(
            0.04, 0.96,
            f"b = {fit['slope']:.4g}\np = {p_text}\nR^2 = {fit['r2']:.3f}\nn = {fit['n']}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
    return fit


def _local_ei_setup():
    m=load_model(); p=m.params; i=m.areas.index("PG")
    w=np.array([[p["beta_exc"]*p["wEE"]*p["local_exc_scale"][i],-p["beta_exc"]*p["wEI"]],
                [p["beta_inh"]*p["wIE"]*p["local_inh_scale"][i],-p["beta_inh"]*p["wII"]]])
    tau=np.array([p["tau_exc"],p["tau_inh"]])/1000
    # In the standard local blocks an E-E scan reaches a real saddle before
    # trace zero. PG minimizes the cross-coupling increase required for a true
    # complex Hopf. Lock that balance, then use E-E gain as the bifurcation axis.
    j22=(w[1,1]-1)/tau[1]; j12=w[0,1]/tau[0]; j21=w[1,0]/tau[1]
    balance=1.08*j22*j22/(-j12*j21)
    w[0,1]*=balance
    # trace((W-I)/tau)=0 gives the local E-E gain at Hopf.
    gh=((1-w[1,1])/tau[1]+1/tau[0])/(w[0,0]/tau[0])
    return w,tau,float(gh),float(balance)


def _critical_projection(x: np.ndarray,w: np.ndarray,tau: np.ndarray):
    j=(w-np.eye(2))/tau[:,None]
    vals,left=np.linalg.eig(j.T.conj()); k=np.argmax(vals.real)
    lv=left[:,k]; lv=lv/(np.linalg.norm(lv)+1e-15)
    z=x@lv.conj()
    if np.isrealobj(z) or abs(vals[k].imag)<1e-8:
        z=signal.hilbert(np.asarray(z.real))
    return z,vals[k]


def run(cfg: dict) -> Path:
    started=time.time(); out=prepare_output(cfg,"exp2_nonlinear_hopf"); ecfg=cfg["exp2"]
    dt=float(ecfg["dt_s"]); dur=float(ecfg["duration_s"]); burn=float(ecfg["burn_s"]); win=float(ecfg["window_s"]); n=int(dur/dt)
    mode=cfg.get("mode","full")
    mus=[-.8,-.15,.08] if mode=="smoke" else [-1.0,-.3,-.05,.1,.5]
    ds=[.02,.08] if mode=="smoke" else [.01,.04,.12]
    cis=[-4.,0.,4.]; crs=[1.,2.] if mode=="full" else [1.]
    seeds=range(4 if mode=="smoke" else 8); rows=[]; density_vars={}; density_rows=[]; reps={}; radial_samples={}; rep_psd=None
    for mu in mus:
      for d in ds:
       for ci in cis:
        for cr in crs:
         cond=f"SL_mu{mu:g}_D{d:g}_ci{ci:g}_cr{cr:g}"; m2,m4,v=radial_moments(mu,cr,d); density_vars[cond]=v
         density_rows.append({"condition":cond,"mu":mu,"D":d,"c_i":ci,"c_r":cr,"E_R2":m2,"E_R4":m4,"Var_R2_theory":v})
         for seed in seeds:
            z=_sl_path(mu,2*np.pi*10,cr,ci,d,dt,n,int(cfg["seed"])+seed+int((mu+2)*1000)+int(d*1e4)+int((ci+5)*100))
            rows.extend(_windows(z,dt,burn,win,seed,cond,"Stuart-Landau"))
            if seed==0 and cond not in reps: reps[cond]=z[::max(1,int(.002/dt))][:5000]
            radial_samples.setdefault(cond,[]).append(np.abs(z[int(burn/dt)::max(1,int(.02/dt))]))
            if rep_psd is None:
                ff,pp=signal.welch(z[int(burn/dt):].real,fs=1/dt,nperseg=min(int(win/dt),len(z)-int(burn/dt)))
                rep_psd=(ff,pp,cond)
    radial_samples={k:np.concatenate(v) for k,v in radial_samples.items()}
    sl=pd.DataFrame(rows); dens=pd.DataFrame(density_rows)
    emp=pd.DataFrame([{"condition":k,"E_R2_emp":float(np.mean(v**2)),"Var_R2_emp":float(np.var(v**2,ddof=1))} for k,v in radial_samples.items()])
    dens=dens.merge(emp,on="condition",how="left"); pred_sl=_predict_by_condition(sl,density_vars)

    # Existing local E-I block with a smooth saturation and E-E gain scan.
    w0,tau,gh,balance=_local_ei_setup(); gains=gh+np.array([-.12,-.04,-.01,.02,.08]); ei_rows=[]; eig_rows=[]
    for gain in gains:
      w=w0.copy(); w[0,0]*=gain
      j=(w-np.eye(2))/tau[:,None]; ev=np.linalg.eigvals(j); eig_rows.append({"gain":gain,"distance_to_hopf":gain-gh,"max_real":ev.real.max(),"critical_imag":ev[np.argmax(ev.real)].imag})
      for noise in ([.001,.004] if mode=="smoke" else [.0005,.002,.006]):
       cond=f"EI_gain{gain:.5f}_noise{noise:g}"
       for seed in seeds:
        x=_ei_path(w,tau,noise,dt,n,int(cfg["seed"])+50000+seed+int(gain*1e5)+int(noise*1e6))
        z,crit=_critical_projection(x,w,tau)
        ei_rows.extend(_windows(z,dt,burn,win,seed,cond,"smooth local E-I"))
    ei=pd.DataFrame(ei_rows); pred_ei=_predict_by_condition(ei,{})
    # Linearized local E-I negative control at the same near-critical Jacobian.
    lin_rows=[]; gain=gh-.04; w=w0.copy(); w[0,0]*=gain; j=(w-np.eye(2))/tau[:,None]
    for noise in ([.001] if mode=="smoke" else [.0005,.002,.006]):
      cond=f"EI_linear_gain{gain:.5f}_noise{noise:g}"
      for seed in seeds:
        x=_ei_linear_path(j,noise,dt,n,int(cfg["seed"])+90000+seed+int(noise*1e6)); z,crit=_critical_projection(x,w,tau)
        lin_rows.extend(_windows(z,dt,burn,win,seed,cond,"linearized local E-I"))
    lin=pd.DataFrame(lin_rows); pred_lin=_predict_by_condition(lin,{})
    all_windows=pd.concat([sl,ei,lin],ignore_index=True); preds=pd.concat([pred_sl,pred_ei,pred_lin],ignore_index=True)
    all_windows.to_csv(out/"trials.csv.gz",index=False,compression="gzip"); preds.to_csv(out/"theory_predictions.csv",index=False)
    condition_regression_stats(all_windows).to_csv(out/"condition_regression_stats.csv", index=False)
    summary=all_windows.groupby(["layer","condition"]).agg(n_windows=("window_id","size"),peak_valid_rate=("peak_valid","mean"),mean_R2=("mean_R2","mean"),mean_dot_phi=("mean_dot_phi","mean"),mean_exponent=("exponent","mean")).reset_index()
    summary.to_csv(out/"condition_summary.csv",index=False)
    dens.to_csv(out/"radial_moments.csv",index=False); pd.DataFrame(eig_rows).to_csv(out/"ei_bifurcation.csv",index=False)
    pd.DataFrame({"seed":list(seeds)}).to_csv(out/"seeds.csv",index=False)
    metrics={}
    for layer,g in preds.groupby("layer"):
      metrics[layer]={}
      for key in ("C_semi","C_full","C_reg"):
        if g[key].notna().sum()>2: metrics[layer][key]=prediction_metrics(g.C_obs.values,g[key].values)
    metrics.update({"runtime_s":time.time()-started,"hopf_gain":gh,"PG_I_to_E_balance_multiplier":balance})
    save_json(out/"metrics.json",metrics); save_json(out/"manifest.json",{"experiment":"nonlinear_hopf","dt_s":dt,"duration_s":dur,"burn_s":burn,"window_s":win,"seed_split":"even train, odd held-out","local_area":"PG","hopf_gain":gh,"PG_I_to_E_balance_multiplier":balance})
    np.savez(out/"representative_timeseries.npz",**{k.replace('.','p').replace('-','m'):v for k,v in list(reps.items())[:6]})
    np.savez(out/"radial_samples.npz",**{k.replace('.','p').replace('-','m'):v for k,v in radial_samples.items()})
    if rep_psd is not None: np.savez(out/"representative_psd.npz",freq_hz=rep_psd[0],psd=rep_psd[1],condition=rep_psd[2])
    brng=np.random.default_rng(int(cfg["seed"])+70000); boot=[]
    q=pred_sl.dropna(subset=["C_obs","C_semi"])
    for _ in range(int(ecfg["bootstrap"])):
        bq=q.iloc[brng.integers(0,len(q),len(q))]
        boot.append(prediction_metrics(bq.C_obs.values,bq.C_semi.values)["R2"])
    np.savez(out/"bootstrap_results.npz",stuart_landau_Csemi_R2=np.asarray(boot))
    _figures(out,sl,ei,preds,dens,pd.DataFrame(eig_rows),gh,radial_samples)
    return out


def _figures(out,sl,ei,preds,dens,eigs,gh,radial_samples):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].plot(eigs.gain,eigs.max_real,"o-"); axs[0].axhline(0,color="k",ls="--"); axs[0].axvline(gh,color="crimson",ls=":"); axs[0].set(xlabel="local E-E gain multiplier",ylabel=r"critical $\mathrm{Re}(\lambda)$",title="Smooth local E-I Hopf scan")
    axs[1].plot(eigs.gain,abs(eigs.critical_imag)/(2*np.pi),"o-"); axs[1].set(xlabel="gain multiplier",ylabel="linear frequency (Hz)")
    save_figure(fig,out/"figures"/"ei_bifurcation"); plt.close(fig)
    representative="SL_mu-1_D0.01_ci4_cr1"
    g=sl[(sl.condition==representative) & (sl.seed%2==0)].copy()
    fig,axs=plt.subplots(1,2,figsize=(10,4)); axs[0].scatter(g.mean_R2,g.mean_dot_phi,s=18,alpha=.62); axs[1].scatter(g.mean_R2,g.exponent,s=18,alpha=.62)
    _annotated_regression(axs[0],g.mean_R2,g.mean_dot_phi)
    _annotated_regression(axs[1],g.mean_R2,g.exponent)
    axs[0].set(xlabel="window mean R^2",ylabel="mean dot Phi (rad/s)"); axs[1].set(xlabel="window mean R^2",ylabel="A (100-1000 Hz)")
    fig.suptitle(f"Training windows: {representative}",fontsize=11)
    fig.tight_layout(rect=[0,0,1,.94])
    save_figure(fig,out/"figures"/"common_amplitude_driver"); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,layer in zip(axs,["Stuart-Landau","smooth local E-I"]):
      q=preds[preds.layer==layer].dropna(subset=["C_semi"]); ax.scatter(q.C_semi,q.C_obs,c=q.c_i_train,cmap="coolwarm",s=28)
      if len(q): lo=min(q.C_obs.min(),q.C_semi.min()); hi=max(q.C_obs.max(),q.C_semi.max()); ax.plot([lo,hi],[lo,hi],"k--",lw=1)
      _annotated_regression(ax,q.C_semi,q.C_obs)
      ax.set(xlabel="Eq. 134 semi-theory",ylabel="held-out observed covariance",title=layer)
    save_figure(fig,out/"figures"/"eq134_observed_predicted"); plt.close(fig)
    row=dens.iloc[len(dens)//2]; r=np.linspace(1e-5,max(2,np.sqrt(max(row.E_R2,0))*4),800); p=radial_density(r,row.mu,row.c_r,row.D)
    cond=row.condition; vals=radial_samples[cond]
    fig,ax=plt.subplots(figsize=(6,4)); ax.hist(vals,bins=20,density=True,alpha=.45,label="instantaneous empirical"); ax.plot(r,p,lw=2,label="Eq. 136"); ax.set(xlabel="R",ylabel="density",title=cond); ax.legend(frameon=False)
    save_figure(fig,out/"figures"/"radial_density"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); zero=preds[(preds.layer=="Stuart-Landau") & (abs(preds.c_i_train)<1.5)]; non=preds[(preds.layer=="Stuart-Landau") & (abs(preds.c_i_train)>=1.5)]
    ax.boxplot([zero.C_obs.dropna(),non.C_obs.dropna()],tick_labels=["c_i = 0 control","c_i != 0"]); ax.set_ylabel("held-out Cov(dot Phi, A)")
    save_figure(fig,out/"figures"/"ci_zero_control"); plt.close(fig)
