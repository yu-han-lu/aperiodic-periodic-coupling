from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    central_moments,
    correlation_summary,
    finite_derivatives,
    frequency_grid,
    load_model,
    network_features,
    prediction_metrics,
    save_figure,
    save_json,
    theory_covariances,
    transfer_power,
)
from .io import prepare_output


Z0 = np.array([10.0, 6.0, -0.5, 1.0])
ZSCALE = np.array([3.0, 4.0, 0.8, 0.7])
ZLO = np.array([5.1, 1.0, -1.8, 0.15])
ZHI = np.array([19.5, 15.0, 1.0, 3.0])
ZNAMES = ["bump_center_hz", "bump_amplitude", "beta", "aperiodic_amplitude"]


def _draw_design(name: str, n: int, eps: float, rng: np.random.Generator, rho: float = 0.0) -> np.ndarray:
    q = len(Z0)
    if name == "independent_gaussian" or name == "finite_sample":
        x = rng.normal(size=(n, q))
    elif name == "correlated_gaussian":
        cov = np.eye(q); cov[0, 2] = cov[2, 0] = rho
        x = rng.multivariate_normal(np.zeros(q), cov, size=n)
    elif name == "rank_one":
        u = rng.normal(size=n); x = np.c_[u, 0.6*u, -0.8*u, 0.4*u]
    elif name == "uniform":
        x = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n, q))
    elif name == "fixed_total":
        x = rng.normal(size=(n, q)); x[:, 3] = -x[:, 1]
    elif name == "skewed":
        x = rng.lognormal(0, 0.55, size=(n, q)); x = (x-x.mean(0))/x.std(0, ddof=1)
    elif name == "one_factor_mixture":
        x = np.zeros((n, q)); factors = np.arange(n) % q
        x[np.arange(n), factors] = rng.normal(size=n)
    else:
        raise ValueError(name)
    return x * eps


def _feature_fun(transfer, freqs, peak_band, slope_band):
    def fun(u):
        z = Z0 + ZSCALE*u
        feat, _ = network_features(np.empty((0,0)), z, 0, 0, freqs, transfer, peak_band, slope_band)
        return np.array([feat["peak_omega"], feat["exponent"]])
    return fun


def _theory_rho(fun, u: np.ndarray, method: str, rng: np.random.Generator) -> float:
    # Matching-order variances are obtained from the same Taylor polynomial,
    # avoiding an empirical denominator in a nominally theoretical correlation.
    grad, hess, third = _theory_rho.derivatives
    m = 30000
    ids = rng.integers(0, len(u), m)
    x = u[ids]
    vals = np.empty((m, 2))
    for k in range(2):
        vals[:, k] = fun(np.zeros(u.shape[1]))[k] + x @ grad[k]
        if method in ("C48", "Cquad", "C64"):
            vals[:, k] += 0.5*np.einsum("ni,ij,nj->n", x, hess[k], x)
        if method == "C64":
            vals[:, k] += np.einsum("ni,nj,nk,ijk->n", x, x, x, third[k])/6
    return float(stats.pearsonr(vals[:,0], vals[:,1]).statistic)


def run(cfg: dict) -> Path:
    started = time.time()
    out = prepare_output(cfg, "exp1_input_sampling")
    rng = np.random.default_rng(int(cfg["seed"]))
    model = load_model(Path(cfg["model_dir"]))
    freqs = frequency_grid()
    input_idx, output_idx = model.areas.index("V1"), model.areas.index("PG")
    transfer = transfer_power(model.J, input_idx, output_idx, freqs)
    peak_band, slope_band = tuple(cfg["peak_band_hz"]), tuple(cfg["slope_band_hz"])
    fun = _feature_fun(transfer, freqs, peak_band, slope_band)
    grad, hess, third = finite_derivatives(fun, np.zeros(4), h=2e-3)
    _theory_rho.derivatives = (grad, hess, third)
    np.savez(out/"input_derivatives.npz", grad=grad, hess=hess, third=third, z0=Z0, zscale=ZSCALE)
    validation=[]
    for vi in range(50 if cfg.get("mode")=="full" else 6):
        x=rng.normal(size=4)*.08
        g1,h1,t1=finite_derivatives(fun,x,h=4e-3); g2,h2,t2=finite_derivatives(fun,x,h=2e-3)
        rel=lambda a,b: float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-15))
        validation.append({"point":vi,"grad_rel_error":rel(g1,g2),"hessian_rel_error":rel(h1,h2),"third_rel_error":rel(t1,t2)})
    pd.DataFrame(validation).to_csv(out/"derivative_validation.csv",index=False)

    designs = ["independent_gaussian", "rank_one", "uniform", "fixed_total", "skewed", "one_factor_mixture", "finite_sample"]
    conditions = []
    for eps in cfg["exp1"]["epsilons"]:
        conditions += [(d, float(eps), 0.0) for d in designs]
        conditions += [("correlated_gaussian", float(eps), float(r)) for r in cfg["exp1"]["rhos"]]
    trial_rows, pred_rows, summaries = [], [], []
    n = int(cfg["exp1"]["n_trials"])
    for ci, (design, eps, rho) in enumerate(conditions):
        crng = np.random.default_rng(int(cfg["seed"]) + 1000 + ci)
        u = _draw_design(design, n, eps, crng, rho)
        raw = Z0 + u*ZSCALE
        z = np.clip(raw, ZLO, ZHI)
        clipped = np.any(z != raw, axis=1)
        vals = np.empty((n, 2)); valid = np.zeros(n, bool); reasons=[]
        for i in range(n):
            feat, _ = network_features(model.J, z[i], input_idx, output_idx, freqs, transfer, peak_band, slope_band)
            vals[i] = feat["peak_omega"], feat["exponent"]
            valid[i] = feat["peak_valid"]
            reasons.append(feat["feature_failure_reason"])
            trial_rows.append({"experiment":"input_sampling", "condition":f"{design}_eps{eps:g}_rho{rho:g}", "trial_id":i,
                "seed":int(cfg["seed"])+1000+ci, **dict(zip(ZNAMES,z[i])), "clipped":bool(clipped[i]),
                "peak_hz":vals[i,0]/(2*np.pi), "peak_omega":vals[i,0], "exponent":vals[i,1],
                "peak_valid":bool(valid[i]), "peak_boundary":("boundary" in reasons[-1]), "peak_switch":("multiple" in reasons[-1]),
                "stability_valid":True, "feature_failure_reason":reasons[-1]})
        use = valid & ~clipped
        # The actual accepted sample moments are used, including finite-sample covariance.
        uu = (z[use]-Z0)/ZSCALE
        sigma, m3 = central_moments(uu)
        theory = theory_covariances(grad, hess, third, sigma, m3)
        empirical = correlation_summary(vals[use,0], vals[use,1], int(cfg["exp1"]["bootstrap"]), crng)
        cond = f"{design}_eps{eps:g}_rho{rho:g}"
        summaries.append({"condition":cond,"design":design,"epsilon":eps,"rho":rho,**empirical,
                          "n_excluded":int(n-use.sum()),"clip_rate":float(clipped.mean())})
        for method in ("C49","C48","Cquad","C64"):
            pred_rows.append({"condition":cond,"design":design,"epsilon":eps,"rho":rho,"method":method,
                "observed_covariance":empirical["covariance"],"predicted_covariance":theory[method],
                "observed_r":empirical["pearson_r"],"predicted_r":_theory_rho(fun,uu,method,crng),
                **{k:v for k,v in theory.items() if k not in ("C49","C48","Cquad","C64")}})

    trials = pd.DataFrame(trial_rows); preds=pd.DataFrame(pred_rows); summary=pd.DataFrame(summaries)
    trials.to_csv(out/"trials.csv.gz",index=False,compression="gzip")
    preds.to_csv(out/"theory_predictions.csv",index=False)
    summary.to_csv(out/"condition_summary.csv",index=False)
    pd.DataFrame({"seed":[int(cfg["seed"])+1000+i for i in range(len(conditions))],"condition":[f"{d}_eps{e:g}_rho{r:g}" for d,e,r in conditions]}).to_csv(out/"seeds.csv",index=False)

    metrics={m:prediction_metrics(g.observed_covariance.values,g.predicted_covariance.values) for m,g in preds.groupby("method")}
    vd=pd.DataFrame(validation); metrics["derivative_validation"]={c:float(vd[c].median()) for c in vd.columns if c.endswith("error")}
    metrics["runtime_s"]=time.time()-started
    save_json(out/"metrics.json",metrics)
    save_json(out/"manifest.json",{"experiment":"input_sampling","n_trials_rows":len(trials),"n_conditions":len(summary),"input":"V1 E","output":"PG E","z_names":ZNAMES,"frequency_unit":"Hz; peak_omega rad/s"})
    np.savez(out/"bootstrap_results.npz", condition=summary.condition.values, ci_low=summary.pearson_ci_low,ci_high=summary.pearson_ci_high)
    base_psd = network_features(model.J,Z0,input_idx,output_idx,freqs,transfer,peak_band,slope_band)[1]
    np.savez(out/"representative_psd.npz",freq_hz=freqs,psd=base_psd,z=Z0)
    _figures(out,trials,preds,summary,grad)
    return out


def _figures(out: Path, trials: pd.DataFrame, preds: pd.DataFrame, summary: pd.DataFrame, grad: np.ndarray):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    chosen=summary.sort_values("epsilon").groupby("design").tail(1).head(4).condition
    for ax,c in zip(axs.flat,chosen):
        d=trials[(trials.condition==c)&trials.peak_valid]
        ax.scatter(d.peak_hz,d.exponent,s=6,alpha=.35); ax.set_title(c); ax.set_xlabel("Peak (Hz)"); ax.set_ylabel("A")
    save_figure(fig,out/"figures"/"input_design_scatter"); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4.2))
    marks={"C49":"o","C48":"s","Cquad":"^","C64":"D"}
    for m,g in preds.groupby("method"):
        axs[0].scatter(g.observed_covariance,g.predicted_covariance,s=24,alpha=.7,label=m,marker=marks[m])
        axs[1].scatter(g.observed_r,g.predicted_r,s=24,alpha=.7,label=m,marker=marks[m])
    for ax,label in zip(axs,["Covariance","Correlation"]):
        lo=min(ax.get_xlim()[0],ax.get_ylim()[0]); hi=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([lo,hi],[lo,hi],"k--",lw=1)
        ax.set_xlabel(f"Observed {label}"); ax.set_ylabel(f"Predicted {label}")
    axs[1].legend(frameon=False,ncol=2); save_figure(fig,out/"figures"/"observed_vs_predicted"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    err=preds.assign(abs_error=lambda d:abs(d.predicted_covariance-d.observed_covariance)).groupby(["method","epsilon"]).abs_error.mean().reset_index()
    for m,g in err.groupby("method"): ax.plot(g.epsilon,g.abs_error,"o-",label=m)
    ax.set(xlabel="epsilon",ylabel="Mean absolute covariance error"); ax.legend(frameon=False); save_figure(fig,out/"figures"/"error_vs_epsilon"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5)); x=np.arange(4); ax.bar(x,grad[0],label="grad P"); ax.bar(x,grad[1],bottom=grad[0],label="grad A")
    ax.set_xticks(x,ZNAMES,rotation=20,ha="right"); ax.set_ylabel("standardized sensitivity"); ax.legend(frameon=False); save_figure(fig,out/"figures"/"sensitivity_contributions"); plt.close(fig)
