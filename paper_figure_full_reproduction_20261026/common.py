from __future__ import annotations

"""Shared plotting and small deterministic simulation helpers.

The default entry points in each panel directory use only data shipped in the
archive.  No path outside the archive is consulted.  Expensive simulations
were frozen into source-data tables; representative two-dimensional traces are
recomputed here with the stated Heun integrator.
"""

import json
import math
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy import signal, stats

BLUE = "#247BA0"
ORANGE = "#F28E2B"
TEAL = "#2A9D8F"
RED = "#D1495B"
PURPLE = "#6A4C93"
BLACK = "#222222"
GREY = "#7A7A7A"


def style(font_size: float = 8.0) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.labelsize": font_size + 0.6,
            "axes.linewidth": 0.8,
            "xtick.labelsize": font_size - 0.2,
            "ytick.labelsize": font_size - 0.2,
            "legend.fontsize": font_size - 0.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def require_matplotlib_panel_alignment(fig: plt.Figure) -> None:
    """Portable render-time alignment gate used by every output."""
    fig.align_labels()


def finish(fig: plt.Figure, out_dir: Path, name: str) -> None:
    require_matplotlib_panel_alignment(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def scatter_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, ols: bool = True) -> stats.LinregressResult:
    fit = stats.linregress(x, y)
    ax.scatter(x, y, s=11, color=BLUE, alpha=0.48, edgecolor="none")
    if ols:
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
        ax.plot(grid, fit.intercept + fit.slope * grid, color=BLACK, lw=1.5)
    return fit


def overlay_hist_box(theory: np.ndarray, simulation: np.ndarray, xlim: tuple[float, float]) -> plt.Figure:
    edges = np.linspace(min(theory.min(), simulation.min()), max(theory.max(), simulation.max()), 21)
    fig, (box, hist) = plt.subplots(
        2, 1, figsize=(3.3, 3.25), sharex=True,
        gridspec_kw={"height_ratios": [1.05, 4]}, constrained_layout=True,
    )
    for values, pos, color, label in (
        (theory, 2, GREY, "Theory"), (simulation, 1, ORANGE, "Simulation")
    ):
        box.boxplot(
            values, vert=False, positions=[pos], widths=0.48, patch_artist=True,
            boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.48},
            medianprops={"color": BLACK, "lw": 1.3},
            whiskerprops={"color": color}, capprops={"color": color},
            flierprops={"marker": "o", "markersize": 2, "markerfacecolor": color,
                        "markeredgecolor": color, "alpha": 0.45},
        )
    box.set(yticks=[2, 1], yticklabels=["Theory", "Simulation"], xlim=xlim)
    box.tick_params(axis="x", bottom=False, labelbottom=False)
    box.tick_params(axis="y", length=0)
    for spine in box.spines.values():
        spine.set_visible(False)
    hist.hist(theory, bins=edges, color=GREY, edgecolor=GREY, alpha=0.46, label="Theory")
    hist.hist(simulation, bins=edges, color=ORANGE, edgecolor=ORANGE, alpha=0.46, label="Simulation")
    hist.set(xlim=xlim, xlabel="Correlation", ylabel="Frequency")
    hist.spines[["top", "right"]].set_visible(False)
    return fig


def plot_main(panel: str, panel_dir: Path) -> None:
    style(8.5)
    data = panel_dir / "data"
    out = panel_dir / "output"
    if panel == "fig2a":
        model = pickle.load(open(data / "subgraph_data_marmoset.pkl", "rb"))
        matrix = np.asarray(model["fln_mat"], float)
        areas = model["areas"]
        fig, ax = plt.subplots(figsize=(4.0, 3.3), constrained_layout=True)
        image = ax.imshow(np.log10(np.maximum(matrix, 1e-8)), cmap="Blues", vmin=-8, vmax=0)
        ax.set(xlabel="Source area", ylabel="Target area", xticks=[], yticks=[])
        fig.colorbar(image, ax=ax, label=r"$\log_{10}$ FLN", fraction=0.05, pad=0.03)
        for name, color in (("V1", ORANGE), ("PG", RED), ("A6Va", TEAL)):
            if name in areas:
                i = areas.index(name)
                ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=color, lw=1.5))
        finish(fig, out, "Fig2a_network_structure")
    elif panel == "fig2b":
        frame = pd.read_csv(data / "experiment1_results_heun.csv")
        x = frame["theory_r_heun"].to_numpy(float)
        y = frame["simulation_r_heun"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(3.0, 2.5), constrained_layout=True)
        fit = scatter_fit(ax, x, y)
        ax.set(xlabel="Theory correlation", ylabel="Simulation correlation")
        ax.text(.05, .95, f"slope={fit.slope:.3f}", transform=ax.transAxes, va="top")
        finish(fig, out, "Fig2b")
    elif panel in {"fig2c", "fig2g"}:
        filename = "Fig2c_histogram_source_data.csv" if panel == "fig2c" else "Fig2g_histogram_source_data.csv"
        frame = pd.read_csv(data / filename)
        limits = (-.2, .2) if panel == "fig2c" else (-.6, .6)
        fig = overlay_hist_box(
            frame["theory_correlation"].to_numpy(float),
            frame["simulation_correlation"].to_numpy(float), limits,
        )
        finish(fig, out, panel)
    elif panel == "fig2d":
        frame = pd.read_csv(data / "two_stage_eigen_cross_spectrum.csv")
        # The I-to-E multiplier is the registered local PG transition sweep.
        part = frame.loc[frame["stage"].astype(str).str.contains("I_to_E")].copy()
        x = part["scan_value"].to_numpy(float)
        real_col = "critical_real_per_s"
        imag_col = "critical_imag_per_s"
        fig, ax = plt.subplots(figsize=(3.2, 2.5), constrained_layout=True)
        ax.plot(x, part[real_col], color=BLUE, label=r"Re($\lambda$)")
        ax.plot(x, np.abs(part[imag_col]), color=RED, label=r"|Im($\lambda$)|")
        ax.axhline(0, color="0.5", ls="--", lw=.8)
        ax.set(xlabel="PG I→E multiplier", ylabel="Eigenvalue (s⁻¹)")
        ax.legend(frameon=False)
        finish(fig, out, "Fig2d")
    elif panel == "fig2e1_e2":
        z = np.load(data / "cross_spectrum_examples.npz")
        f = z["condition_6_freq_hz"]
        output_psd = z["condition_6_E_psd"]
        input_psd = (np.maximum(f, .5) / 10) ** (-.5) + .1**2
        input_psd /= input_psd.max()
        output_psd = output_psd / output_psd.max()
        for name, values, color in (("Fig2e1_input", input_psd, GREY), ("Fig2e2_output", output_psd, BLUE)):
            fig, ax = plt.subplots(figsize=(2.7, 2.2), constrained_layout=True)
            ax.loglog(f, values, color=color, lw=1.4)
            ax.set(xlabel="Frequency (Hz)", ylabel="Normalized PSD", xlim=(.7, 200))
            finish(fig, out, name)
    elif panel == "fig2f":
        frame = pd.read_csv(data / "Fig2f_chi_source_values.csv")
        x = frame["theory_r_P_chi"].to_numpy(float)
        y = frame["simulation_r_P_chi"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(3.0, 2.4), constrained_layout=True)
        fit = scatter_fit(ax, x, y)
        ax.set(xlabel="", ylabel="")
        ax.text(.05, .95, f"slope={fit.slope:.3f}", transform=ax.transAxes, va="top")
        finish(fig, out, "Fig2f_chi")
    elif panel == "fig2h1_h2":
        model = pickle.load(open(data / "subgraph_data_marmoset.pkl", "rb"))
        matrix = np.asarray(model["fln_mat"], float)
        selected = json.loads((data / "selected_single_node.json").read_text())
        idx = model["areas"].index(selected["node"])
        deleted = matrix.copy(); deleted[idx, :] = 0; deleted[:, idx] = 0
        for name, values in (("Fig2h1_intact", matrix), ("Fig2h2_deleted", deleted)):
            fig, ax = plt.subplots(figsize=(2.45, 2.25), constrained_layout=True)
            ax.imshow(np.log10(np.maximum(values, 1e-8)), cmap="Blues", vmin=-8, vmax=0)
            ax.set(xticks=[], yticks=[])
            finish(fig, out, name)
    elif panel == "fig2i":
        frame = pd.read_csv(data / "plot_source_data_abs_log.csv")
        x = frame["abs_theory_delta_r_mean"].to_numpy(float)
        y = frame["abs_simulation_delta_r_mean"].to_numpy(float)
        use = (x > 0) & (y > 0)
        fit = stats.linregress(np.log10(x[use]), np.log10(y[use]))
        colors = {1: BLUE, 2: RED, 3: ORANGE, 4: "#41AB5D", 5: "#9E9AC8", 12: GREY}
        markers = {1: "o", 2: "P", 3: "^", 4: "s", 5: "D", 12: "|"}
        fig, ax = plt.subplots(figsize=(3.1, 3.0), constrained_layout=True)
        for n in sorted(frame.n_deleted.unique()):
            take = use & (frame.n_deleted.to_numpy(int) == n)
            ax.scatter(x[take], y[take], s=18, c=colors[int(n)], marker=markers[int(n)],
                       edgecolor="white", linewidth=.25, label=f"{n} node" if n == 1 else f"{n} nodes")
        low = min(x[use].min(), y[use].min()) * .7; high = max(x[use].max(), y[use].max()) * 1.4
        grid = np.geomspace(low, high, 300)
        ax.plot(grid, 10**fit.intercept * grid**fit.slope, color=BLACK, lw=1.35)
        ax.plot(grid, grid, color=GREY, ls="--", lw=.8)
        ax.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
               xlabel="|Theory mean Δr|", ylabel="|Simulation mean Δr|")
        ax.legend(frameon=False, ncol=2, fontsize=6)
        ax.text(.05, .95, f"r={fit.rvalue:.2f}\nslope={fit.slope:.2f}", transform=ax.transAxes, va="top")
        finish(fig, out, "Fig2i")
    elif panel == "fig2j":
        units = pd.read_csv(data / "plot_units_absolute_delta_r.csv.gz")
        sig = pd.read_csv(data / "significance.csv")
        colors = {("Theory", True): BLUE, ("Theory", False): "#78B7D0",
                  ("Simulation", True): RED, ("Simulation", False): "#E69AAB"}
        offsets = {("Theory", True): -.30, ("Theory", False): -.10,
                   ("Simulation", True): .10, ("Simulation", False): .30}
        fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
        rng = np.random.default_rng(20260931)
        for n in range(1, 6):
            for source in ("Theory", "Simulation"):
                for direct in (True, False):
                    v = units.loc[(units.n_deleted == n) & (units.source == source) & (units.direct == direct), "value"].to_numpy(float)
                    pos = n - 1 + offsets[(source, direct)]
                    b = ax.boxplot(v, positions=[pos], widths=.16, patch_artist=True, showfliers=False)
                    b["boxes"][0].set(facecolor=colors[(source, direct)], edgecolor=colors[(source, direct)], alpha=.38)
                    b["medians"][0].set(color=BLACK, lw=1.2)
                    ax.scatter(pos + rng.uniform(-.05, .05, len(v)), v, s=2.2, c=colors[(source, direct)], alpha=.22, edgecolor="none")
                q = sig.loc[(sig.n_deleted == n) & (sig.source == source), "sign_flip_fdr_q"].iloc[0]
                stars = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"
                vmax = units.loc[(units.n_deleted == n) & (units.source == source), "value"].max()
                ax.text(n - 1 + (offsets[(source, True)] + offsets[(source, False)]) / 2, vmax * 1.5, stars, ha="center")
        positive = units.loc[units.value > 0, "value"].to_numpy(float)
        ax.set(yscale="log", ylim=(positive.min() * .65, positive.max() * 35),
               xticks=range(5), xticklabels=["1", "2", "3", "4", "5"], xlabel="", ylabel="")
        handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[k], label=f"{k[0]} {'direct' if k[1] else 'non-direct'}") for k in colors]
        ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.15))
        finish(fig, out, "Fig2j")
    elif panel == "figS5":
        frame = pd.read_csv(data / "heldout_mean_correlation_exact.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(4.0, 3.5), constrained_layout=True)
        im = ax.imshow(frame.to_numpy(float), cmap="viridis", vmin=.90, vmax=1.0)
        ax.set(xticks=range(len(frame.columns)), xticklabels=frame.columns, yticks=range(len(frame.index)), yticklabels=frame.index,
               xlabel="Output area", ylabel="Input area")
        ax.tick_params(axis="x", rotation=45)
        fig.colorbar(im, ax=ax, label="Theory–simulation Pearson r", fraction=.045, pad=.03)
        finish(fig, out, "FigS5")
    else:
        raise ValueError(panel)


def _cov_ellipse(ax: plt.Axes, mean: np.ndarray, cov: np.ndarray, color: str) -> None:
    values, vectors = np.linalg.eigh(cov)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    ax.add_patch(Ellipse(mean, 4 * np.sqrt(values[0]), 4 * np.sqrt(values[1]), angle=angle,
                         fill=False, ec=color, ls="--", lw=1.2))


def plot_s2(panel: str, panel_dir: Path) -> None:
    style(7.0)
    root = panel_dir.parents[0]
    state = np.load(root / "representative_state.npz") if panel in {"a", "b1_b4", "c"} else None
    data = panel_dir / "data"; out = panel_dir / "output"
    if panel == "a":
        f = state["fooof_component_frequency"]
        fig, ax = plt.subplots(figsize=(3.2, 2.45), constrained_layout=True)
        ax.plot(f, state["fooof_source_psd"], color=BLACK, lw=1.1, label="Source log PSD")
        ax.plot(f, state["fooof_aperiodic_fit"], color=BLUE, lw=1.2, label="FOOOF aperiodic")
        ax.plot(f, state["fooof_periodic_fit"], color=RED, lw=1.0, label="FOOOF periodic")
        ax.set(xlabel="Frequency (Hz)", ylabel="Log power")
        ax.legend(frameon=False)
        finish(fig, out, "FigS2a")
    elif panel == "b1_b4":
        # The estimator influence functions use the registered 0.2-Hz grid.
        freq = np.arange(.2, 200.0 + .2, .2)
        foo = state["influence"]; spl = state["split_influence"]
        xo = state["train_o"] - state["train_o"].mean(axis=0)
        xr0 = state["train_o"] - state["train_f"]
        xr = xr0 - xr0.mean(axis=0)
        xf = state["train_f"] - state["train_f"].mean(axis=0)
        foo_transfer = xo.T @ (xo @ foo[1]) / (len(xo) - 1)
        split_transfer = xr.T @ (xf @ spl[1]) / (len(xr) - 1)
        foo_contrib = foo[0] * foo_transfer
        split_contrib = spl[0] * split_transfer
        visible = (freq >= 60) & (freq <= 150)
        for weights, title, name in (
            (foo, "FOOOF error transfer", "FigS2b1_FOOOF_error_transfer"),
            (spl, "SPLIT error transfer", "FigS2b2_SPLIT_error_transfer"),
        ):
            fig, ax = plt.subplots(figsize=(2.7, 2.0), constrained_layout=True)
            jp = weights[0] / (np.max(np.abs(weights[0, visible])) or 1)
            jchi = weights[1] / (np.max(np.abs(weights[1, visible])) or 1)
            ax.plot(freq[visible], jp[visible], color=RED, lw=1.0, label=r"$J_P$")
            ax.plot(freq[visible], jchi[visible], color=BLUE, lw=1.0, label=r"$J_\chi$")
            ax.set(xlim=(60, 150), xlabel="Frequency (Hz)", ylabel="Normalized weight")
            ax.set_title(title, fontsize=7.2, fontweight="bold")
            ax.legend(frameon=False, ncol=2, loc="lower left")
            finish(fig, out, name)
        for values, title, name in (
            (foo_contrib, "FOOOF covariance contribution", "FigS2b3_contribution_FOOOF"),
            (split_contrib, "SPLIT covariance contribution", "FigS2b4_contribution_SPLIT"),
        ):
            fig, ax = plt.subplots(figsize=(2.7, 2.0), constrained_layout=True)
            normalized = values / (np.max(np.abs(values[visible])) or 1)
            xvis, yvis = freq[visible], normalized[visible]
            ax.fill_between(xvis, 0, yvis, where=yvis >= 0, color=RED, alpha=.9, label="Positive")
            ax.fill_between(xvis, 0, yvis, where=yvis < 0, color=BLUE, alpha=.9, label="Negative")
            ax.set(xlim=(60, 150), xlabel="Frequency (Hz)", ylabel="Normalized term")
            ax.set_title(title, fontsize=7.2, fontweight="bold")
            ax.legend(frameon=False, ncol=2, loc="upper left")
            finish(fig, out, name)
    elif panel == "c":
        fig, ax = plt.subplots(figsize=(3.0, 2.5), constrained_layout=True)
        foo = state["fooof"]; spl = state["split"]
        ax.scatter(foo[:, 0], foo[:, 1], s=6, color=BLUE, alpha=.42, label="FOOOF")
        ax.scatter(spl[:, 0], spl[:, 1], s=6, color=TEAL, alpha=.42, label="IRASA–SPLIT")
        _cov_ellipse(ax, np.nanmean(foo, axis=0), state["v_fooof"], BLUE)
        _cov_ellipse(ax, np.nanmean(spl, axis=0), state["v_split"], TEAL)
        ax.set(xlabel="Estimated peak centre (Hz)", ylabel="Estimated exponent")
        ax.legend(frameon=False)
        finish(fig, out, "FigS2c")
    elif panel == "d":
        frame = pd.read_csv(data / "FigS2d_source_data.csv")
        x = frame.fooof_pred.to_numpy(float); y = frame.fooof_emp.to_numpy(float)
        fig, ax = plt.subplots(figsize=(2.6, 2.35), constrained_layout=True)
        fit = scatter_fit(ax, x, y)
        ax.set(xlabel="Theory-predicted error ρ", ylabel="Held-out error ρ")
        ax.text(.96, .05, f"r={fit.rvalue:.2f}\nslope={fit.slope:.2f}", transform=ax.transAxes, ha="right")
        finish(fig, out, "FigS2d")
    elif panel == "e":
        frame = pd.read_csv(data / "FigS2e_source_data.csv")
        fig, ax = plt.subplots(figsize=(2.8, 2.25), constrained_layout=True)
        ax.fill_between(frame.center, frame.fooof_lo, frame.fooof_hi, color=BLUE, alpha=.14)
        ax.fill_between(frame.center, frame.split_lo, frame.split_hi, color=RED, alpha=.14)
        ax.plot(frame.center, frame.fooof_emp, "o-", color=BLUE, label="FOOOF empirical")
        ax.plot(frame.center, frame.fooof_pred, "--", color=BLUE, label="FOOOF theory")
        ax.plot(frame.center, frame.split_emp, "o-", color=RED, label="SPLIT empirical")
        ax.plot(frame.center, frame.split_pred, "--", color=RED, label="SPLIT theory")
        ax.axhline(0, color="0.5", lw=.6)
        ax.set(xscale="log", xlabel="Oscillation band center (Hz)", ylabel="Spearman ρ", ylim=(-.3, .3))
        ax.legend(frameon=False, fontsize=5.6)
        finish(fig, out, "FigS2e")
    elif panel == "f":
        frame = pd.read_csv(data / "FigS2f_source_data.csv")
        values = [frame.fooof_emp, frame.fooof_observed_minus_predicted, frame.split_emp]
        fig, ax = plt.subplots(figsize=(2.7, 2.25), constrained_layout=True)
        vio = ax.violinplot(values, showmeans=True, showextrema=False)
        for body, color in zip(vio["bodies"], [BLUE, TEAL, RED]):
            body.set(facecolor=color, edgecolor="black", alpha=.8)
        vio["cmeans"].set(color=BLACK, lw=1)
        ax.axhline(0, color="0.5", lw=.6)
        ax.set(xticks=[1, 2, 3], xticklabels=["FOOOF", "FOOOF\nobserved − predicted", "SPLIT"], ylabel="Spearman ρ or Δρ")
        finish(fig, out, "FigS2f")
    else:
        raise ValueError(panel)


DT = .00025
FS = 1 / DT
OMEGA_HZ = 25.0


def _heun(model: str, parameter: float, forcing: np.ndarray) -> np.ndarray:
    x = 0.0; y = 0.0; out = np.empty(len(forcing))
    omega = 2 * np.pi * OMEGA_HZ
    for i, u in enumerate(forcing):
        if model == "linear":
            gamma = parameter * omega
            dx1, dy1 = y + u, -2 * gamma * y - omega**2 * x
            xp, yp = x + DT * dx1, y + DT * dy1
            dx2, dy2 = yp + u, -2 * gamma * yp - omega**2 * xp
        else:
            mu = 2 * np.pi * parameter; r2 = x*x + y*y
            dx1, dy1 = mu*x - omega*y - r2*x + u, omega*x + mu*y - r2*y
            xp, yp = x + DT*dx1, y + DT*dy1; rp2 = xp*xp + yp*yp
            dx2, dy2 = mu*xp - omega*yp - rp2*xp + u, omega*xp + mu*yp - rp2*yp
        x += .5 * DT * (dx1 + dx2); y += .5 * DT * (dy1 + dy2); out[i] = x
    return out


def _forcing(seed: int, duration: float, bump: bool) -> np.ndarray:
    rng = np.random.default_rng(seed); n = int(round(duration / DT)); white = rng.normal(size=n)
    b, a = signal.butter(2, [7/(FS/2), 13/(FS/2)], btype="bandpass")
    osc = signal.sosfiltfilt(signal.butter(2, [7/(FS/2), 13/(FS/2)], btype="bandpass", output="sos"), white)
    osc /= np.sqrt(np.mean(osc**2))
    background = signal.lfilter([1], [1, -.997], rng.normal(size=n)); background /= np.sqrt(np.mean(background**2))
    return background + (2.4 * osc if bump else 0)


def _phase_linear(ratio: float, x0: float, y0: float, duration=.22) -> tuple[np.ndarray, np.ndarray]:
    n = int(duration/DT); x=np.empty(n); y=np.empty(n); x[0]=x0; y[0]=y0*2*np.pi*OMEGA_HZ
    omega=2*np.pi*OMEGA_HZ; gamma=ratio*omega
    for i in range(1,n):
        dx1=y[i-1]; dy1=-2*gamma*y[i-1]-omega**2*x[i-1]
        xp=x[i-1]+DT*dx1; yp=y[i-1]+DT*dy1
        x[i]=x[i-1]+.5*DT*(dx1+yp); y[i]=y[i-1]+.5*DT*(dy1-2*gamma*yp-omega**2*xp)
    return x, y/omega


def _phase_hopf(mu_hz: float, omega_hz: float, x0: float, y0: float, duration=.55) -> tuple[np.ndarray, np.ndarray]:
    n=int(duration/DT); x=np.empty(n); y=np.empty(n); x[0]=x0; y[0]=y0
    mu=2*np.pi*mu_hz; omega=2*np.pi*omega_hz
    for i in range(1,n):
        r2=x[i-1]**2+y[i-1]**2
        dx1=mu*x[i-1]-omega*y[i-1]-r2*x[i-1]; dy1=omega*x[i-1]+mu*y[i-1]-r2*y[i-1]
        xp=x[i-1]+DT*dx1; yp=y[i-1]+DT*dy1; rp2=xp*xp+yp*yp
        dx2=mu*xp-omega*yp-rp2*xp; dy2=omega*xp+mu*yp-rp2*yp
        x[i]=x[i-1]+.5*DT*(dx1+dx2); y[i]=y[i-1]+.5*DT*(dy1+dy2)
    return x,y


def plot_2d(model: str, panel: str, panel_dir: Path) -> None:
    style(9)
    data=panel_dir/"data"; out=panel_dir/"output"
    prefix="FigS3" if model=="linear" else "FigS4"
    if panel in {"a", "e1_e2"}:
        bump = panel == "a"; parameter = (2.0 if model=="linear" else -40.0) if bump else (.16 if model=="linear" else -2.5)
        forcing=_forcing(20260810 if bump else 20260839, 34.816, bump); response=_heun(model,parameter,forcing)
        cut=int(2.048/DT); nseg=int(4.096/DT)
        f,pin=signal.welch(forcing[cut:],fs=FS,nperseg=nseg,noverlap=nseg//2)
        _,pout=signal.welch(response[cut:],fs=FS,nperseg=nseg,noverlap=nseg//2)
        keep=(f>=.5)&(f<=1200); f=f[keep]; pin=pin[keep]; pout=pout[keep]
        if panel=="a":
            fig,axs=plt.subplots(1,2,figsize=(5.4,2.25),constrained_layout=True)
            axs[0].loglog(f,pin/pin.max(),color=RED); axs[1].loglog(f,pout/pout.max(),color=BLUE)
            axs[0].set(xlabel="Frequency (Hz)",ylabel="Normalized PSD"); axs[1].set(xlabel="Frequency (Hz)",ylabel="Normalized PSD")
            finish(fig,out,f"{prefix}a")
        else:
            for name,values,color in ((f"{prefix}e1",pin,GREY),(f"{prefix}e2",pout,BLUE)):
                fig,ax=plt.subplots(figsize=(2.7,2.2),constrained_layout=True); ax.loglog(f,values/values.max(),color=color)
                ax.set(xlabel="Frequency (Hz)",ylabel="Normalized PSD"); finish(fig,out,name)
    elif panel in {"b", "f1"}:
        filename = "exp1_1000_conditions.csv" if (model=="linear" and panel=="b") else "exp2_1000_conditions_rlt0p6.csv" if model=="linear" else "condition_results_exact.csv"
        frame=pd.read_csv(data/filename)
        if panel=="b": xcol,ycol="theory_r","simulation_r"
        else: xcol,ycol="residual_corrected_theory_r","simulation_r"
        x=-frame[xcol].to_numpy(float); y=-frame[ycol].to_numpy(float)
        fig,ax=plt.subplots(figsize=(3.0,2.5),constrained_layout=True); fit=scatter_fit(ax,x,y,ols=(panel=="f1"))
        ax.set(xlabel="Theory corr(P, χ)",ylabel="Simulation corr(P, χ)")
        if panel=="f1": ax.text(.05,.95,f"slope={fit.slope:.3f}",transform=ax.transAxes,va="top")
        finish(fig,out,f"{prefix}{panel}_exponent")
    elif panel=="c":
        fig,ax=plt.subplots(figsize=(3.0,2.4),constrained_layout=True)
        if model=="linear":
            x=np.linspace(2.2,.15,300); omega=2*np.pi*OMEGA_HZ
            eig=np.array([np.linalg.eigvals([[0,1],[-omega**2,-2*r*omega]]) for r in x]); xlabel=r"Damping ratio $\gamma/\omega_0$"
        else:
            x=np.linspace(0,35,300); mu=2*np.pi*(-3); eig=np.array([np.linalg.eigvals([[mu,-2*np.pi*f],[2*np.pi*f,mu]]) for f in x]); xlabel=r"$\Omega/(2\pi)$ (Hz)"
        critical=eig[np.arange(len(x)),np.argmax(eig.real,axis=1)]
        ax.plot(x,critical.real,color=BLUE,label=r"Re($\lambda$)"); ax.plot(x,np.abs(critical.imag),color=RED,label=r"|Im($\lambda$)|")
        ax.set(xlabel=xlabel,ylabel="Eigenvalue (s⁻¹)"); ax.legend(frameon=False); finish(fig,out,f"{prefix}c")
    elif panel=="d1_d3":
        regimes=[("Real",1.8),("Critical",1.0),("Stable complex",.18)] if model=="linear" else [("Real",(-3,0)),("Complex onset",(-3,.5)),("Stable complex",(-3,25))]
        initials=[(1,.15),(-.75,.7),(.45,-.95),(-.8,-.45)]; colors=[BLUE,RED,TEAL]
        fig,axs=plt.subplots(1,3,figsize=(7.2,2.3),constrained_layout=True)
        for ax,(label,param),color in zip(axs,regimes,colors):
            for x0,y0 in initials:
                x,y=_phase_linear(param,x0,y0) if model=="linear" else _phase_hopf(param[0],param[1],x0,y0)
                ax.plot(x,y,color=color,lw=1.1,alpha=.75)
            ax.scatter([0],[0],s=10,c="black"); ax.set(title=label,xlabel="x",ylabel="y",xlim=(-1.15,1.15),ylim=(-1.15,1.15))
        finish(fig,out,f"{prefix}d1_d3")
    elif panel=="f2":
        filename="damped_complex_regime_PA_trials.csv" if model=="linear" else "representative_heldout_trials_exact.csv"
        frame=pd.read_csv(data/filename); x=frame.peak_hz.to_numpy(float); y=-frame.aperiodic_slope.to_numpy(float)
        fig,ax=plt.subplots(figsize=(3.0,2.55),constrained_layout=True); fit=scatter_fit(ax,x,y)
        ax.set(xlabel="Peak frequency P (Hz)",ylabel="Aperiodic exponent χ")
        ax.text(1.0,1.02,f"r={fit.rvalue:.3f}",transform=ax.transAxes,ha="right",va="bottom"); finish(fig,out,f"{prefix}f2_exponent")
    else:
        raise ValueError(panel)
