from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .make_detailed_report import BASE, _fmt, _write_markdown, _write_pdf


def build() -> tuple[Path, Path]:
    output = BASE / "outputs"
    report_dir = BASE / "reports"
    pdf_dir = BASE / "output" / "pdf"
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    exp1 = output / "finite_window_exp1_pairwise_nonlinear"
    exp2 = output / "exp2_nonlinear_hopf"
    exp2_two_stage = output / "exp2_two_stage_scan"
    exp2_stable = output / "exp2_stable_filter_validation"
    exp2_pap = output / "exp2_pap_correlation_validation"
    exp2_window_second = output / "exp2_window_second_order_validation"
    exp3 = output / "finite_window_exp3_pairwise_single_phase_12c_exact"
    exp3_phase = output / "finite_window_exp3_pairwise_phase_ensemble"
    exp3_robust = output / "exp3_trial_seed_robustness"
    exp3_targeted = output / "exp3_targeted_degree10"
    exp3_dt = output / "exp3_targeted_degree10_dt_convergence"
    exp3_maskwise_exp1 = output / "exp3_maskwise_exp1"
    exp3_absolute_phase = output / "exp3_absolute_phase_robustness"
    exp3_large_delta = output / "exp3_large_delta_screen"
    exp3_important = output / "exp3_important_edge_ensemble_validation"
    exp3_emergent_second = output / "exp3_emergent_second_order_validation"

    e1_cov = pd.read_csv(exp1 / "covariance_R2_by_pair.csv")
    e1_cor = pd.read_csv(exp1 / "correlation_R2_by_pair.csv")
    e2_stats = pd.read_csv(exp2 / "condition_regression_stats.csv")
    e2_radial = pd.read_csv(exp2 / "radial_moments.csv")
    e2_metrics = json.loads((exp2 / "metrics.json").read_text())
    e2_two_stage_manifest = json.loads((exp2_two_stage / "manifest.json").read_text())
    e2_two_stage_scan = pd.read_csv(exp2_two_stage / "two_stage_eigen_cross_spectrum.csv")
    e2_stable_manifest = json.loads((exp2_stable / "manifest.json").read_text())
    e2_stable_coupling = pd.read_csv(exp2_stable / "input_colour_coupling.csv")
    e2_stable_nonlinear = pd.read_csv(exp2_stable / "nonlinear_cross_spectrum_selected.csv")
    e2_pap_results = pd.read_csv(exp2_pap / "condition_results.csv")
    e2_pap_metrics = pd.read_csv(exp2_pap / "prediction_metrics.csv").set_index("prediction")
    e2_pap_manifest = json.loads((exp2_pap / "manifest.json").read_text())
    e2_window_results = pd.read_csv(exp2_window_second / "window_length_results.csv")
    e2_second_metrics = pd.read_csv(exp2_window_second / "second_order_prediction_metrics.csv").set_index("prediction")
    e3_metrics = pd.read_csv(exp3 / "delta_correlation_R2_by_pair.csv")
    e3_results = pd.read_csv(exp3 / "delta_correlation_predictions.csv.gz")
    e3_qc = pd.read_csv(exp3 / "rectification_qc.csv")
    e3_phase_metrics = pd.read_csv(exp3_phase / "delta_correlation_R2_by_pair.csv")
    e3_seed_metrics = pd.read_csv(exp3_robust / "seed_diagnostic_metrics.csv")
    e3_seed_stability = pd.read_csv(exp3_robust / "seed_stability_by_trial_count.csv")
    e3_robust_metrics = pd.read_csv(exp3_robust / "robust_9x9_metrics.csv")
    e3_robust_rows = pd.read_csv(exp3_robust / "robust_9x9_delta_correlations.csv.gz")
    e3_targeted_metrics = pd.read_csv(exp3_targeted / "targeted_degree10_metrics.csv")
    e3_targeted_results = pd.read_csv(exp3_targeted / "targeted_degree10_predictions.csv.gz")
    e3_targeted_qc = pd.read_csv(exp3_targeted / "targeted_degree10_qc.csv.gz")
    e3_dt_metrics = pd.read_csv(exp3_dt / "dt_convergence_metrics.csv")
    e3_dt_agreement = pd.read_csv(exp3_dt / "actual_delta_dt_agreement.csv")
    e3_dt_results = pd.read_csv(exp3_dt / "dt_convergence_predictions.csv.gz")
    e3_dt_qc = pd.read_csv(exp3_dt / "dt_convergence_qc.csv.gz")
    e3_maskwise_metrics = pd.read_csv(exp3_maskwise_exp1 / "maskwise_exp1_identity_metrics.csv")
    e3_maskwise_conditions = pd.read_csv(exp3_maskwise_exp1 / "maskwise_exp1_conditions.csv.gz")
    e3_maskwise_phase = pd.read_csv(exp3_maskwise_exp1 / "maskwise_exp1_phase_agreement.csv")
    e3_absolute_noise = pd.read_csv(exp3_absolute_phase / "phase_repeat_maskwise_noise.csv")
    e3_absolute_points = pd.read_csv(exp3_absolute_phase / "phase_repeat_points.csv.gz")
    e3_absolute_manifest = json.loads((exp3_absolute_phase / "manifest.json").read_text())
    e3_large_manifest = json.loads((exp3_large_delta / "manifest.json").read_text())
    e3_large_screen = pd.read_csv(exp3_large_delta / "exact_resolvent_screen.csv.gz")
    e3_large_selected = pd.read_csv(exp3_large_delta / "selected_blind_masks.csv")
    e3_large_validation = pd.read_csv(exp3_large_delta / "long_validation_all_repeats.csv.gz")
    e3_large_metrics = pd.read_csv(exp3_large_delta / "long_validation_metrics.csv")
    e3_important_summary = pd.read_csv(exp3_important / "final_ensemble_summary.csv").iloc[0]
    e3_important_ensembles = pd.read_csv(exp3_important / "ensemble_correlations.csv")
    e3_important_edges = pd.read_csv(exp3_important / "selected_deleted_edges.csv")
    e3_emergent_correlations = pd.read_csv(exp3_emergent_second / "correlation_validation.csv").set_index(["method", "network"])
    e3_emergent_features = pd.read_csv(exp3_emergent_second / "feature_approximation_metrics.csv").set_index(["network", "method"])
    e3_emergent_phases = pd.read_csv(exp3_emergent_second / "phase_repeat_validation.csv")
    e3_emergent_manifest = json.loads((exp3_emergent_second / "manifest.json").read_text())

    def subset(frame: pd.DataFrame, window: str, order: str) -> pd.DataFrame:
        return frame[(frame.window == window) & (frame.order == order)]

    e1csf, e1css = subset(e1_cor, "short", "first"), subset(e1_cor, "short", "second")
    e1clf, e1cls = subset(e1_cor, "long", "first"), subset(e1_cor, "long", "second")
    e1vsf, e1vss = subset(e1_cov, "short", "first"), subset(e1_cov, "short", "second")
    e1vlf, e1vls = subset(e1_cov, "long", "first"), subset(e1_cov, "long", "second")

    sl_stats = e2_stats[e2_stats.layer == "Stuart-Landau"]
    kappa_ci0 = ((sl_stats.kappa_R_ci_low <= 0) & (sl_stats.kappa_R_ci_high >= 0)).mean()
    radial_mean_error = np.median(abs(e2_radial.E_R2_emp - e2_radial.E_R2) / e2_radial.E_R2)
    radial_var_error = np.median(
        abs(e2_radial.Var_R2_emp - e2_radial.Var_R2_theory) / e2_radial.Var_R2_theory
    )
    radial_var_r = e2_radial[["Var_R2_emp", "Var_R2_theory"]].corr().iloc[0, 1]
    sl_csemi = e2_metrics["Stuart-Landau"]["C_semi"]
    sl_cfull = e2_metrics["Stuart-Landau"]["C_full"]
    smooth_csemi = e2_metrics["smooth local E-I"]["C_semi"]
    complex_kappa = float(e2_two_stage_manifest["complex_eigenvalue_onset_kappa_I"])
    hopf_gain = float(e2_two_stage_manifest["hopf_E_E_gain"])
    stage1 = e2_two_stage_scan[e2_two_stage_scan.stage == "I_to_E"]
    stage2 = e2_two_stage_scan[e2_two_stage_scan.stage == "E_to_E"]

    e3sf, e3ss = subset(e3_metrics, "short", "first"), subset(e3_metrics, "short", "second")
    e3lf, e3ls = subset(e3_metrics, "long", "first"), subset(e3_metrics, "long", "second")
    e3_v1pg = e3_metrics[(e3_metrics.input_area == "V1") & (e3_metrics.output_area == "PG")].set_index(
        ["window", "order"]
    )
    phase_v1pg = e3_phase_metrics[
        (e3_phase_metrics.input_area == "V1") & (e3_phase_metrics.output_area == "PG")
    ].set_index(["window", "order"])
    e3_pivot = e3_metrics.pivot(index=["input_area", "output_area", "order"], columns="window", values="R2")
    long_improved = {
        order: int((e3_pivot.xs(order, level="order").long > e3_pivot.xs(order, level="order").short).sum())
        for order in ("first", "second")
    }
    order_pivot = e3_metrics.pivot(index=["input_area", "output_area", "window"], columns="order", values="R2")
    second_improved = {
        window: int(
            (order_pivot.xs(window, level="window")["second"] > order_pivot.xs(window, level="window")["first"]).sum()
        )
        for window in ("short", "long")
    }
    robust_64 = e3_robust_metrics[e3_robust_metrics.n_input_trials == 64]
    robust_512 = e3_robust_metrics[e3_robust_metrics.n_input_trials == 512]
    robust_valid_512 = e3_robust_rows[e3_robust_rows.n_input_trials == 512].n_valid_common

    def robust_subset(frame: pd.DataFrame, window: str, order: str) -> pd.DataFrame:
        return frame[(frame.window == window) & (frame.order == order)]

    r64sf = robust_subset(robust_64, "short", "first")
    r64ss = robust_subset(robust_64, "short", "second")
    r64lf = robust_subset(robust_64, "long", "first")
    r64ls = robust_subset(robust_64, "long", "second")
    r512sf = robust_subset(robust_512, "short", "first")
    r512ss = robust_subset(robust_512, "short", "second")
    r512lf = robust_subset(robust_512, "long", "first")
    r512ls = robust_subset(robust_512, "long", "second")
    seed64short = e3_seed_stability[
        (e3_seed_stability.n_input_trials == 64) & (e3_seed_stability.window == "short")
    ].iloc[0]
    seed512short = e3_seed_stability[
        (e3_seed_stability.n_input_trials == 512) & (e3_seed_stability.window == "short")
    ].iloc[0]
    seed64long = e3_seed_stability[
        (e3_seed_stability.n_input_trials == 64) & (e3_seed_stability.window == "long")
    ].iloc[0]
    seed512long = e3_seed_stability[
        (e3_seed_stability.n_input_trials == 512) & (e3_seed_stability.window == "long")
    ].iloc[0]
    targeted_exact = e3_targeted_metrics[e3_targeted_metrics.order == "exact_resolvent"].copy()
    targeted_table = []
    for row in targeted_exact.itertuples(index=False):
        targeted_table.append([
            f"{row.input_area}/{row.lesion_area}-{row.degree_type}/{row.output_area}",
            "prior significant" if row.prior_class == "previously significant" else "prior null",
            f"{row.prior_5edge_p:.3g}",
            f"{row.median_abs_actual_delta_rho:.3g}",
            _fmt(row.ols_R2),
            _fmt(row.slope),
            _fmt(row.R2),
        ])
    targeted_effect = e3_targeted_results.groupby("prior_class").delta_rho_actual.apply(
        lambda values: float(np.median(np.abs(values)))
    )
    dt_exact = e3_dt_metrics[e3_dt_metrics.order == "exact_resolvent"].copy()
    dt25 = dt_exact[dt_exact.dt_ms == 0.25]
    dt05 = dt_exact[dt_exact.dt_ms == 0.05]
    dt_pivot = dt_exact.pivot(index=["group", "output_area"], columns="dt_ms", values="R2")
    dt_improved = int((dt_pivot[0.05] > dt_pivot[0.25]).sum())
    dt_table = []
    dt_order = pd.concat(
        [
            dt25[dt25.prior_class == "previously significant"],
            dt25[dt25.prior_class == "previously null"],
        ],
        ignore_index=True,
    )
    for row in dt_order.itertuples(index=False):
        fine = dt05[(dt05.group == row.group) & (dt05.output_area == row.output_area)].iloc[0]
        agreement = e3_dt_agreement[
            (e3_dt_agreement.group == row.group) & (e3_dt_agreement.output_area == row.output_area)
        ].iloc[0]
        dt_table.append([
            f"{row.input_area}/{row.lesion_area}-{row.degree_type}/{row.output_area}",
            _fmt(row.R2),
            _fmt(fine.R2),
            _fmt(row.ols_R2),
            _fmt(fine.ols_R2),
            _fmt(agreement.pearson_r),
        ])
    dt25_pg_v4 = dt25[(dt25.group == "PG_V1_out") & (dt25.output_area == "V4")].iloc[0]
    dt05_pg_v4 = dt05[(dt05.group == "PG_V1_out") & (dt05.output_area == "V4")].iloc[0]
    dt_pg_v4_agreement = e3_dt_agreement[
        (e3_dt_agreement.group == "PG_V1_out") & (e3_dt_agreement.output_area == "V4")
    ].iloc[0]

    def maskwise_subset(network_class: str, window: str, order: str) -> pd.DataFrame:
        return e3_maskwise_metrics[
            (e3_maskwise_metrics.network_class == network_class)
            & (e3_maskwise_metrics.window == window)
            & (e3_maskwise_metrics.order == order)
        ]

    maskwise_summary = {
        (network_class, window, order): maskwise_subset(network_class, window, order)
        for network_class in ("baseline", "10-edge deletion")
        for window in ("short", "long")
        for order in ("first", "second", "exact_spectrum")
    }
    maskwise_phase_long = e3_maskwise_phase[
        (e3_maskwise_phase.network_class == "10-edge deletion")
        & (e3_maskwise_phase.window == "long")
        & (e3_maskwise_phase.order.isin(["first", "second"]))
    ].median_pairwise_repeat_r
    absolute_variance_ratio = (
        e3_absolute_noise.delta_simulation_sd
        / np.maximum(e3_absolute_noise.raw_simulation_sd, 1e-15)
    )
    absolute_variance_ratio_by_window = {
        window: float(absolute_variance_ratio[e3_absolute_noise.window == window].median())
        for window in ("short", "long")
    }
    phase_noise_to_theory = []
    exact_phase_points = e3_absolute_points[e3_absolute_points.order == "exact_resolvent"]
    for key, noise_group in e3_absolute_noise.groupby(
        ["window", "group", "input_area", "output_area"], sort=False
    ):
        window, group, input_area, output_area = key
        points = exact_phase_points[
            (exact_phase_points.window == window)
            & (exact_phase_points.group == group)
            & (exact_phase_points.input_area == input_area)
            & (exact_phase_points.output_area == output_area)
        ]
        theory_signal_sd = points.groupby("mask_id").theory_delta_r.mean().std(ddof=1)
        phase_noise_to_theory.append(
            {
                "window": window,
                "ratio": float(noise_group.delta_simulation_sd.median() / theory_signal_sd),
            }
        )
    phase_noise_to_theory = pd.DataFrame(phase_noise_to_theory)

    stable_group = e2_stable_coupling.groupby(
        ["layer", "I_to_E_multiplier"], as_index=False
    )[["m_beta_hz_per_beta", "kappa_beta_per_beta", "P_A_r"]].mean()

    def stable_at(value: float, layer: str) -> pd.Series:
        group = stable_group[stable_group.layer == layer]
        return group.iloc[np.argmin(abs(group.I_to_E_multiplier - value))]

    stable_sc_theory = stable_at(e2_stable_manifest["complex_onset_S"], "exact_resolvent")
    stable_sc_sim = stable_at(e2_stable_manifest["complex_onset_S"], "nonlinear_simulation")
    stable_res_theory = stable_at(e2_stable_manifest["constant_numerator_resonance_onset_S"], "exact_resolvent")
    stable_res_sim = stable_at(e2_stable_manifest["constant_numerator_resonance_onset_S"], "nonlinear_simulation")
    stable_target_cross = e2_stable_nonlinear.iloc[
        np.argmin(abs(e2_stable_nonlinear.I_to_E_multiplier - e2_stable_manifest["target_S"]))
    ]
    pap_valid = e2_pap_results[
        (e2_pap_results.family == "main_grid") & (e2_pap_results.n_valid_common >= 80)
    ]
    pap_reference = pap_valid[
        np.isclose(pap_valid.S, e2_stable_manifest["constant_numerator_resonance_onset_S"])
        & np.isclose(pap_valid.sigma_beta, 0.40)
        & np.isclose(pap_valid.floor_ratio, 0.10)
    ]
    if pap_reference.empty:
        pap_reference = pap_valid.iloc[[np.argmin(abs(pap_valid.S - 4.0))]]
    pap_reference = pap_reference.iloc[0]

    large_phase_mean = e3_large_validation.groupby(
        ["selection_id", "selection_class"], as_index=False
    )[["theory_delta_r_long_design", "simulation_delta_r"]].mean()
    large_high = large_phase_mean[large_phase_mean.selection_class == "high_effect"]
    large_control = large_phase_mean[large_phase_mean.selection_class == "matched_low_effect"]
    large_high_metric = e3_large_metrics[
        (e3_large_metrics.selection_class == "high_effect")
        & (e3_large_metrics.repeat.astype(str) == "phase_mean")
    ].iloc[0]
    large_control_metric = e3_large_metrics[
        (e3_large_metrics.selection_class == "matched_low_effect")
        & (e3_large_metrics.repeat.astype(str) == "phase_mean")
    ].iloc[0]

    sections: list[tuple] = []
    add = sections.append
    add(("title", "周期-非周期耦合理论数值实验 Summary"))
    add(("subtitle", "Experiment 1-3 的理论、真实时域模拟与主要结论"))
    add(("h1", "一、总体问题与判据"))
    add(("p", "三个实验分别检验输入抽样、近Hopf共同振幅变量和结构扰动是否能解释振荡频率P与非周期指数A之间的耦合。真实结果均来自显式时域simulation；resolvent、Taylor展开或正常形公式只作为理论预测。"))
    add(("p", "报告同时使用OLS R2和identity R2。OLS R2检验理论能否线性排序实际结果，允许再拟合斜率和截距；identity R2直接比较predicted与actual，理想值为1，不允许经验校准。高OLS R2但低或负identity R2意味着排序存在，但方向、幅度或截距尚未校准。"))
    add(("callout", "总体结论：Experiment 1的短窗偏差主要来自有限记录的observation layer。Experiment 2的两阶段扫描证明I-to-E可先形成稳定复模态，E-to-E再独立推动Hopf；但100-1000 Hz exponent的kappa_R约为0。Experiment 3中，短窗首先破坏每个网络内部的原始correlation校准；长窗可把baseline和全部删边网络的原始correlation identity R2都恢复到约0.99，但结构Delta r仍因信号极小、peak switching和残余观测误差而难以校准。"))

    add(("h1", "二、Experiment 1：输入改变与长短窗对比"))
    add(("h2", "2.1 方法"))
    add(("p", "在完整55脑区threshold-linear E-I网络中固定结构，只改变输入参数分布。九个脑区依次作为E输入，每次积分同时读取九个E输出，得到9 x 9路径。每个输入位置包含14个condition，每condition 64个trial，共8064条长窗网络轨迹。burn-in=2.048 s，dt=0.25 ms。短窗使用同一长轨迹前4.096 s和2.048 s Hann-Welch段；长窗使用32.768 s和4.096 s Hann-Welch段。"))
    add(("p", "理论一阶与二阶分别为输入feature map F=(P,A)在基准输入附近的线性和完整二次Taylor展开。对每条input-output路径，比较14个condition上的理论与真实时域covariance及Pearson correlation。"))
    add(("eq", "F_first(u) = F(0) + grad(F)^T u\nF_second(u) = F_first(u) + 1/2 u^T H_F u"))
    add(("resulttable", ["Experiment 1: 81路径中位数", "short first", "short second", "long first", "long second", "解释"], [
        ["covariance OLS R2", _fmt(e1vsf.ols_R2.median()), _fmt(e1vss.ols_R2.median()), _fmt(e1vlf.ols_R2.median()), _fmt(e1vls.ols_R2.median()), "condition排序"],
        ["covariance identity R2", _fmt(e1vsf.R2.median()), _fmt(e1vss.R2.median()), _fmt(e1vlf.R2.median()), _fmt(e1vls.R2.median()), "绝对校准"],
        ["correlation OLS R2", _fmt(e1csf.ols_R2.median()), _fmt(e1css.ols_R2.median()), _fmt(e1clf.ols_R2.median()), _fmt(e1cls.ols_R2.median()), "condition排序"],
        ["correlation identity R2", _fmt(e1csf.R2.median()), _fmt(e1css.R2.median()), _fmt(e1clf.R2.median()), _fmt(e1cls.R2.median()), "绝对校准"],
        ["correlation slope", _fmt(e1csf.slope.median()), _fmt(e1css.slope.median()), _fmt(e1clf.slope.median()), _fmt(e1cls.slope.median()), "理想值1"],
    ]))
    add(("p", f"短窗correlation slope中位数仅约 {e1css.slope.median():.3f}，identity R2约 {e1css.R2.median():.3f}，显示明显attenuation；长窗后slope约 {e1cls.slope.median():.3f}、OLS R2={e1cls.ols_R2.median():.3f}、identity R2={e1cls.R2.median():.3f}。一阶和二阶结果几乎相同，说明当前输入扰动幅度下主要误差不是缺少Hessian，而是短记录PSD feature与correlation estimator的方差和偏差。"))
    add(("figure", exp1 / "figures" / "exp1_correlation_9x9_identity_R2.png", "图 S1. Experiment 1的9 x 9 correlation identity R2。短窗全部路径严重失配，长窗全部恢复至接近1。"))
    add(("conclusion", "Experiment 1结论：输入抽样几何可决定P-A耦合的符号和幅度。完整55脑区真实时域模拟支持理论feature map；4.096 s短窗主要压缩相关幅度，而32.768 s长窗几乎恢复全部81条路径的排序和identity校准。"))

    add(("h1", "三、Experiment 2：近Hopf理论、实际结果与kappa_R约为0"))
    add(("h2", "3.1 方法与理论"))
    add(("p", "Stuart-Landau层扫描90个condition，每condition运行8个seed；总时长80 s、burn-in 10 s，以4 s窗口提取平均R2、瞬时频率dotPhi和100-1000 Hz PSD exponent A。偶数seed训练c_i和kappa_R，奇数seed作为held-out。另在PG局部smooth E-I系统中扫描Hopf距离和噪声强度，并以linearized E-I作为负对照。"))
    add(("eq", "p_R(r) proportional to r exp[mu r^2/(2D) - c_r r^4/(4D)]"))
    add(("eq", "dotPhi = omega0 - c_i R^2 + phase noise\nA = chi0 + kappa_R R^2 + residual\nCov(dotPhi,A) approximately -c_i kappa_R Var(R^2)"))
    add(("h2", "3.2 径向平稳分布的验证"))
    add(("p", f"理论与经验E[R2]的中位相对误差为 {radial_mean_error:.2%}，Var(R2)的中位相对误差为 {radial_var_error:.2%}；跨90个condition的理论-经验Var(R2) Pearson r={radial_var_r:.4f}。因此随机积分器、噪声convention和式（136）的径向统计得到强定量支持。"))
    add(("figure", exp2 / "figures" / "radial_density.png", "图 S2. 代表性Stuart-Landau condition的经验径向分布与式（136）理论密度。"))
    add(("h2", "3.3 kappa_R约为0的验证"))
    add(("p", f"在90个Stuart-Landau condition中，kappa_R估计中位数={sl_stats.kappa_R_estimate.median():.4f}、均值={sl_stats.kappa_R_estimate.mean():.4f}、绝对值中位数={sl_stats.kappa_R_estimate.abs().median():.4f}。A~mean(R2)的R2中位数仅 {sl_stats.A_R2_R2.median():.4f}；{kappa_ci0:.1%} 的95% CI包含0，只有 {(sl_stats.A_R2_p<0.05).mean():.1%} condition达到p<0.05，与名义偶然水平接近。"))
    add(("p", "这里的结论不是kappa_R在所有频带或所有模型中严格等于0，而是本项目预注册的100-1000 Hz exponent对近10 Hz振荡振幅R2缺乏可辨识敏感性。由于Eq.134的乘积项包含kappa_R，该通路被直接压低。"))
    add(("figure", exp2 / "figures" / "common_amplitude_driver.png", "图 S3. 代表condition中dotPhi与A分别对mean(R2)的训练回归。A通路几乎不可辨识，对应kappa_R约为0。"))
    add(("h2", "3.4 理论与实际耦合比较"))
    add(("resulttable", ["Experiment 2 held-out", "Pearson r", "identity R2", "slope", "RMSE", "符号准确率"], [
        ["Stuart-Landau C_semi", _fmt(sl_csemi["pearson_r"]), _fmt(sl_csemi["R2"]), _fmt(sl_csemi["slope"]), _fmt(sl_csemi["RMSE"]), _fmt(sl_csemi["sign_accuracy"])],
        ["Stuart-Landau C_full", _fmt(sl_cfull["pearson_r"]), _fmt(sl_cfull["R2"]), _fmt(sl_cfull["slope"]), _fmt(sl_cfull["RMSE"]), _fmt(sl_cfull["sign_accuracy"])],
        ["smooth local E-I C_semi", _fmt(smooth_csemi["pearson_r"]), _fmt(smooth_csemi["R2"]), _fmt(smooth_csemi["slope"]), _fmt(smooth_csemi["RMSE"]), _fmt(smooth_csemi["sign_accuracy"])],
    ]))
    add(("figure", exp2 / "figures" / "eq134_observed_predicted.png", "图 S4. Eq.134 semi-theory与held-out observed covariance。Stuart-Landau和smooth E-I均没有形成稳健的跨condition预测。"))
    add(("h2", "3.5 两阶段单参数扫描：先形成复模态，再推动Hopf"))
    add(("p", "为消除旧对角homotopy同时改变两个参数造成的路径依赖，重新进行两个连续但彼此独立的扫描。阶段1固定E-to-E gain=1，只把PG局部I-to-E抑制倍数kappa_I从1增加到6.694；阶段2固定kappa_I=6.694，只把E-to-E gain从1增加到Hopf之后。这样“实根变成复根”和“复根实部过零”分别由不同参数控制。"))
    add(("eq", "Stage 1:  g_EE = 1,       kappa_I: 1 -> 6.693999\nStage 2:  kappa_I = 6.693999,  g_EE: 1 -> 4.419376\nJ = diag(tau)^(-1) [W - I]"))
    add(("p", "阶段1的复根起点由[tr J]^2-4 det J=0求解；阶段2的Hopf点由max Re lambda(J)=0且Im lambda不为0求解。cross spectrum理论仍采用S(f)=R(f)R(f)*，非线性验证使用smooth tanh E-I积分：dt=0.25 ms、80 s、burn-in=10 s、8个seed、噪声强度0.002，并将分析频带扩到80 Hz以覆盖阶段1的高频阻尼模态。"))
    add(("resulttable", ["事件", "固定参数", "扫描参数值", "临界实部 [s^-1]", "模态频率 [Hz]", "解释"], [
        ["复共轭根首次出现", "g_EE=1", f"kappa_I={complex_kappa:.4f}", f"{np.interp(complex_kappa, stage1.scan_value, stage1.critical_real_per_s):.2f}", "约0", "稳定node转稳定focus"],
        ["阶段1终点", "g_EE=1", "kappa_I=6.6940", f"{stage1.iloc[-1].critical_real_per_s:.2f}", f"{stage1.iloc[-1].critical_frequency_hz:.2f}", "高频但强阻尼"],
        ["Hopf crossing", "kappa_I=6.6940", f"g_EE={hopf_gain:.4f}", "0", f"{np.interp(hopf_gain, stage2.scan_value, stage2.critical_frequency_hz):.2f}", "自持振荡边界"],
    ]))
    add(("figure", exp2_two_stage / "figures" / "exp2_two_stage_eigen_cross_spectrum.png", "图 S5. 两阶段扫描的临界实部、模态频率和E-I cross-spectrum峰。阶段1只增加I-to-E并形成稳定复模态；阶段2只增加E-to-E，使阻尼逐步减小并在红线处发生Hopf。"))
    add(("figure", exp2_two_stage / "figures" / "exp2_two_stage_cross_spectrum_examples.png", "图 S6. 两阶段扫描的代表性非线性E-I cross spectrum。阶段1末端存在约69-75 Hz但强阻尼的宽峰；阶段2接近Hopf时形成约23-26 Hz的窄而清晰的振荡峰。"))
    add(("h2", "3.6 新推导验证一：稳定node-to-focus路径"))
    add(("p", "按照内部推导，只改变PG局部I-to-E倍数S，固定E-to-E gain=1。对S=1到6.694进行1099点扫描，并在复根起点附近采用小于0.001的加密步长。每一点直接对2 x 2 Jacobian求特征值、判别式、阻尼和exact resolvent；另在S=1.5、Sc、2.4、共振起点、5和6.694运行smooth tanh E-I非线性模拟。时域参数为dt=0.25 ms、80 s、burn-in=10 s、8 seeds、噪声0.002。"))
    add(("eq", "Delta(S) = [tr J(S)]^2 - 4 det J(S)\nf_lambda(S) = |Im lambda(S)|/(2 pi)\nf_res(S) = sqrt(max(omega(S)^2 - gamma(S)^2, 0))/(2 pi)"))
    add(("resulttable", ["稳定I-to-E路径事件", "S", "数值结果", "解释"], [
        ["复根首次出现", f"{e2_stable_manifest['complex_onset_S']:.6f}", "Delta=0", "稳定node转稳定focus"],
        ["f_lambda^2线性律", "S > Sc", f"斜率={e2_stable_manifest['frequency_squared_slope_hz2_per_S']:.2f} Hz2/S; R2={e2_stable_manifest['frequency_squared_R2']:.6f}", "支持平方根起始"],
        ["恒分子共振首次允许", f"{e2_stable_manifest['constant_numerator_resonance_onset_S']:.6f}", "omega=gamma", "晚于复根起点"],
        ["路径终点", f"{e2_stable_manifest['target_S']:.6f}", f"Re(lambda)={e2_stable_manifest['critical_real_per_s']:.2f} s^-1; tau_decay={e2_stable_manifest['decay_time_ms']:.3f} ms", "仍强稳定，无临界减速"],
        ["非线性cross-spectrum峰", f"S={e2_stable_manifest['target_S']:.3f}", f"{stable_target_cross.sim_cross_spectrum_peak_hz:.2f} Hz", "本实现指标下低于内部note的69-75 Hz范围"],
    ]))
    add(("figure", exp2_stable / "figures" / "stable_node_focus_validation.png", "图 S21. 稳定node-to-focus推导验证。复根起点Sc、共振允许点Sres和Hopf是三个不同事件；沿I-to-E路径实部固定为约-226.5 s^-1。"))
    add(("p", "结果严格支持推导中最关键的区分：出现Im(lambda)不等于出现可辨识PSD共振，更不等于Hopf。f_lambda^2对S-Sc的截距约为数值零，斜率1186.19 Hz2/S，与推导的1185.90高度一致；但非线性cross-spectrum峰在S=6.694时为%.2f Hz，说明具体读出、噪声注入和cross-spectrum峰定义会让峰位偏离恒分子近似，不能把64.6 Hz或note中的69-75 Hz当成普适常数。" % stable_target_cross.sim_cross_spectrum_peak_hz))

    add(("h2", "3.7 新推导验证二：稳定滤波器的input-colour P-A耦合"))
    add(("p", "在同一稳定E-I路径上令输入PSD与f^beta成比例，并叠加固定白噪声floor。beta在[-0.8,0.8]内抽取81个trial；每个S运行3个独立phase realization。非线性Heun积分使用dt=0.25 ms、burn-in=2.048 s、分析窗32.768 s、4.096 s Hann-Welch段，colored input sigma=0.02、white floor sigma=0.002。P在5-100 Hz内取峰，A为100-1000 Hz signed log-log slope。exact理论直接使用H(f;S)Q(f;beta)，没有对S作结构Taylor展开。"))
    add(("eq", "m_beta = dP/d beta,     kappa_beta = dA/d beta\nr_PA approximately m_beta kappa_beta Var(beta) / sqrt[(m_beta^2 Var(beta)+sigma_P^2)(kappa_beta^2 Var(beta)+sigma_A^2)]"))
    add(("resulttable", ["位置", "m_beta theory/sim [Hz]", "kappa_beta theory/sim", "r(P,A) theory/sim"], [
        ["复根起点Sc", f"{stable_sc_theory.m_beta_hz_per_beta:.2f} / {stable_sc_sim.m_beta_hz_per_beta:.2f}", f"{stable_sc_theory.kappa_beta_per_beta:.3f} / {stable_sc_sim.kappa_beta_per_beta:.3f}", f"{stable_sc_theory.P_A_r:.3f} / {stable_sc_sim.P_A_r:.3f}"],
        ["共振起点附近", f"{stable_res_theory.m_beta_hz_per_beta:.2f} / {stable_res_sim.m_beta_hz_per_beta:.2f}", f"{stable_res_theory.kappa_beta_per_beta:.3f} / {stable_res_sim.kappa_beta_per_beta:.3f}", f"{stable_res_theory.P_A_r:.3f} / {stable_res_sim.P_A_r:.3f}"],
    ]))
    add(("figure", exp2_stable / "figures" / "stable_filter_input_colour_coupling.png", "图 S22. 沿稳定I-to-E路径的m_beta、kappa_beta和P-A correlation。实线为exact resolvent，虚线为非线性长窗模拟；竖线为复根起点。"))
    add(("figure", exp2_stable / "figures" / "stable_filter_input_colour_examples.png", "图 S23. 共振起点和路径终点的trial-level P-A散点。颜色表示beta；图内报告相关系数和p值。"))
    add(("p", "该实验验证了“无需接近Hopf，稳定滤波器也可把输入颜色共同映射到P和A”这一机制。Sc处m_beta理论/模拟为%.2f/%.2f Hz，Sres附近为%.2f/%.2f Hz；kappa_beta保持为正，因此P-A相关性为正。与此同时，路径终点的理论与非线性峰敏感度开始明显分离，表明强滤波和有限窗下的峰选择非线性仍需进入观测模型。它支持机制存在，但不支持把exact-resolvent数值在所有S上视为无偏定量预测。" % (stable_sc_theory.m_beta_hz_per_beta, stable_sc_sim.m_beta_hz_per_beta, stable_res_theory.m_beta_hz_per_beta, stable_res_sim.m_beta_hz_per_beta)))

    add(("h2", "3.8 P-A correlation新理论的系统定量验证"))
    add(("p", "进一步把稳定滤波器推导直接作为可证伪的correlation预测。局部smooth tanh E-I网络沿九个I-to-E倍数S扫描；每个S交叉扫描5个beta标准差和6个白噪声floor，并加入两档输入振幅负对照，共288个condition。每个condition固定256个截断高斯beta trial，运行3个calibration phase和5个完全独立held-out phase。Heun步长0.25 ms、burn-in 2.048 s、分析窗32.768 s；以4.096 s Hann-Welch估计PSD，P为5-100 Hz最高内部峰，A为100-1000 Hz signed log-log slope。"))
    add(("eq", "L(z,beta)=log |H(exp(z);S)|^2 + log Q(exp(z);beta)\n"
               "m_beta=dP/d beta = -P L_(z beta)/L_(zz)\n"
               "kappa_beta=dA/d beta = OLS_slope_z[d log Q/d beta]\n"
               "r_PA = Cov(P,A)/sqrt(Var(P)Var(A))"))
    add(("p", "三层预测彼此预注册并分开检验：一是对每个beta直接计算exact ideal spectrum H(f;S)Q(f;beta)后得到r；二是只使用m_beta和kappa_beta的局部一阶映射；三是在一阶信号协方差上加入3个calibration phase估计的peak/slope残差协方差，再预测held-out r。另用离散线性E-I时域滤波作为有限窗负对照，真实检验层始终是非线性Heun轨迹。解析m_beta和kappa_beta均用中心有限差分独立核验。"))
    add(("resulttable", ["跨condition held-out预测", "n", "Pearson r", "identity R2", "OLS slope", "RMSE", "95% coverage"], [
        ["exact beta mapping", int(e2_pap_metrics.loc["exact_mapping_r", "n"]), _fmt(e2_pap_metrics.loc["exact_mapping_r", "pearson_r"]), _fmt(e2_pap_metrics.loc["exact_mapping_r", "R2"]), _fmt(e2_pap_metrics.loc["exact_mapping_r", "slope"]), _fmt(e2_pap_metrics.loc["exact_mapping_r", "RMSE"]), _fmt(e2_pap_metrics.loc["exact_mapping_r", "coverage"])],
        ["local first order", int(e2_pap_metrics.loc["local_first_order_r", "n"]), "constant r=1", _fmt(e2_pap_metrics.loc["local_first_order_r", "R2"]), "not identifiable", _fmt(e2_pap_metrics.loc["local_first_order_r", "RMSE"]), "-"],
        ["residual corrected", int(e2_pap_metrics.loc["residual_corrected_r", "n"]), _fmt(e2_pap_metrics.loc["residual_corrected_r", "pearson_r"]), _fmt(e2_pap_metrics.loc["residual_corrected_r", "R2"]), _fmt(e2_pap_metrics.loc["residual_corrected_r", "slope"]), _fmt(e2_pap_metrics.loc["residual_corrected_r", "RMSE"]), "-"],
        ["exact vs linear time-domain", int(e2_pap_metrics.loc["exact_mapping_vs_linear_time_domain", "n"]), _fmt(e2_pap_metrics.loc["exact_mapping_vs_linear_time_domain", "pearson_r"]), _fmt(e2_pap_metrics.loc["exact_mapping_vs_linear_time_domain", "R2"]), _fmt(e2_pap_metrics.loc["exact_mapping_vs_linear_time_domain", "slope"]), _fmt(e2_pap_metrics.loc["exact_mapping_vs_linear_time_domain", "RMSE"]), "-"],
    ]))
    add(("resulttable", ["代表condition", "theory", "nonlinear held-out", "说明"], [
        ["m_beta [Hz/beta]", _fmt(pap_reference.m_beta_implicit), f"{pap_reference.nonlinear_m_beta:.3f} [{pap_reference.nonlinear_m_beta_ci_low:.3f}, {pap_reference.nonlinear_m_beta_ci_high:.3f}]", "peak sensitivity"],
        ["kappa_beta", _fmt(pap_reference.kappa_beta_analytic), f"{pap_reference.nonlinear_kappa_beta:.3f} [{pap_reference.nonlinear_kappa_beta_ci_low:.3f}, {pap_reference.nonlinear_kappa_beta_ci_high:.3f}]", "100-1000 Hz slope sensitivity"],
        ["r(P,A)", _fmt(pap_reference.exact_mapping_r), f"{pap_reference.nonlinear_heldout_r_mean:.3f} +/- {pap_reference.nonlinear_heldout_r_sd:.3f}", f"S={pap_reference.S:.3f}, sigma_beta={pap_reference.sigma_beta:.2f}, floor={pap_reference.floor_ratio:.2f}"],
    ]))
    add(("figure", exp2_pap / "figures" / "exp2_pap_path_validation.png", "图 S29. 沿稳定I-to-E路径比较模态、m_beta、kappa_beta、理论/非线性P-A correlation及峰值可辨识性。"))
    add(("figure", exp2_pap / "figures" / "exp2_pap_variance_floor.png", "图 S30. beta方差和白噪声floor对P-A correlation及kappa_beta的系统影响。"))
    add(("figure", exp2_pap / "figures" / "exp2_pap_representative_trials.png", "图 S31. 代表condition的trial-level beta-P、beta-A与P-A散点；红线为exact-spectrum映射或线性回归。"))
    add(("figure", exp2_pap / "figures" / "exp2_pap_theory_simulation_identity.png", "图 S32. exact mapping、局部一阶和残差修正理论对非线性held-out correlation的跨condition定量检验。"))
    add(("figure", exp2_pap / "figures" / "exp2_pap_error_decomposition.png", "图 S33. 理想谱、离散线性时域与非线性时域之间的误差分解。"))
    add(("p", f"共有{len(pap_valid)}个condition达到至少80个共同有效trial。解析导数与有限差分的一致性分别达到m_beta最大相对误差{e2_pap_manifest['max_analytic_finite_m_relative_error']:.2g}、kappa_beta最大相对误差{e2_pap_manifest['max_analytic_finite_kappa_relative_error']:.2g}。该网格直接回答理论是否能预测真实P-A correlation，而不只验证复根transition。"))
    add(("p", f"结果支持机制方向，但不支持把无观测误差的exact谱相关性直接当作真实有限窗相关性。exact beta mapping跨condition Pearson r={e2_pap_metrics.loc['exact_mapping_r','pearson_r']:.3f}、identity R2={e2_pap_metrics.loc['exact_mapping_r','R2']:.3f}；局部一阶模型因为P和A都只是同一个beta的线性函数，理论r退化为常数+1，无法解释有限窗attenuation。加入独立calibration phase估计的残差方差和协方差后，held-out Pearson r提高到{e2_pap_metrics.loc['residual_corrected_r','pearson_r']:.3f}、identity R2={e2_pap_metrics.loc['residual_corrected_r','R2']:.3f}、slope={e2_pap_metrics.loc['residual_corrected_r','slope']:.3f}。因此稳定滤波器的input-colour机制成立，但定量correlation理论必须显式包含PSD feature observation layer；残差修正取得实质改善，尚未达到严格identity校准。"))

    add(("h2", "3.9 增加窗长与input-colour二阶展开"))
    e2_window = e2_window_results.set_index("spec")
    add(("p", "为直接检验图S31中的有限窗解释，在代表condition S=6、sigma_beta=0.4、white-floor ratio=0.1上固定256个beta trial和3个独立Fourier phase，一次生成131.072 s非线性Heun轨迹，再读取32.768、65.536和131.072 s前缀。三种主比较均保持4.096 s Hann-Welch segment，因此只改变可平均segment数量。另比较8.192和16.384 s segment，区分总记录长度与单段长度。"))
    add(("resulttable", ["analysis window", "Welch segments", "simulation r(P,A)", "residual SD(P)", "P-exact RMSE", "r(beta,P)"], [
        ["32.768 s", int(e2_window.loc["T32_L4", "n_welch_segments"]), f"{e2_window.loc['T32_L4','phase_ensemble_r_PA']:.4f}", f"{e2_window.loc['T32_L4','residual_sd_P_hz']:.2f} Hz", f"{e2_window.loc['T32_L4','P_theory_simulation_RMSE_hz']:.2f} Hz", f"{e2_window.loc['T32_L4','r_beta_P']:.4f}"],
        ["65.536 s", int(e2_window.loc["T65_L4", "n_welch_segments"]), f"{e2_window.loc['T65_L4','phase_ensemble_r_PA']:.4f}", f"{e2_window.loc['T65_L4','residual_sd_P_hz']:.2f} Hz", f"{e2_window.loc['T65_L4','P_theory_simulation_RMSE_hz']:.2f} Hz", f"{e2_window.loc['T65_L4','r_beta_P']:.4f}"],
        ["131.072 s", int(e2_window.loc["T131_L4", "n_welch_segments"]), f"{e2_window.loc['T131_L4','phase_ensemble_r_PA']:.4f}", f"{e2_window.loc['T131_L4','residual_sd_P_hz']:.2f} Hz", f"{e2_window.loc['T131_L4','P_theory_simulation_RMSE_hz']:.2f} Hz", f"{e2_window.loc['T131_L4','r_beta_P']:.4f}"],
    ]))
    add(("p", f"代表condition的exact-spectrum r={e2_window.loc['T131_L4','exact_r_same_trials']:.4f}。分析窗增至131.072 s后，simulation r从{e2_window.loc['T32_L4','phase_ensemble_r_PA']:.3f}升至{e2_window.loc['T131_L4','phase_ensemble_r_PA']:.3f}，peak residual SD从{e2_window.loc['T32_L4','residual_sd_P_hz']:.2f}降至{e2_window.loc['T131_L4','residual_sd_P_hz']:.2f} Hz。固定总时长时，增大Welch segment没有继续改善，说明主要收益来自更多segment平均，而不是只提高单段频率分辨率。"))
    add(("eq", "P2(beta)=P0+P_beta beta+1/2 P_(beta beta) beta^2\nA2(beta)=A0+A_beta beta+1/2 A_(beta beta) beta^2\nr2=Corr[P2(beta),A2(beta)]"))
    add(("p", "二阶相关性使用匹配阶数的P2、A2 covariance和variance，并在每个condition内加入由3个calibration phase估计的有限窗残差协方差；残差自由度修正为Ntrial*(Q-1)，避免旧实现约2/3的低估。真实target为5个独立held-out phase correlation的均值。"))
    add(("pagebreak",))
    add(("resulttable", ["148个共同有效condition", "prediction-vs-heldout r", "identity R2", "OLS slope", "RMSE"], [
        ["exact mapping", _fmt(e2_second_metrics.loc["exact_mapping_r", "pearson_r"]), _fmt(e2_second_metrics.loc["exact_mapping_r", "R2"]), _fmt(e2_second_metrics.loc["exact_mapping_r", "slope"]), _fmt(e2_second_metrics.loc["exact_mapping_r", "RMSE"])],
        ["first order", "constant r=1", _fmt(e2_second_metrics.loc["first_order_r", "R2"]), "NA", _fmt(e2_second_metrics.loc["first_order_r", "RMSE"])],
        ["second order", _fmt(e2_second_metrics.loc["second_order_r", "pearson_r"]), _fmt(e2_second_metrics.loc["second_order_r", "R2"]), _fmt(e2_second_metrics.loc["second_order_r", "slope"]), _fmt(e2_second_metrics.loc["second_order_r", "RMSE"])],
        ["first + residual", _fmt(e2_second_metrics.loc["first_order_residual_r", "pearson_r"]), _fmt(e2_second_metrics.loc["first_order_residual_r", "R2"]), _fmt(e2_second_metrics.loc["first_order_residual_r", "slope"]), _fmt(e2_second_metrics.loc["first_order_residual_r", "RMSE"])],
        ["second + residual", _fmt(e2_second_metrics.loc["second_order_residual_r", "pearson_r"]), _fmt(e2_second_metrics.loc["second_order_residual_r", "R2"]), _fmt(e2_second_metrics.loc["second_order_residual_r", "slope"]), _fmt(e2_second_metrics.loc["second_order_residual_r", "RMSE"])],
    ]))
    add(("figure", exp2_window_second / "figures" / "exp2_window_length_representative.png", "图 S34. S31风格的32.768、65.536与131.072 s长窗比较；三行使用相同beta trial和4.096 s Welch segment。"))
    add(("figure", exp2_window_second / "figures" / "exp2_window_length_summary.png", "图 S35. 总分析时长及Welch segment对r(P,A)、peak residual、P-exact RMSE与r(beta,P)的影响。"))
    add(("figure", exp2_window_second / "figures" / "exp2_second_order_representative.png", "图 S36. 代表condition的exact、一阶、二阶feature map与非线性held-out trial。"))
    add(("figure", exp2_window_second / "figures" / "exp2_second_order_identity.png", "图 S37. S32风格的一阶、二阶、残差修正及残差对照的跨condition检验。"))
    add(("p", "二阶在S=6代表condition中紧贴exact curve，但在完整网格不改善held-out correlation。接近complex transition的S=3.177条件出现peak identity/position高曲率，P_(beta beta)最高约-1955 Hz/beta2；当sigma_beta达到0.2-0.6时，单点局部二次多项式越出适用域。当前最好的跨condition模型仍是一阶信号加condition-specific finite-window residual。"))
    add(("conclusion", "Experiment 2结论：复特征值、可辨识共振和Hopf是可分离事件。I-to-E在S=2.0816形成稳定复模态，恒分子共振条件到S约3.177才满足，且整条路径没有临界减速；随后只有增强E-to-E才在g_EE=4.3394推动Hopf。稳定滤波器本身可通过输入颜色beta产生P-A耦合，但定量峰敏感度仍受有限窗和峰选择影响。原100-1000 Hz kappa_R约为0的共同振幅负结果与这一input-colour正机制并不矛盾。"))

    add(("h1", "四、Experiment 3：9 x 9结构理论与真实simulation"))
    add(("h2", "4.1 方法"))
    add(("p", "使用真实九脑区threshold-linear E-I子网络，九个脑区依次作为E输入，每次积分同时读取九个E输出。固定图12c的单相位seed offset=220000、64个输入trial和120个held-out random-15删边mask。短窗4.096 s与长窗32.768 s分别合成各自长度的Fourier输入；burn-in均为2.048 s，dt=0.25 ms。每条input-output-window路径在baseline与120个mask的共同peak-valid trial上计算真实Delta r。"))
    add(("eq", "Delta r_first = D_J r[Delta J]\nDelta r_second = Delta r_first + 1/2 D_J^2 r[Delta J,Delta J]"))
    add(("p", "每个identity R2以120个mask为样本：横轴是结构一阶或二阶预测，纵轴是九脑区真实时域simulation得到的Delta r。短窗共同有效输入trial为52-59，中位数58；长窗为48-64，中位数57。最大rectification fraction极低，约5.86e-7。"))
    add(("resulttable", ["Experiment 3: 81路径", "short first", "short second", "long first", "long second", "解释"], [
        ["OLS R2 中位数", _fmt(e3sf.ols_R2.median()), _fmt(e3ss.ols_R2.median()), _fmt(e3lf.ols_R2.median()), _fmt(e3ls.ols_R2.median()), "mask排序"],
        ["identity R2 中位数", _fmt(e3sf.R2.median()), _fmt(e3ss.R2.median()), _fmt(e3lf.R2.median()), _fmt(e3ls.R2.median()), "绝对校准"],
        ["identity R2 > 0", _fmt((e3sf.R2>0).mean()), _fmt((e3ss.R2>0).mean()), _fmt((e3lf.R2>0).mean()), _fmt((e3ls.R2>0).mean()), "优于actual均值"],
        ["identity R2 > 0.5", _fmt((e3sf.R2>0.5).mean()), _fmt((e3ss.R2>0.5).mean()), _fmt((e3lf.R2>0.5).mean()), _fmt((e3ls.R2>0.5).mean()), "较强校准"],
        ["正OLS slope比例", _fmt((e3sf.slope>0).mean()), _fmt((e3ss.slope>0).mean()), _fmt((e3lf.slope>0).mean()), _fmt((e3ls.slope>0).mean()), "方向一致"],
    ]))
    add(("p", f"V1->PG内部对照得到短窗一阶/二阶identity R2={e3_v1pg.loc[('short','first'),'R2']:.3f}/{e3_v1pg.loc[('short','second'),'R2']:.3f}，长窗={e3_v1pg.loc[('long','first'),'R2']:.3f}/{e3_v1pg.loc[('long','second'),'R2']:.3f}，复现图12c的长窗改善。但全81路径中位数四种条件均为负。长窗相对短窗在一阶/二阶各改善{long_improved['first']}/81和{long_improved['second']}/81条路径；二阶相对一阶在短窗/长窗改善{second_improved['short']}/81和{second_improved['long']}/81条路径。"))
    add(("figure", exp3 / "figures" / "exp3_12c_single_phase_9x9_identity_R2.png", "图 S7. 与图12c相同单相位realization下，Experiment 3的9 x 9一阶/二阶、短窗/长窗identity R2。"))
    add(("p", f"相位稳健性复算进一步显示，六相位mean-PSD条件下V1->PG二阶identity R2从短窗 {phase_v1pg.loc[('short','second'),'R2']:.3f} 变为长窗 {phase_v1pg.loc[('long','second'),'R2']:.3f}。因此单相位V1->PG的0.774不能作为整个系统或跨相位的普遍校准证据。"))
    add(("conclusion", "Experiment 3的9 x 9初步结论：结构Taylor kernel能够保留部分mask排序，且二阶通常比一阶更好；但绝对Delta-r校准在大多数路径上仍不足。长窗改善是路径特异的，随机相位还会改变结果方向和尺度。"))

    add(("h2", "4.2 定向集中删除10条in/out-degree边"))
    add(("p", "为了验证较强、集中且具脑区归属的结构扰动，从此前5边实验中选择三组曾出现p<0.01的input-lesion-degree-output组合，并在同一input-lesion-degree下各配一个此前不显著的output。显著组为A6Va输入/删MST入边/读MST、V2输入/删A6Va入边/读A6Va、PG输入/删V1出边/读V4；对应null readout分别为A23a、V4和V2。此前5边p值只用于分层选例，不作为本次10边实验的显著性结论。"))
    add(("p", "每组在完整55脑区FLN中生成40个独立mask；每个mask从指定脑区的全部非零in-degree或out-degree中无放回抽取10条并删除。每个网络由512个共同输入trial驱动，dt=0.25 ms、burn-in=2.048 s、分析窗4.096 s；在baseline和全部40个mask共同peak-valid的trial上计算P与A的Pearson r及Delta r。共同有效trial为%d-%d，整流比例最大值为%.2g。" % (e3_targeted_results.n_valid_common.min(), e3_targeted_results.n_valid_common.max(), e3_targeted_qc.rectification_fraction.max())))
    add(("eq", "Delta J_M = sum_(e in M) Delta J_e,  |M|=10\nDelta r_1 = D_J r[Delta J_M]\nDelta r_2 = Delta r_1 + 1/2 D_J^2 r[Delta J_M, Delta J_M]\nDelta r_exact = r(J + Delta J_M) - r(J)"))
    add(("p", "关键点是理论直接作用于完整的10边联合方向Delta J_M：二阶项包含不同被删边之间的cross terms，exact-resolvent也对联合扰动后的完整J重新求解；因此这里不是把10个单边Delta r简单相加。每个input-output组合的identity R2以40个joint mask为样本。"))
    add(("resulttable", ["input/lesion-degree/output", "先前分层", "先前5边p", "中位|实际Delta r|", "exact OLS R2", "OLS slope", "exact identity R2"], targeted_table))
    add(("p", f"效应量分层部分复现：合并三组后，prior-significant readout的中位|Delta r|为 {targeted_effect['previously significant']:.3g}，prior-null为 {targeted_effect['previously null']:.3g}。两个匹配组的差异尤其明显：A6Va/MST-in中为7.12e-3对1.73e-4，V2/A6Va-in中为2.00e-4对1.65e-6。PG/V1-out组的两个readout效应更接近，且null V2含少数peak-selection跳变造成的大离群值。"))
    add(("figure", exp3_targeted / "figures" / "exp3_targeted_degree10_scatter.png", "图 S8. 六个定向10边组合的完整扰动resolvent预测与完整55脑区非线性时域实际Delta r。红线为OLS，黑虚线为identity。仅PG输入/删V1出边/读V4具有很高且正向的mask排序，但其尺度仍偏大约4.36倍；另一些组合出现反向排序或峰值切换分支。"))
    add(("figure", exp3_targeted / "figures" / "exp3_targeted_degree10_summary.png", "图 S9. 联合一阶、二阶、exact-resolvent的identity R2及真实|Delta r|分布。六个组合的identity R2均小于0；橙色为此前显著readout，灰色为匹配null readout。"))
    add(("conclusion_keep", "定向10边结论：集中删边可放大部分敏感组合的实际Delta r，此前显著组通常也有更大的典型效应；但六个选例的identity R2均小于0，联合二阶没有系统优于一阶。效应显著或OLS排序较好，均不能替代绝对尺度校准。"))

    add(("h2", "4.3 输入seed与trial数的稳健性验证"))
    add(("p", "为检验Experiment 3是否主要受trial-by-trial输入噪声限制，新增两级真实时域验证。第一级固定V1输入、PG输出和40个held-out random-15删边mask，运行6个完全独立的input realization；每个realization同时更换输入参数抽样seed和Fourier phase seed，生成512个trial，并在同一trial池的N=32、64、128、256、512嵌套子样本上重算correlation。第二级使用一个512-trial稳健池依次驱动九个输入脑区，同时读取九个输出，对全部81条路径作相同的五档trial数比较。"))
    add(("p", "全部结果来自九脑区threshold-linear E-I网络的显式时域积分；dt=0.25 ms、burn-in=2.048 s，短窗4.096 s、长窗32.768 s。baseline与同一mask共享完全相同的输入trial，以common random numbers降低Delta r方差。每个input-output-window组合只使用baseline与40个mask共同peak-valid的trial，并在该子集上重新计算输入协方差Sigma_u及一阶、二阶结构理论。稳健9 x 9共积分9 x 41 x 512=188,928条长轨迹，每条轨迹同时读取九个输出；N=512时共同有效trial范围为%d-%d，中位数%d。" % (robust_valid_512.min(), robust_valid_512.max(), robust_valid_512.median())))
    add(("figure", exp3_robust / "figures" / "exp3_trial_seed_convergence.png", "图 S10. V1输入到PG输出的输入seed与trial数诊断。阴影表示6个独立realization的分布范围；下排量化actual Delta r的seed标准差与mask-wise重复一致性。"))
    add(("p", f"seed诊断显示两种不同情形。短窗的mask-wise seed一致性从N=64时的 {seed64short.mean_pairwise_seed_r:.3f} 提升到N=512时的 {seed512short.mean_pairwise_seed_r:.3f}，说明增加trial确实能稳定部分排序。长窗则从 {seed64long.mean_pairwise_seed_r:.3f} 仅变为 {seed512long.mean_pairwise_seed_r:.3f}；N=512时seed SD与跨mask信号SD之比仍为 {seed512long.noise_to_signal_ratio:.3f}。长窗Delta r本身约为10^-4至10^-3量级，trial误差与结构信号相当，因此不同seed仍可改变方向。"))
    add(("resulttable", ["稳健9 x 9中位数", "64 short", "512 short", "64 long", "512 long", "解释"], [
        ["一阶 OLS R2", _fmt(r64sf.ols_R2.median()), _fmt(r512sf.ols_R2.median()), _fmt(r64lf.ols_R2.median()), _fmt(r512lf.ols_R2.median()), "mask排序"],
        ["二阶 OLS R2", _fmt(r64ss.ols_R2.median()), _fmt(r512ss.ols_R2.median()), _fmt(r64ls.ols_R2.median()), _fmt(r512ls.ols_R2.median()), "mask排序"],
        ["一阶 identity R2", _fmt(r64sf.R2.median()), _fmt(r512sf.R2.median()), _fmt(r64lf.R2.median()), _fmt(r512lf.R2.median()), "绝对校准"],
        ["二阶 identity R2", _fmt(r64ss.R2.median()), _fmt(r512ss.R2.median()), _fmt(r64ls.R2.median()), _fmt(r512ls.R2.median()), "绝对校准"],
        ["二阶 identity R2 > 0", _fmt((r64ss.R2 > 0).mean()), _fmt((r512ss.R2 > 0).mean()), _fmt((r64ls.R2 > 0).mean()), _fmt((r512ls.R2 > 0).mean()), "81路径比例"],
    ]))
    add(("figure", exp3_robust / "figures" / "exp3_robust_9x9_trial_count_comparison.png", "图 S11. 81条输入输出路径的理论-真实模拟拟合优度随每次correlation使用的trial数变化。线为中位数，阴影为四分位范围。"))
    add(("p", f"系统结果没有显示identity校准随N单调恢复。二阶OLS R2中位数从64到512 trial在短窗由 {r64ss.ols_R2.median():.3f} 变为 {r512ss.ols_R2.median():.3f}，长窗由 {r64ls.ols_R2.median():.3f} 变为 {r512ls.ols_R2.median():.3f}，说明跨mask排序略有改善；但二阶identity R2短窗由 {r64ss.R2.median():.3f} 变为 {r512ss.R2.median():.3f}，长窗由 {r64ls.R2.median():.3f} 变为 {r512ls.R2.median():.3f}。N=512时identity R2>0的路径仅占短窗 {(r512ss.R2 > 0).mean():.1%}、长窗 {(r512ls.R2 > 0).mean():.1%}。"))
    add(("figure", exp3_robust / "figures" / "exp3_robust_9x9_identity_R2.png", "图 S12. 512 trial稳健真实模拟中，Experiment 3的9 x 9短窗/长窗、一阶/二阶identity R2。多数路径仍未达到绝对校准。"))
    add(("conclusion", "Experiment 3 seed验证结论：trial-by-trial输入噪声确实是重要误差源，尤其当真实结构Delta r很小时；增加到512 trial可提高部分mask排序的重复性和OLS R2。然而8倍trial数没有改善全系统identity R2，部分条件反而因抽样误差缩小而更清楚地暴露反号和尺度失配。因此原结果不好不能主要归因于trial数太少。剩余偏差更符合结构Taylor理论尚未包含有限窗PSD feature、peak选择非连续性、correlation分母随结构变化及连续resolvent与离散时域readout差异。稳健程序应至少使用数百trial、common random numbers并报告跨seed重复，但增加trial本身不能替代理论校准。"))

    add(("h2", "4.4 Euler步长收敛：0.25 ms与0.05 ms"))
    add(("p", "为检验连续resolvent与Euler离散积分的差异，针对4.2节完全相同的三组input-lesion条件、40个10边mask和512个输入trial，分别以dt=0.25 ms和0.05 ms重算完整55脑区时域模拟。两个步长都使用严格低于2000 Hz的相同Fourier系数；每个trial独立定相位，0.05 ms输入每5点下采样后与0.25 ms输入的最大误差为3.2e-12。Welch段长仍为2.048 s，分析窗仍为4.096 s，因此频率分辨率没有改变。"))
    add(("p", "相关性只在baseline、全部40个mask及两个步长共同peak-valid的trial上计算，有效trial为%d-%d；两个步长的rectification fraction均为0，说明比较发生在线性工作区。横轴理论保持同一个连续时间joint-10-edge exact-resolvent，因此本实验只测试减小Euler误差是否让真实Delta r靠近理论。" % (e3_dt_results.n_valid_common_across_dt.min(), e3_dt_results.n_valid_common_across_dt.max())))
    add(("resulttable", ["input/lesion-degree/output", "identity R2 0.25", "identity R2 0.05", "OLS R2 0.25", "OLS R2 0.05", "actual跨dt Pearson r"], dt_table))
    add(("p", f"细步长使6个组合中的{dt_improved}/6个exact identity R2提高，中位数由 {dt25.R2.median():.3f} 改善到 {dt05.R2.median():.3f}；OLS R2中位数由 {dt25.ols_R2.median():.3f} 变为 {dt05.ols_R2.median():.3f}。改善最明显的是PG输入/删V1出边/读V4：identity R2由 {dt25_pg_v4.R2:.3f} 变为 {dt05_pg_v4.R2:.3f}，OLS R2由 {dt25_pg_v4.ols_R2:.3f} 变为 {dt05_pg_v4.ols_R2:.3f}，slope由 {dt25_pg_v4.slope:.1f} 恢复到 {dt05_pg_v4.slope:.3f}。"))
    add(("p", f"但步长收敛并不普遍。PG/V4的实际Delta r跨步长Pearson r={dt_pg_v4_agreement.pearson_r:.3f}，原因是dt=0.25 ms下一个mask发生约-0.038的峰值切换，而dt=0.05 ms没有该跳变。V2输入/删A6Va入边/读A6Va和PG输入/删V1出边/读V2的跨步长r分别为 {e3_dt_agreement[(e3_dt_agreement.group=='V2_A6Va_in') & (e3_dt_agreement.output_area=='A6Va')].pearson_r.iloc[0]:.3f} 和 {e3_dt_agreement[(e3_dt_agreement.group=='PG_V1_out') & (e3_dt_agreement.output_area=='V2')].pearson_r.iloc[0]:.3f}，而两个小效应null readout接近不复现。"))
    add(("figure", exp3_dt / "figures" / "exp3_dt_convergence_scatter.png", "图 S13. 两个Euler步长下，joint-10-edge exact-resolvent与完整55脑区时域Delta r。红色为0.05 ms，蓝色为0.25 ms；黑虚线为identity。细步长修复了部分粗步长特有的峰值跳变，但MST和PG/V2中的大分支仍存在。"))
    add(("figure", exp3_dt / "figures" / "exp3_dt_convergence_summary.png", "图 S14. 六个组合的identity R2、OLS R2及实际Delta r跨步长复现性。细步长总体改善identity校准，但不同路径表现不一致，且两个小效应null输出的跨步长相关性很低。"))
    add(("conclusion_keep", "步长收敛结论：dt=0.05 ms能够消除部分Euler离散引起的高频PSD和peak-selection误差，说明原Experiment 3确有数值步长贡献；但中位identity R2仍略低于0，且若干组合在两个步长间不能稳定复现。因而步长误差是重要因素之一，却不足以解释全部理论-实际偏差。下一步仍需使用离散时间resolvent并显式纳入Welch/peak观测层。"))

    add(("pagebreak",))
    add(("h2", "4.5 每个删边网络内重做Experiment 1：原始相关性与做差诊断"))
    add(("p", "为区分两种解释，针对图S13的全部六个input-lesion-output点，在baseline及每个联合10边删边网络内分别完整重做Experiment 1。三组结构扰动为A6Va输入/删MST入边、V2输入/删A6Va入边、PG输入/删V1出边；每组40个独立10边mask，并各读取一个此前显著和一个匹配null输出，共6条路径。每个网络使用14个输入condition，每condition 128个trial，并运行3个独立Fourier phase realization。短窗4.096 s、Welch段2.048 s；长窗32.768 s、Welch段4.096 s；短窗是同一条长轨迹的前缀。"))
    add(("p", "真实输出由完整55脑区网络的离散时间Euler递推生成。由于本批数据rectification为0，程序使用该线性工作区递推的精确离散传递函数加速批量计算；它不是连续时间resolvent直接代替simulation。与逐步显式递推的独立验证得到输出时序r=%.6f、peak频率最大绝对误差=%.2g、PSD exponent最大绝对误差=%.2g。每个网络和输出只在baseline与全部40个mask共同peak-valid的trial上计算P-A correlation，有效trial范围为%d-%d。" % (e3_absolute_manifest["explicit_validation_output_r"], e3_absolute_manifest["explicit_validation_peak_omega_max_abs_error"], e3_absolute_manifest["explicit_validation_exponent_max_abs_error"], e3_maskwise_conditions.n_valid_common.min(), e3_maskwise_conditions.n_valid_common.max())))
    add(("p", "横轴理论分别采用输入feature map的一阶Taylor、二阶Taylor及exact ideal-spectrum mapping；纵轴始终是对应网络真实时域输出在14个condition中的P-A correlation。这里的identity R2在condition轴上计算，因而直接检验每一个baseline或删边网络内Experiment 1本身是否成立，而不是先做baseline与perturbed的差。"))
    add(("resulttable", ["每网络Experiment 1", "short first", "short second", "long first", "long second", "统计对象"], [
        ["baseline identity R2中位数", _fmt(maskwise_summary[("baseline", "short", "first")].R2.median()), _fmt(maskwise_summary[("baseline", "short", "second")].R2.median()), _fmt(maskwise_summary[("baseline", "long", "first")].R2.median()), _fmt(maskwise_summary[("baseline", "long", "second")].R2.median()), "6路径 x 3 phase"],
        ["10边网络identity R2中位数", _fmt(maskwise_summary[("10-edge deletion", "short", "first")].R2.median()), _fmt(maskwise_summary[("10-edge deletion", "short", "second")].R2.median()), _fmt(maskwise_summary[("10-edge deletion", "long", "first")].R2.median()), _fmt(maskwise_summary[("10-edge deletion", "long", "second")].R2.median()), "6 x 40 x 3网络"],
        ["10边网络identity R2范围", f"{maskwise_summary[('10-edge deletion', 'short', 'first')].R2.min():.2f} to {maskwise_summary[('10-edge deletion', 'short', 'first')].R2.max():.2f}", f"{maskwise_summary[('10-edge deletion', 'short', 'second')].R2.min():.2f} to {maskwise_summary[('10-edge deletion', 'short', 'second')].R2.max():.2f}", f"{maskwise_summary[('10-edge deletion', 'long', 'first')].R2.min():.3f} to {maskwise_summary[('10-edge deletion', 'long', 'first')].R2.max():.3f}", f"{maskwise_summary[('10-edge deletion', 'long', 'second')].R2.min():.3f} to {maskwise_summary[('10-edge deletion', 'long', 'second')].R2.max():.3f}", "全部network-phase"],
        ["10边网络OLS slope中位数", _fmt(maskwise_summary[("10-edge deletion", "short", "first")].slope.median()), _fmt(maskwise_summary[("10-edge deletion", "short", "second")].slope.median()), _fmt(maskwise_summary[("10-edge deletion", "long", "first")].slope.median()), _fmt(maskwise_summary[("10-edge deletion", "long", "second")].slope.median()), "理想值1"],
        ["10边网络OLS R2中位数", _fmt(maskwise_summary[("10-edge deletion", "short", "first")].ols_R2.median()), _fmt(maskwise_summary[("10-edge deletion", "short", "second")].ols_R2.median()), _fmt(maskwise_summary[("10-edge deletion", "long", "first")].ols_R2.median()), _fmt(maskwise_summary[("10-edge deletion", "long", "second")].ols_R2.median()), "condition排序"],
    ]))
    add(("p", "短窗下，baseline与全部720个删边network-phase的一阶/二阶identity R2均小于0，且两类网络的中位数都约为-7.27；因此短窗结果首先是Experiment 1原始correlation本身的严重attenuation，不是删边使非baseline网络额外失效。长窗下，baseline的一阶/二阶中位数为%.3f/%.3f，720个删边network-phase为%.3f/%.3f，且删边结果全部高于%.3f。删除10条边并没有降低长窗原始correlation理论的校准。" % (maskwise_summary[("baseline", "long", "first")].R2.median(), maskwise_summary[("baseline", "long", "second")].R2.median(), maskwise_summary[("10-edge deletion", "long", "first")].R2.median(), maskwise_summary[("10-edge deletion", "long", "second")].R2.median(), min(maskwise_summary[("10-edge deletion", "long", "first")].R2.min(), maskwise_summary[("10-edge deletion", "long", "second")].R2.min()))))
    add(("figure", exp3_maskwise_exp1 / "figures" / "maskwise_exp1_identity_R2.png", "图 S15. 图S13全部六个组合中，baseline菱形与40个联合10边删边网络的Experiment 1 identity R2。上排为短窗，下排为长窗；列为一阶、二阶和exact ideal-spectrum mapping。红线是删边网络中位数。"))
    add(("figure", exp3_maskwise_exp1 / "figures" / "representative_exp1_scatter_short_first.png", "图 S16. 每个组合选取一个中位表现删边网络，展示短窗一阶理论correlation与真实55脑区simulation correlation。每点为一个输入condition；严重幅度压缩导致identity R2为负。"))
    add(("figure", exp3_maskwise_exp1 / "figures" / "representative_exp1_scatter_short_second.png", "图 S17. 与图S16相同，但使用短窗二阶输入Taylor。二阶未消除有限记录造成的correlation attenuation。"))
    add(("figure", exp3_maskwise_exp1 / "figures" / "representative_exp1_scatter_long_first.png", "图 S18. 长窗一阶代表网络。14个condition的理论与simulation接近identity，说明结构扰动后Experiment 1本身仍然成立。"))
    add(("figure", exp3_maskwise_exp1 / "figures" / "representative_exp1_scatter_long_second.png", "图 S19. 长窗二阶代表网络。结果与一阶几乎重合，进一步排除输入Taylor阶数不足是主要误差源。"))
    add(("p", "跨3个独立phase realization，长窗删边网络的mask-wise identity R2排序具有很强重复性：一阶/二阶在六个组合中的pairwise repeat r中位数总体范围为%.3f-%.3f。与此同时，配对做差并没有增加绝对随机相位方差。跨240个mask-output点，SD(Delta r)/SD(raw perturbed r)的中位数在短窗为%.4f、长窗为%.4f，分别相当于约%.0f倍和%.0f倍的绝对方差缩减。" % (maskwise_phase_long.min(), maskwise_phase_long.max(), absolute_variance_ratio_by_window["short"], absolute_variance_ratio_by_window["long"], 1 / absolute_variance_ratio_by_window["short"], 1 / absolute_variance_ratio_by_window["long"])))
    add(("figure", exp3_absolute_phase / "figures" / "phase_repeat_differencing_variance.png", "图 S20. 三个独立Fourier phase realization中，配对Delta r与raw perturbed r的跨相位标准差之比。所有组合均小于1，说明common-random-number做差降低了绝对realization方差。"))
    add(("p", "关键在于相对信噪比而非绝对方差：以每个组合的mask间exact理论Delta r标准差作为结构信号，残余phase noise与理论信号之比在短窗六个组合中的中位数为%.2f（范围%.2f-%.2f），长窗仍为%.2f（范围%.2f-%.2f）。所以做差虽然消除了大部分共同波动，却把两个接近的相关系数压缩到约10^-3甚至更小的结构效应尺度；剩余PSD feature偏差、peak switching及correlation分母变化便足以主导identity R2。" % (phase_noise_to_theory[phase_noise_to_theory.window == "short"].ratio.median(), phase_noise_to_theory[phase_noise_to_theory.window == "short"].ratio.min(), phase_noise_to_theory[phase_noise_to_theory.window == "short"].ratio.max(), phase_noise_to_theory[phase_noise_to_theory.window == "long"].ratio.median(), phase_noise_to_theory[phase_noise_to_theory.window == "long"].ratio.min(), phase_noise_to_theory[phase_noise_to_theory.window == "long"].ratio.max())))
    add(("conclusion_keep", "原始相关性与做差诊断结论：短窗主要支持解释（1）- baseline和perturbed网络中的Experiment 1都因有限窗而失配。长窗则排除了“删边网络内Experiment 1失效”：所有删边网络的原始correlation都接近identity。长窗Delta r仍不稳健的主要原因是两个已校准但非常接近的相关系数相减后，目标结构信号过小，残余observation-layer误差相对于Delta r被放大。配对做差降低绝对方差，却不能保证差值具有足够的相对信噪比。"))

    add(("h2", "4.6 Blind筛选较大Delta correlation的10边模式"))
    add(("p", "为避免用simulation结果挑选删边模式，新增完全理论盲筛。结构底座为完整55脑区网络，输入和输出只限制在V1、V2、V4、MST、LIP、PG、A23a、A6DR和A6Va。每个mask一次删除10条非零FLN边；候选包括uniform、按w采样、按w^2采样和集中于某脑区in/out-degree四类，每类90个，共360个mask。对每个mask计算完整J+Delta J的exact resolvent并遍历9 x 9输入输出，因此筛选总计29,160个mask-input-output条件。结构扰动理论没有使用一阶或二阶展开。"))
    add(("p", "按理论|Delta r|预注册36个高效应条件，并从低效应区匹配18个对照；选择完成后才运行时域数据。每个条件使用256个trial、32.768 s分析窗、2.048 s burn-in、4.096 s Hann-Welch段和3个独立phase realization，baseline与perturbed共享同一个trial输入。时域求解采用Euler递推的精确离散传递函数；另对最大效应的3个条件、baseline和perturbed各8个长窗trial运行显式55脑区threshold-linear recurrence。显式与加速输出r至少为%.6f，peak omega最大误差%.2g、exponent最大误差%.2g，且rectification总数为0。" % (e3_large_manifest["min_explicit_output_r"], e3_large_manifest["max_explicit_peak_omega_error"], e3_large_manifest["max_explicit_exponent_error"])))
    add(("figure", exp3_large_delta / "figures" / "large_delta_screen_9x9.png", "图 S24. 360个盲筛mask中，每条9 x 9输入输出路径的最大exact-resolvent |Delta r|。颜色表示该路径能够找到的最大理论结构效应。"))
    add(("resulttable", ["盲筛长窗验证", "高效应36条件", "低效应18对照", "解释"], [
        ["筛选阶段最大|Delta r|", f"{e3_large_selected[e3_large_selected.selection_class == 'high_effect'].abs_exact_delta_r.max():.4g}", f"{e3_large_selected[e3_large_selected.selection_class == 'matched_low_effect'].abs_exact_delta_r.max():.4g}", "exact resolvent盲筛"],
        ["长窗exact |Delta r|中位数", f"{large_high.theory_delta_r_long_design.abs().median():.4g}", f"{large_control.theory_delta_r_long_design.abs().median():.4g}", "3 phase理论均值"],
        ["长窗simulation |Delta r|中位数", f"{large_high.simulation_delta_r.abs().median():.4g}", f"{large_control.simulation_delta_r.abs().median():.4g}", "3 phase模拟均值"],
        ["phase-mean OLS R2", _fmt(large_high_metric.pearson_r ** 2), _fmt(large_control_metric.pearson_r ** 2), "mask排序"],
        ["phase-mean identity R2", _fmt(large_high_metric.R2), _fmt(large_control_metric.R2), "绝对校准"],
        ["phase-mean slope", _fmt(large_high_metric.slope), _fmt(large_control_metric.slope), "理想值1"],
    ]))
    add(("figure", exp3_large_delta / "figures" / "large_delta_long_validation.png", "图 S25. blind高效应与匹配低效应mask的exact-resolvent Delta r和长窗真实时域Delta r。左：phase均值散点与identity线；右：两类条件的|Delta r|中位数。"))
    add(("figure", exp3_large_delta / "figures" / "large_delta_phase_examples.png", "图 S26. 六个最大理论效应条件在3个独立phase realization中的Delta r。每个小图的三点对应三次独立实现。"))
    add(("p", "盲筛确实提高了目标结构信号：梯度筛选尺度下最大|Delta r|为%.4g，高效应组在实际长窗设计上的exact |Delta r|中位数为%.4g，远高于低效应对照的%.4g；simulation |Delta r|中位数也从%.4g提高到%.4g。然而高效应组phase-mean OLS R2仅%.3f、identity R2=%.3f、slope=%.3f，且独立phase之间的mask排序仍不稳定。也就是说，删除10条强边能把信号从单边约10^-3提高到数10^-3，但还没有把结构差值推到足以压过有限窗peak/slope误差的范围。" % (e3_large_screen.abs_exact_delta_r.max(), large_high.theory_delta_r_long_design.abs().median(), large_control.theory_delta_r_long_design.abs().median(), large_control.simulation_delta_r.abs().median(), large_high.simulation_delta_r.abs().median(), large_high_metric.pearson_r ** 2, large_high_metric.R2, large_high_metric.slope)))
    add(("conclusion_keep", "大Delta盲筛结论：exact-resolvent筛选可以富集理论结构效应，却没有自动恢复真实Delta correlation的identity校准。这个负结果进一步支持Experiment 3的核心诊断：问题不主要是使用了一阶/二阶结构展开，也不只是随机选到了无效边；即使用完整J+Delta J和长窗，Delta r仍是两个接近相关系数之差，其相对观测误差足以主导结果。后续若要检验结构机制，应预注册更强、具有路由梯度对齐的多边perturbation，并把phase ensemble或trial数提高到能直接稳定Delta r的水平。"))

    add(("h2", "4.7 定向删除重要边：找到稳健的exact-theory/simulation一致案例"))
    add(("p", "随机10边未产生足够信号后，改用理论驱动的重要边搜索。首先在完整55脑区网络中预注册6条高敏感input-output路径，并构造763条候选边：包括全网络FLN权重最大的500条边，以及与V1、V2、V4、PG和A6Va相邻的全部边。对每条边单独删除后计算完整J+Delta J的exact resolvent，不读取任何时域simulation；随后分别按正向和负向single-edge Delta r排序，形成10、15、20、30、40、50、75和100边的累积组合。"))
    add(("p", "理论discovery使用512个输入参数trial，独立理论holdout使用2048个trial，并要求至少80% trial在baseline和perturbed网络中共同peak-valid。由理论产生finalists后，3个pilot phase只用于排除观测极不稳定的组合；最终冻结为A6DR输入、V2输出、删除50条重要边。pilot之后的结果完全来自新生成的holdout数据。"))
    add(("p", "为了估计与exact-resolvent相对应的期望谱特征，不再把单个有限窗phase当作无噪声feature。固定256个输入参数trial；每个trial运行24个独立32.768 s长窗时域realization，逐trial平均peak和aperiodic exponent，再计算P-A correlation。该过程重复10个完全独立holdout ensemble，共61,440条baseline轨迹和61,440条删边轨迹；全部ensemble均未参与删边选择。"))
    add(("p", f"此前图中exact resolvent在不同ensemble之间有轻微变化，并不是网络理论随trial改变，而是旧程序分别使用每个ensemble自身的peak-valid trial子集计算相关性。相关性是对样本集合定义的条件统计量，子集变化会令同一理论feature map得到不同r。本次已改为预先冻结一个共同有效集合：trial必须在理论端有效，且在每个24-phase holdout中、baseline与删边网络都至少18次可测；最终固定{int(e3_important_ensembles.n_valid_common.iloc[0])}个trial。因而每个holdout的exact baseline、perturbed及Delta r严格相同，只有真实simulation随独立phase变化。"))
    add(("resulttable", ["50条重要边最终验证", "exact resolvent", "长窗phase-ensemble simulation", "说明"], [
        ["删边前correlation", f"{e3_important_summary.theory_baseline_r_mean:.6f}", f"{e3_important_summary.simulation_baseline_r_mean:.6f} +/- {e3_important_summary.simulation_baseline_r_sd:.6f}", "Experiment 1 baseline"],
        ["删边后correlation", f"{e3_important_summary.theory_perturbed_r_mean:.6f}", f"{e3_important_summary.simulation_perturbed_r_mean:.6f} +/- {e3_important_summary.simulation_perturbed_r_sd:.6f}", "完整J+Delta J"],
        ["Delta correlation", f"{e3_important_summary.theory_delta_r_mean:.6f}", f"{e3_important_summary.simulation_delta_r_mean:.6f} +/- {e3_important_summary.simulation_delta_r_sd:.6f}", "perturbed minus baseline"],
        ["Delta r 95% CI", "theory为点预测", f"[{e3_important_summary.simulation_delta_r_ci_low:.6f}, {e3_important_summary.simulation_delta_r_ci_high:.6f}]", f"p={e3_important_summary.simulation_delta_r_one_sample_p:.4f}"],
        ["方向稳定性", "正", f"{e3_important_summary.sign_consistency:.0%}", "10/10独立ensemble同号"],
        ["|theory-simulation Delta|", "-", f"{e3_important_summary.absolute_delta_error:.6f}", "固定集合上的幅度误差"],
    ]))
    add(("figure", exp3_important / "figures" / "important_edge_phase_ensemble_validation.png", "图 S27. 定向50边删边的稳健验证。左上列出权重最大的25条被删边；右上比较删边前后exact与simulation correlation；左下显示10个独立ensemble的Delta r及固定exact理论；右下为代表性ensemble的trial-level P-A散点。"))
    add(("figure", exp3_important / "figures" / "selected_deleted_edges_complete.png", "图 S28. 最终删除的全部50条边及其原始FLN权重。组合由single-edge exact-resolvent敏感度排序产生，不是随机删边。"))
    add(("p", "最终10个holdout ensemble的simulation Delta r依次为%s，全部为正；均值%.6f，95%% CI [%.6f, %.6f]，双侧one-sample t检验p=%.4g。固定exact-resolvent预测为%.6f，低于simulation 95%% CI下界；simulation与理论的绝对差为%.6f。删边前后绝对correlation仍接近：theory为%.3f到%.3f，simulation为%.3f到%.3f。增加holdout并冻结样本集合后，方向证据更强，但也更清楚地暴露了exact理论对Delta r幅度的低估。" % (", ".join(f"{value:.6f}" for value in e3_important_ensembles.simulation_delta_r), e3_important_summary.simulation_delta_r_mean, e3_important_summary.simulation_delta_r_ci_low, e3_important_summary.simulation_delta_r_ci_high, e3_important_summary.simulation_delta_r_one_sample_p, e3_important_summary.theory_delta_r_mean, e3_important_summary.absolute_delta_error, e3_important_summary.theory_baseline_r_mean, e3_important_summary.theory_perturbed_r_mean, e3_important_summary.simulation_baseline_r_mean, e3_important_summary.simulation_perturbed_r_mean)))
    add(("conclusion_keep", "重要边定向删边结论：可以找到exact-resolvent与真实长窗simulation方向稳定一致、数量级相近的特定结构组合。本例删除50条按理论敏感度排序的重要边；10个独立24-phase holdout全部得到正Delta r。冻结共同trial集合后exact预测不再随ensemble变化，但其幅度低于simulation 95% CI，故证据支持机制方向而非严格identity校准。该结果也说明旧图中变化的理论线来自样本筛选，而不是resolvent本身具有随机性。"))

    add(("h2", "4.8 68边路由删边：两个Experiment 1的二阶验证"))
    add(("p", f"进一步验证此前找到的emergent-correlation regime：输入A6Va、读取V2，删除V2除A6Va以外的全部入边，并删除A6Va除V2以外的全部出边，只保留A6Va->V2，共删除{e3_emergent_manifest['n_deleted_edges']}条非零FLN边。输入四参数为bump中心、bump幅度、beta和背景幅度；4096个trial来自预冻结的相关Gaussian设计，epsilon=0.05。simulation使用32.768 s分析窗、2.048 s burn-in、0.25 ms步长、4.096 s Hann-Welch段和6个独立phase。"))
    add(("p", f"主分析冻结{e3_emergent_manifest['n_frozen_trials']}个共同trial：连续理论有效，且baseline/deleted网络各至少75% phase可测。每个trial先平均其5-6个有效长窗phase。对baseline与deleted两个网络分别在冻结trial经验中心计算Experiment 1一阶和二阶input Taylor feature map，再各自计算r，最后取Delta r=r_deleted-r_baseline；exact、一阶、二阶与simulation严格使用同一集合。"))
    add(("eq", "P2(u)=P0+grad(P)'x+1/2 x'Hess(P)x\nA2(u)=A0+grad(A)'x+1/2 x'Hess(A)x\nDelta r2=Corr[P2_del,A2_del]-Corr[P2_base,A2_base]"))
    add(("pagebreak",))
    add(("resulttable", ["68边验证", "baseline r", "deleted r", "Delta r", "|Delta-simulation|"], [
        ["first-order Experiment 1", f"{e3_emergent_correlations.loc[('first_order','baseline'),'r']:.6f}", f"{e3_emergent_correlations.loc[('first_order','deleted'),'r']:.6f}", f"{e3_emergent_correlations.loc[('first_order','deleted_minus_baseline'),'r']:.6f}", f"{abs(e3_emergent_correlations.loc[('first_order','deleted_minus_baseline'),'r']-e3_emergent_correlations.loc[('simulation','deleted_minus_baseline'),'r']):.6f}"],
        ["second-order Experiment 1", f"{e3_emergent_correlations.loc[('second_order','baseline'),'r']:.6f}", f"{e3_emergent_correlations.loc[('second_order','deleted'),'r']:.6f}", f"{e3_emergent_correlations.loc[('second_order','deleted_minus_baseline'),'r']:.6f}", f"{abs(e3_emergent_correlations.loc[('second_order','deleted_minus_baseline'),'r']-e3_emergent_correlations.loc[('simulation','deleted_minus_baseline'),'r']):.6f}"],
        ["exact continuous resolvent", f"{e3_emergent_correlations.loc[('exact_resolvent','baseline'),'r']:.6f}", f"{e3_emergent_correlations.loc[('exact_resolvent','deleted'),'r']:.6f}", f"{e3_emergent_correlations.loc[('exact_resolvent','deleted_minus_baseline'),'r']:.6f}", f"{abs(e3_emergent_correlations.loc[('exact_resolvent','deleted_minus_baseline'),'r']-e3_emergent_correlations.loc[('simulation','deleted_minus_baseline'),'r']):.6f}"],
        ["6-phase long-window simulation", f"{e3_emergent_correlations.loc[('simulation','baseline'),'r']:.6f}", f"{e3_emergent_correlations.loc[('simulation','deleted'),'r']:.6f}", f"{e3_emergent_correlations.loc[('simulation','deleted_minus_baseline'),'r']:.6f}", "-"],
    ]))
    add(("p", f"真实simulation中baseline r={e3_emergent_correlations.loc[('simulation','baseline'),'r']:.4f}、p={e3_emergent_correlations.loc[('simulation','baseline'),'p']:.3g}，不显著；68边删边后r={e3_emergent_correlations.loc[('simulation','deleted'),'r']:.4f}、p={e3_emergent_correlations.loc[('simulation','deleted'),'p']:.3g}，显著，因而实现了原网络无明确correlation、路由删边后出现correlation。simulation Delta r={e3_emergent_correlations.loc[('simulation','deleted_minus_baseline'),'r']:.5f}；条件于这6个phase的paired-trial bootstrap 95% CI为[{e3_emergent_correlations.loc[('simulation','deleted_minus_baseline'),'bootstrap_ci_low']:.5f}, {e3_emergent_correlations.loc[('simulation','deleted_minus_baseline'),'bootstrap_ci_high']:.5f}]，包含一阶、二阶和exact点预测。"))
    add(("p", f"二阶确实改善单trial feature：baseline peak RMSE相对exact从{e3_emergent_features.loc[('baseline','first_order'),'peak_rmse_hz_vs_exact']:.5f}降到{e3_emergent_features.loc[('baseline','second_order'),'peak_rmse_hz_vs_exact']:.5f} Hz，deleted从{e3_emergent_features.loc[('deleted','first_order'),'peak_rmse_hz_vs_exact']:.5f}降到{e3_emergent_features.loc[('deleted','second_order'),'peak_rmse_hz_vs_exact']:.6f} Hz。但二阶Delta r误差略高于一阶，exact误差最小；三者差异很小。说明Taylor截断已经不是瓶颈，raw baseline/deleted correlation约0.10的共同attenuation主要来自finite-window observation layer，做差消除了大部分共同误差。"))
    add(("figure", exp3_emergent_second / "figures" / "exp3_68edge_second_order_validation.png", "图 S38. 将68边Experiment 3写成baseline/deleted两个Experiment 1：比较一阶、二阶、exact及六相位simulation的raw correlation与Delta r。"))
    add(("figure", exp3_emergent_second / "figures" / "exp3_68edge_trial_maps.png", "图 S39. 同一3224-trial集合上的baseline/deleted exact、二阶Taylor和六相位长窗P-A平面。"))
    add(("figure", exp3_emergent_second / "figures" / "exp3_68edge_phase_repeats.png", "图 S40. 三个独立两相位ensemble的Delta r realization检查；simulation均为正，一阶/二阶/exact稳定在约0.042-0.044。"))
    add(("conclusion_keep", "68边二阶验证结论：分别对两个网络做Experiment 1再相减是正确计算路径。真实长窗simulation复现了baseline不显著、删边后显著，且Delta r与exact/一阶/二阶同方向、同数量级。二阶显著提高feature map对exact的局部逼近，却没有改善最终Delta r；剩余差异来自有限窗PSD feature observation layer，而非输入Taylor阶数不足。"))

    add(("h1", "五、主要结论"))
    add(("bullet", "Experiment 1：有限记录可严重压缩P-A correlation。完整55脑区9 x 9结果中，长窗几乎恢复全部路径的理论排序和identity校准。"))
    add(("bullet", f"Experiment 2：两阶段扫描将机制分开。I-to-E在kappa_I={complex_kappa:.4f}产生复根但保持强稳定；固定该抑制平衡后，E-to-E在g_EE={hopf_gain:.4f}才推动Hopf。复模态、谱峰锐化和自持振荡不是同一事件。"))
    add(("bullet", "Experiment 2新推导：稳定I-to-E路径的f_lambda^2平方根律得到R2=1；Sc=2.0816、Sres=3.1774且无临界减速。稳定滤波器可通过input beta同时改变P和A，因此无需Hopf也能产生P-A coupling。"))
    add(("bullet", "Experiment 2的Eq.134限制仍然存在：预注册100-1000 Hz exponent的kappa_R在94.4% condition中置信区间包含0，因此临界动力学成立不等于该aperiodic readout能观察到共同振幅耦合。"))
    add(("bullet", f"Experiment 3：将Euler步长从0.25缩小到0.05 ms后，6个定向10边组合中{dt_improved}个identity R2提高，中位数由{dt25.R2.median():.3f}变为{dt05.R2.median():.3f}；数值离散是偏差来源之一，但仍不足以恢复普遍identity校准。"))
    add(("bullet", "Experiment 3稳健性验证：512 trial提高了部分seed重复性和OLS排序，但9 x 9二阶identity R2仍为短窗-0.253、长窗-0.500；trial噪声是贡献因素而非充分解释。"))
    add(("bullet", "Experiment 3原始相关性诊断：长窗在baseline和全部720个删边network-phase中都把每网络Experiment 1 correlation identity R2恢复到约0.99；因此长窗Delta r失配不是非baseline网络的输入理论失效，而是微小结构差值相对于残余观测误差的条件数问题。"))
    add(("bullet", "Experiment 3 blind大效应筛选：360个exact-resolvent候选覆盖29,160个输入输出条件，最大理论|Delta r|约0.0038。高效应富集成功，但256 trial x 3 phase的长窗真实模拟仍未恢复identity校准，说明随机小幅多边删边的结构信号依旧偏小。"))
    add(("bullet", f"Experiment 3重要边案例：A6DR输入、V2输出，删除50条理论重要边。10个独立24-phase holdout全部得到正Delta r；simulation均值{e3_important_summary.simulation_delta_r_mean:.5f}、95% CI [{e3_important_summary.simulation_delta_r_ci_low:.5f}, {e3_important_summary.simulation_delta_r_ci_high:.5f}]，固定exact预测{e3_important_summary.theory_delta_r_mean:.5f}，方向一致但幅度偏低。"))
    add(("bullet", f"Experiment 3的68边路由案例：A6Va输入、V2输出。baseline simulation r={e3_emergent_correlations.loc[('simulation','baseline'),'r']:.3f}不显著，删边后r={e3_emergent_correlations.loc[('simulation','deleted'),'r']:.3f}显著；simulation Delta r={e3_emergent_correlations.loc[('simulation','deleted_minus_baseline'),'r']:.3f}，exact/一阶/二阶约0.043。二阶改善feature精度但不改善Delta r，误差瓶颈位于有限窗观测层。"))
    add(("bullet", "跨实验方法学结论：必须同时报告OLS R2、slope、identity R2、有效trial数和随机相位重复。只看相关系数平方可能掩盖反号或幅度失配。"))

    md_path = report_dir / "three_experiment_summary_zh.md"
    pdf_path = pdf_dir / "three_experiment_summary_zh.pdf"
    _write_markdown(sections, md_path)
    _write_pdf(sections, pdf_path)
    return md_path, pdf_path


if __name__ == "__main__":
    print("\n".join(map(str, build())))
