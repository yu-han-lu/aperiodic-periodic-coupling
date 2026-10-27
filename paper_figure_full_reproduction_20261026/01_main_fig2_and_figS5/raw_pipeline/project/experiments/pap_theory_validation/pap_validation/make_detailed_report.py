from __future__ import annotations

import argparse
import json
import shutil
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt(x, digits=4):
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{digits}g}"


def _compact_condition(value: str) -> str:
    value = str(value)
    if value.startswith("SL_"):
        return value.removeprefix("SL_").replace("_", ", ")
    if value.startswith("EI_linear_"):
        return "linear: " + value.removeprefix("EI_linear_").replace("_", ", ")
    if value.startswith("EI_"):
        return value.removeprefix("EI_").replace("_", ", ")
    return value


def build(output_root: Path, report_dir: Path, pdf_dir: Path, basename: str = "detailed_experiment_summary_zh_finite_window"):
    e1 = output_root / "exp1_input_sampling"
    e1p = output_root / "exp1_pairwise_io"
    e1n = output_root / "exp1_nonlinear_full55"
    e2 = output_root / "exp2_nonlinear_hopf"
    e3 = output_root / "exp3_structural_kernel"
    e3n = output_root / "exp3_nonlinear_subnetwork"
    fw = output_root / "finite_window_validation"
    e1pw = output_root / "finite_window_exp1_pairwise_nonlinear"
    e3pw = output_root / "finite_window_exp3_pairwise_nonlinear"
    e3single = output_root / "exp3_single_phase_12c_comparison"
    e3single9 = output_root / "finite_window_exp3_pairwise_single_phase_12c_exact"
    e3phase = output_root / "finite_window_exp3_pairwise_phase_ensemble"
    e3seed = output_root / "exp3_v1pg_phase_seed_validation"
    m1, m1p, m1n, m2, m3, m3n = (_json(path / "metrics.json") for path in (e1, e1p, e1n, e2, e3, e3n))
    s1 = pd.read_csv(e1 / "condition_summary.csv")
    t1 = pd.read_csv(e1 / "trials.csv.gz", low_memory=False)
    p1 = pd.read_csv(e1 / "theory_predictions.csv")
    dv = pd.read_csv(e1 / "derivative_validation.csv")
    s1n = pd.read_csv(e1n / "condition_summary.csv")
    d2 = pd.read_csv(e2 / "radial_moments.csv")
    w2 = pd.read_csv(e2 / "trials.csv.gz")
    b2 = pd.read_csv(e2 / "ei_bifurcation.csv")
    p2 = pd.read_csv(e2 / "theory_predictions.csv")
    r2c = pd.read_csv(e2 / "condition_regression_stats.csv")
    t3 = pd.read_csv(e3 / "trials.csv.gz")
    gb = pd.read_csv(e3 / "graph_baselines.csv").set_index("predictor")
    s3n = pd.read_csv(e3n / "delta_correlation_results.csv")
    fw_summary = pd.read_csv(fw / "summary_metrics.csv")
    e1pw_cov = pd.read_csv(e1pw / "covariance_R2_by_pair.csv")
    e1pw_cor = pd.read_csv(e1pw / "correlation_R2_by_pair.csv")
    e1pw_qc = pd.read_csv(e1pw / "rectification_qc.csv")
    e3pw_metrics = pd.read_csv(e3pw / "delta_correlation_R2_by_pair.csv")
    e3pw_results = pd.read_csv(e3pw / "delta_correlation_predictions.csv.gz")
    e3pw_qc = pd.read_csv(e3pw / "rectification_qc.csv")
    e1pw_manifest = _json(e1pw / "manifest.json")
    e3pw_manifest = _json(e3pw / "manifest.json")
    e3single_summary = pd.read_csv(e3single / "single_phase_identity_R2_summary.csv")
    e3single9_metrics = pd.read_csv(e3single9 / "delta_correlation_R2_by_pair.csv")
    e3single9_results = pd.read_csv(e3single9 / "delta_correlation_predictions.csv.gz")
    e3single9_qc = pd.read_csv(e3single9 / "rectification_qc.csv")
    e3single9_manifest = _json(e3single9 / "manifest.json")
    e3phase_metrics = pd.read_csv(e3phase / "delta_correlation_R2_by_pair.csv")
    e3phase_results = pd.read_csv(e3phase / "delta_correlation_predictions.csv.gz")
    e3phase_qc = pd.read_csv(e3phase / "rectification_qc.csv")
    e3phase_manifest = _json(e3phase / "manifest.json")
    e3seed_metrics = pd.read_csv(e3seed / "metrics_by_phase_repeat.csv")

    radial_mean_err = np.median(abs(d2.E_R2_emp - d2.E_R2) / d2.E_R2)
    radial_var_err = np.median(abs(d2.Var_R2_emp - d2.Var_R2_theory) / d2.Var_R2_theory)
    radial_var_r = d2[["Var_R2_emp", "Var_R2_theory"]].corr().iloc[0, 1]
    eps_rmse = p1.assign(err2=lambda d: (d.predicted_covariance-d.observed_covariance)**2).groupby(["epsilon","method"]).err2.mean().pow(.5).unstack()
    sl = m2["Stuart-Landau"]["C_semi"]
    ei = m2["smooth local E-I"]["C_semi"]
    f3, s3 = m3["heldout_C_first"], m3["heldout_C_second"]
    held = t3[(t3.kind == "random15_delete") & (t3.split == "heldout")]
    sl_condition_fits = r2c[r2c.layer == "Stuart-Landau"].merge(d2[["condition", "c_i"]], on="condition")
    ci_coverage = np.mean((sl_condition_fits.c_i_ci_low <= sl_condition_fits.c_i) & (sl_condition_fits.c_i <= sl_condition_fits.c_i_ci_high))

    sections = []
    add = sections.append
    add(("title", "周期-非周期耦合理论数值实验：详细方法、结果与分析"))
    add(("subtitle", "基于 Li, Li & Wang (2025) marmoset E-I 全脑模型与 sEEG p-ap coupling theory notes"))
    add(("h1", "一、研究目标与总体设计"))
    add(("p", "本项目定量检验三条可能产生 periodic-aperiodic coupling 的路径。实验 1 固定网络结构，只改变 trial 间输入参数的联合抽样；实验 2 在近 Hopf 非线性系统中检验共同振幅变量是否同时调制瞬时频率和有限频带 exponent；实验 3 固定输入分布，通过单边、脑区入边/出边以及随机 15 边扰动检验结构 coupling kernel 的一阶和二阶预测。"))
    add(("p", "所有主要结论均来自实际 full simulation。实验 1 主分析共 56 个条件、每条件 4096 trial，总计 229,376 行；九输入区乘九输出区扩展另完成 81 x 56 x 1024=4,644,864 次解析 feature evaluation；完整 55 区阈值非线性时域验证另包含 14 x 64=896 条轨迹。实验 2 共 14,688 个非重叠分析窗口。实验 3 解析层包含 1500 个 random-15 mask 等结构扰动；九区阈值非线性验证对 120 个 held-out mask 和 64 个 common-random-number 输入逐一积分。"))
    add(("callout", "最重要的总体结论：输入抽样理论和结构 kernel 在解析层得到强支持，并能显著排序真实阈值非线性时域结果。32.768 s 长记录使实验 1 correlation slope 从 0.385 恢复到 0.949；但实验 3 原 V1->PG identity R2=0.775 的改善不能跨随机相位复现。六相位重复及相位平均的 9 x 9 复算表明，长窗仅略改善总体 OLS 排序，不能建立 Delta r 的绝对校准。实验 3 的有限窗效应必须与随机相位敏感性、极小 actual Delta r 方差及路径依赖残差共同解释。式（136）的径向平稳分布成立，而式（134）对预注册 100-1000 Hz exponent 没有得到 held-out 支持。"))

    add(("h1", "二、共同模型、PSD 与特征定义"))
    add(("h2", "2.1 复用的全脑 E-I 模型"))
    add(("p", "实验直接复用 Marmoset_Cortex_Model_code/standard_marmoset_simulation.py、subgraph_data_marmoset.pkl 和 hier_vals_fit.npy。每个脑区包含一个 excitatory population E 和一个 inhibitory population I。55 区状态向量采用 block ordering：[E1,...,E55,I1,...,I55]。"))
    add(("eq", "tau_E dr_Ei/dt = -r_Ei + beta_E[(1+eta_E h_i) w_EE r_Ei - w_EI r_Ii + (1+eta_E h_i) mu_EE sum_j FLN_ij r_Ej + I_Ei]"))
    add(("eq", "tau_I dr_Ii/dt = -r_Ii + beta_I[(1+eta_I h_i) w_IE r_Ei - w_II r_Ii + (1+eta_I h_i) mu_IE sum_j FLN_ij r_Ej + I_Ii]"))
    add(("p", "在线性 active set 中，上述方程写为 dx/dt=Jx+Bu。方向约定为 FLN[target,source]，即 source -> target；一条 anatomical edge 不只改变 J 的一个元素，而是同时改变 J[E_target,E_source] 和 J[I_target,E_source]。"))
    add(("table", ["参数", "数值", "单位/说明"], [
        ["beta_E / beta_I", "0.066 / 0.351", "population gain"],
        ["tau_E / tau_I", "20 / 10", "ms"],
        ["w_EE / w_EI / w_II", "24.4 / 19.7 / 12.5", "local weights"],
        ["w_IE", "11.6603801", "diverse_hi=41 标准条件"],
        ["mu_EE / mu_IE", "67.4 / 49.8104157", "long-range E-source gains"],
        ["eta_E / eta_E,local", "0.685 / 0.685", "hierarchical scaling"],
        ["eta_I / eta_I,local", "0.76 / 0.76", "hierarchical scaling"],
    ]))
    add(("h2", "2.2 Resolvent、输入 PSD 与输出 PSD"))
    add(("eq", "R_J(f) = [i 2 pi f / 1000 I - J]^-1"))
    add(("eq", "S_y(f;J,z) = |c^T R_J(f) B|^2 Q(f;z)"))
    add(("p", "J 的单位是 ms^-1，因此 Hz 转角频率时使用 2*pi*f/1000。实验 1 主分析使用完整 55 区网络，input 进入 V1 的 E population，readout 为 PG 的 E population；系统性扩展仍保留完整 55 区网络，但依次把 V1、V2、V4、MST、LIP、PG、A23a、A6DR、A6Va 作为输入与输出，形成 81 个 E-to-E readout 组合。实验 3 使用上述九区诱导子网络，input/output 为 V1 E -> PG E。解析 PSD 频率网格覆盖 0.5-1200 Hz。"))
    add(("eq", "Q(f;z) = q_ap (f/10)^beta + q_bump exp[-0.5 ((f-f0)/2 Hz)^2]"))
    add(("table", ["输入坐标 z", "基准", "标准化尺度", "允许范围"], [
        ["bump center f0", "10 Hz", "3 Hz", "5.1-19.5 Hz"],
        ["bump amplitude q_bump", "6", "4", "1-15"],
        ["aperiodic beta", "-0.5", "0.8", "-1.8-1.0"],
        ["aperiodic amplitude q_ap", "1", "0.7", "0.15-3"],
    ]))
    add(("h2", "2.3 周期特征 P、非周期特征 A 与 QC"))
    add(("p", "P 定义为预注册 5-20 Hz peak band 内的唯一非退化主峰。程序对 log PSD 构造三次样条，求 d logS/df=0 的连续根，并要求二阶导数为负；如果第二个候选峰的功率与主峰相差小于 20%，标为 peak identity switching。表中同时保存 peak_hz 和 peak_omega=2*pi*peak_hz。"))
    add(("eq", "A = - slope[log S(f) versus log f],  f in [100,1000] Hz"))
    add(("p", "A 的频带对全部 condition 固定，不能按 trial 调整。新增全脑/九区时域验证统一使用 dt=0.25 ms、8192 点 Hann-Welch、50% overlap，并在 log PSD 上施加 sigma=2 frequency-bin 的 Gaussian smoothing，再应用同一连续峰标准。每个 trial 保存 peak_valid、peak_boundary、peak_switch 和 failure reason。主要分析只使用直接 PSD feature，不混入 FOOOF/specparam 估计误差。"))

    add(("h1", "三、实验 1：输入抽样是否诱导 p-ap coupling"))
    add(("h2", "3.1 科学问题"))
    add(("p", "固定同一个 55 区 Jacobian J，只改变输入参数 z 的 trial-to-trial 联合分布。目标是区分网络动力学敏感性与 experimenter sampling geometry：即相同网络能否仅因 Sigma_z、三阶矩或抽样约束不同而呈现正、负或近零 P-A covariance。"))
    add(("h2", "3.2 抽样设计与 simulation"))
    add(("table", ["设计", "具体实现"], [
        ["S1 independent Gaussian", "4 个标准化坐标独立 N(0,1)"],
        ["S2 correlated Gaussian", "center 与 beta 的 rho=-0.9,-0.6,-0.3,0,0.3,0.6,0.9"],
        ["S3 rank-one", "xi=u[1,0.6,-0.8,0.4]，u~N(0,1)"],
        ["S4 uniform", "各坐标独立 Uniform[-sqrt(3),sqrt(3)]"],
        ["S5 fixed-total", "bump-amplitude 与 aperiodic-amplitude 的标准化变化互为相反数"],
        ["S6 skewed", "中心化并标准化的 lognormal，产生非零三阶矩"],
        ["S7 one-factor mixture", "每个 trial 只改变 4 个坐标中的一个"],
        ["S8 finite sample", "population 独立，但比较理论 Sigma 与实际 sample Sigma_hat"],
    ]))
    add(("p", "每个设计扫描 epsilon={0.05,0.1,0.2,0.4}。delta z = epsilon * scale * xi。每个 condition 生成 4096 trial；越界样本按预先给定范围裁剪并记录 clipped。网络 transfer spectrum 只计算一次，每个 trial 仅更新输入 PSD 并提取 P、A，因此不存在时域积分 Monte Carlo 误差。"))
    add(("h2", "3.3 对应理论公式与计算"))
    add(("p", "在基准 z0 计算 p=grad_z P、a=grad_z A、H_P、H_A、T_P、T_A。连续峰求根使 P 对 z 可微；导数采用高精度中心差分，并在 50 个随机点使用两种步长交叉验证。"))
    add(("eq", "Eq.49: C_49 = p^T Sigma_z a"))
    add(("eq", "Eq.48: C_48 = C_49 + 1/2 sum_ijk M_ijk [p_i (H_A)_jk + a_i (H_P)_jk]"))
    add(("eq", "C_quad = C_49 + 1/2 tr(H_P Sigma_z H_A Sigma_z)"))
    add(("eq", "Eq.64: C_64 = C_quad + 1/2 p^T Sigma_z (T_A:Sigma_z) + 1/2 a^T Sigma_z (T_P:Sigma_z)"))
    add(("p", "每个 condition 使用实际被接受样本计算 Sigma_hat_z 和三阶中心矩 M。经验量为 sample covariance、Pearson r、Spearman r 及 bootstrap 95% CI。理论 correlation 的分子和分母来自同阶 Taylor surrogate：从实际标准化输入中重采样 30,000 次，用对应阶数同时计算 Cov、Var(P)、Var(A)，避免用理论 covariance 除以经验方差。"))
    add(("h2", "3.4 主要结果"))
    add(("table", ["理论", "R2", "RMSE", "回归斜率", "符号准确率"], [[k, _fmt(m1[k]["R2"]), _fmt(m1[k]["RMSE"]), _fmt(m1[k]["slope"]), _fmt(m1[k]["sign_accuracy"])] for k in ["C49","C48","Cquad","C64"]]))
    add(("p", f"56 个 condition 中，经验 covariance 范围为 {s1.covariance.min():.3f} 到 {s1.covariance.max():.3f}，Pearson r 范围为 {s1.pearson_r.min():.3f} 到 {s1.pearson_r.max():.3f}。总体 peak valid rate={t1.peak_valid.mean():.3%}，clipping rate={t1.clipped.mean():.3%}。rank-one 设计产生接近 1 的相关性，而独立、平衡或 one-factor 设计大多接近零；相关 Gaussian 改变 rho 时可连续改变 coupling 的方向与大小。"))
    add(("p", f"Eq.49 在本参数范围内最稳健，R2={m1['C49']['R2']:.6f}、RMSE={m1['C49']['RMSE']:.5f}。其 RMSE 随 epsilon 从 {eps_rmse.loc[.05,'C49']:.3g} 增至 {eps_rmse.loc[.4,'C49']:.3g}，符合局部展开随扰动尺度增大而退化的预期。Eq.64 在较大 epsilon 下过冲，整体 R2 降至 {m1['C64']['R2']:.3f}。"))
    add(("p", f"导数验证的中位相对误差：gradient={dv.grad_rel_error.median():.2e}，Hessian={dv.hessian_rel_error.median():.2e}，third tensor={dv.third_rel_error.median():.2e}。少数三阶误差较大的点接近强曲率或 peak identity 变化，已在 derivative_validation.csv 保留。"))
    add(("figure", e1/"figures"/"input_design_scatter.png", "图 1. 不同输入抽样设计下 P-A trial scatter。相同 J 在不同 sampling geometry 下可呈现不同相关结构。"))
    add(("figure", e1/"figures"/"observed_vs_predicted.png", "图 2. 56 个 condition 的经验 covariance/correlation 与 Eq.48、Eq.49、quadratic 和 Eq.64 预测。"))
    add(("figure", e1/"figures"/"error_vs_epsilon.png", "图 3. 理论误差随 epsilon 增长。局部一阶理论在小尺度最准确，高阶式并非在所有尺度单调改善。"))
    add(("h2", "3.5 九输入区乘九输出区的系统性验证"))
    add(("p", "在完整 55 区网络中，输入区和输出区分别遍历九个目标脑区，共 81 对。每一对均复用主实验全部 56 种输入抽样 condition；每 condition 使用 1024 trial，并用相同 condition seed 形成 common random numbers。每个脑区对在自己的基准 transfer 上重新计算 grad_z P 与 grad_z A，再以实际接受样本的 Sigma_hat_z 计算 Eq.49。每格相似度由 56 个 condition 上 observed covariance 与 Eq.49 predicted covariance 的 R2 和 Pearson r 定义，而不是单个 trial 的相关系数。"))
    add(("p", f"81 对的 Eq.49 R2 范围为 {m1p['R2_min']:.6f}-{m1p['R2_max']:.6f}，中位数 {m1p['R2_median']:.6f}；Pearson r 最低为 {m1p['pearson_r_min']:.6f}，中位数 {m1p['pearson_r_median']:.6f}。各对的 condition 中位有效率为 {m1p['median_valid_rate']:.3%}。因此 Eq.49 的准确性并非 V1-to-PG 特例，而是在九输入区乘九输出区中系统成立。"))
    add(("figure", e1p/"figures"/"pairwise_io_theory_similarity.png", "图 3b. 九输入区乘九输出区共 81 种组合中，Eq.49 对 56 个输入 condition 的理论-实际相似度。颜色范围限制在实际观测最小值至最大值，以显示接近 1 区间内的细微差别；格内数字给出原始 R2 或 Pearson r。"))
    add(("h2", "3.6 完整 55 区阈值非线性时域验证"))
    add(("p", "为检验解析 PSD 结果是否可由原 published dynamics 读出，直接积分完整 55 区 threshold-linear E-I 方程，保留 max(I,0) 非线性。选择 epsilon=0.2 下 7 种基本设计和 7 个 correlated-Gaussian rho，共 14 condition；每 condition 64 trial。每个 z 的目标 Q(f;z) 由固定 Fourier 幅度和随机相位合成，输入 V1 E，读取 PG E；input sigma=0.5，dt=0.25 ms，burn-in=2.048 s，分析窗=4.096 s。理论和时域结果使用同一有效 trial 子集。"))
    add(("p", f"非线性时域 covariance 对 Eq.49 的跨 condition Pearson r={m1n['nonlinear_vs_C49_covariance']['pearson_r']:.3f}，identity R2={m1n['nonlinear_vs_C49_covariance']['R2']:.3f}，回归 slope={m1n['nonlinear_vs_C49_covariance']['slope']:.3f}，p={m1n['nonlinear_vs_C49_covariance']['regression_p_value']:.2e}，符号准确率={m1n['nonlinear_vs_C49_covariance']['sign_accuracy']:.1%}。correlation 的排序相关仍高，Pearson r={m1n['nonlinear_vs_linear_surrogate_correlation']['pearson_r']:.3f}，但 slope={m1n['nonlinear_vs_linear_surrogate_correlation']['slope']:.3f}、identity R2={m1n['nonlinear_vs_linear_surrogate_correlation']['R2']:.3f}。该幅度压缩与有限窗特征测量误差造成的 attenuation 一致，但本实验本身不能证明窗口长度是唯一来源。"))
    add(("p", f"全部 896 条轨迹的 threshold rectification fraction={m1n['rectification_fraction']:.3g}。这有力排除了 firing rate 穿过零阈值导致的 threshold switching，并将主要差异定位于连续时间 latent PSD 到实际谱特征估计的 observation layer；但它并不排除离散积分、输入合成、burn-in、spectral leakage、feature estimator bias 或有限 trial correlation bias。"))
    add(("figure", e1n/"figures"/"full55_nonlinear_vs_theory.png", "图 3c. 完整 55 区阈值非线性时域 readout 与 Eq.49、linear-surrogate correlation 及直接解析 PSD 的比较。虚线为 identity，实线为 OLS；所有面板标示回归系数、p、Pearson r、identity R2 和 condition 数。"))
    add(("conclusion", "实验 1 结论：P-A correlation 不是网络的单一固有常数。解析层 Eq.49 在 81 个 input-output 组合中都极准确；真实 55 区时域积分仍保留显著的理论排序与 covariance 定量一致性，而观测到的 correlation 幅度压缩与有限窗测量误差相符。必须区分 latent feature coupling 与 estimated feature coupling，并进一步分离 observation layer 内部的随机误差与系统偏差。"))

    add(("h1", "四、实验 2：near-Hopf amplitude-frequency coupling"))
    add(("h2", "4.1 层 A：Stuart-Landau benchmark"))
    add(("eq", "dZ = [(mu+i omega0)Z - (c_r+i c_i)|Z|^2 Z]dt + sqrt(2D)(dW_x+i dW_y)"))
    add(("p", "omega0=2*pi*10 rad/s。参数网格为 mu={-1,-0.3,-0.05,0.1,0.5} s^-1，D={0.01,0.04,0.12}，c_i={-4,0,4}，c_r={1,2}，共 90 condition。每个 condition 8 个 seed；每条轨迹 80 s，burn-in 10 s，dt=0.00025 s，非重叠窗口 4 s。"))
    add(("p", "积分器将线性相位旋转 exp(i omega0 dt) 精确推进，径向 drift 和噪声在 rotating frame 内更新，从而避免显式 Euler 对快速旋转产生的人工径向增长。每个窗口从真实复状态提取 R=|Z|、unwrap(Phi)、dotPhi、mean R^2、PSD peak 和 A。"))
    add(("h2", "4.2 式（136）径向密度检验"))
    add(("eq", "p_R(r) proportional to r exp[mu r^2/(2D) - c_r r^4/(4D)], r>=0"))
    add(("p", "对每个 condition 在一维 r 网格上归一化密度，并数值积分 E[R^2]、E[R^4] 和 Var(R^2)。经验量来自全部 8 个 seed 的 burn-in 后瞬时 R 样本，每 20 ms 抽样一次；这里特意不使用窗口均值方差，因为式（136）描述瞬时平稳径向分布。"))
    add(("p", f"E[R^2] 的中位相对误差为 {radial_mean_err:.2%}，Var(R^2) 的中位相对误差为 {radial_var_err:.2%}，理论与经验 Var(R^2) 跨 90 condition 的 Pearson r={radial_var_r:.4f}。因此随机积分器与式（136）的噪声 convention 一致。"))
    add(("figure", e2/"figures"/"radial_density.png", "图 4. 代表性 condition 的经验径向样本与 Eq.136 理论密度。"))
    add(("h2", "4.3 式（134）的 held-out 计算"))
    add(("eq", "dotPhi = omega0 - c_i R^2 + phase noise"))
    add(("eq", "A = chi0 + kappa_R R^2 + residual"))
    add(("eq", "Eq.134: Cov(dotPhi,A) approximately -c_i kappa_R Var(R^2)"))
    add(("p", "偶数 seed 为训练集，奇数 seed 为 held-out。训练集回归 dotPhi~mean(R^2) 得到 -c_i_train，回归 A~mean(R^2) 得到 kappa_R_train。三种预测为：C_reg=两个训练斜率乘 held-out Var(mean R^2)；C_semi=-c_i_train*kappa_R_train*Var_test(mean R^2)；C_full=-c_i_train*kappa_R_train*Var_density(R^2)。C_full 是文中形式的直接代入，但需要注意 density variance 是瞬时尺度，而观测 covariance 是 4 s 窗口尺度，因此它是一个刻意保留的尺度敏感性检验。"))
    add(("table", ["预测", "R2", "RMSE", "符号准确率"], [
        ["C_semi", _fmt(m2["Stuart-Landau"]["C_semi"]["R2"]), _fmt(m2["Stuart-Landau"]["C_semi"]["RMSE"]), _fmt(m2["Stuart-Landau"]["C_semi"]["sign_accuracy"])],
        ["C_full", _fmt(m2["Stuart-Landau"]["C_full"]["R2"]), _fmt(m2["Stuart-Landau"]["C_full"]["RMSE"]), _fmt(m2["Stuart-Landau"]["C_full"]["sign_accuracy"])],
    ]))
    add(("p", f"主分析结果不支持 Eq.134 对 100-1000 Hz A 的跨条件预测：C_semi R2={sl['R2']:.3f}、RMSE={sl['RMSE']:.5f}、sign accuracy={sl['sign_accuracy']:.3f}。c_i=0 被包含为负对照。约 10 Hz 振荡峰距离 100-1000 Hz 很远，高频 slope 对 R^2 的 kappa_R 很弱，窗口估计噪声因而主导 covariance；低振幅时 phase noise 还会降低 c_i 的回归稳定性。"))
    add(("figure", e2/"figures"/"common_amplitude_driver.png", "图 5. 代表 condition SL_mu-1_D0.01_ci4_cr1 的偶数 seed 训练窗口。实线为 OLS；每个面板标示回归系数 b、双侧 p、R2 与 n。A 与 mean(R2) 的关系几乎不可辨。"))
    add(("figure", e2/"figures"/"eq134_observed_predicted.png", "图 6. Stuart-Landau 与 smooth local E-I 的 Eq.134 semi-theory 对 held-out observed covariance。虚线为 identity，实线为跨 condition OLS；每个面板标示回归系数、p、R2 与 condition 数。"))
    add(("figure", e2/"figures"/"ci_zero_control.png", "图 7. c_i=0 负对照与 c_i非零条件的 held-out covariance 分布。"))
    add(("h2", "4.4 每个 condition 的回归质量与参数区间"))
    add(("p", "每个 condition 单独使用偶数 seed 的训练窗口做两条含截距 OLS。拟合优度定义为 R2=r^2；斜率标准误由 scipy.stats.linregress 给出，95% CI 使用 t_(n-2,0.975) x SE。c_i=-b_phi，因此其区间由 phidot slope 区间取负并反转端点；kappa_R=b_A，直接使用 A slope 区间。这些是 window-level OLS t 区间，未校正同一 seed 内窗口相关，解释时应结合仅 4 个训练 seed 的限制。"))
    add(("table", ["Stuart-Landau condition-level diagnostic", "结果"], [
        ["median R2: dotPhi~mean(R2)", _fmt(sl_condition_fits.phidot_R2_R2.median())],
        ["median R2: A~mean(R2)", _fmt(sl_condition_fits.A_R2_R2.median())],
        ["p<0.05: dotPhi fit", f"{(sl_condition_fits.phidot_R2_p<.05).mean():.1%}"],
        ["p<0.05: A fit", f"{(sl_condition_fits.A_R2_p<.05).mean():.1%}"],
        ["true c_i covered by nominal 95% CI", f"{ci_coverage:.1%}"],
    ]))
    add(("p", "完整 108 个 condition 的结果列于附录 A，并另存 condition_regression_stats.csv。CSV 还包含两条回归各自的 slope 和 p 值。整体上 phidot 通路已有一定信号，但 A 通路的中位 R2 仅约 0.011，且只有约 5% condition 达到 p<0.05，与偶然水平相当。"))
    add(("h2", "4.5 层 B：原模型局部 nonlinear E-I"))
    add(("p", "原始 published model 是 threshold-linear，不能在阈值处直接做光滑 Hopf reduction。因此构造与原局部 Jacobian 一致的 smooth tanh 系统：dx=(-x+tanh(Wx))/tau dt+sqrt(2D)dW，并使用 stochastic Heun 积分。扫描 55 个局部 block 后发现，只增大 E-E 会先到实特征值 instability。PG 是形成 complex Hopf 所需 cross-coupling 调整最小的脑区。"))
    add(("p", f"锁定 PG 的 I-to-E balance multiplier={m2['PG_I_to_E_balance_multiplier']:.4f}，然后扫描 E-E gain。Hopf gain={m2['hopf_gain']:.6f}；使用 gain=Hopf+[-0.12,-0.04,-0.01,0.02,0.08]，noise={0.0005,0.002,0.006}，每条件 8 seed，时间参数与层 A 相同。临界左特征向量 w 用于 Z(t)=w* [x(t)-x*] 投影，避免只用 E population Hilbert transform。"))
    add(("p", f"smooth E-I 的 C_semi held-out R2={ei['R2']:.3f}、RMSE={ei['RMSE']:.5f}，解释力很弱。linearized E-I 作为负对照，多数窗口没有 5-20 Hz 合格内点峰，符合无自持振荡预期。"))
    add(("figure", e2/"figures"/"ei_bifurcation.png", "图 8. PG smooth E-I 的 critical eigenvalue 与 E-E gain。实部在 gain=4.3394 附近过零，同时虚部非零，确认是 complex Hopf。"))
    add(("conclusion", "实验 2 结论：式（136）的径向统计获得强支持，但这不自动保证 Eq.134 可由任意 PSD exponent 观察到。对于本项目固定的 100-1000 Hz A，kappa_R 可辨识性不足，Eq.134 held-out 失败。它主要是 instantaneous-frequency-A 机制，不能无条件替代 PSD-peak-A coupling。"))

    add(("h1", "五、实验 3：structural coupling kernel"))
    add(("h2", "5.1 网络、输入与 perturbation"))
    add(("p", "为完成 1500 个 mask 的 direct+first+second evaluation，使用九区诱导子网络：V1、V2、V4、MST、LIP、PG、A23a、A6DR、A6Va。它保留各区 published local parameters 与精确 FLN-to-J map，包含 65 条非零有向 FLN edge。baseline 最大特征值实部小于 0。"))
    add(("p", "固定 256 个 common-random-number 输入 trial：标准化 z 为 epsilon=0.2 的 Gaussian，center-beta correlation rho=0.6。baseline 和所有 perturbation 使用完全相同的 z。主 perturbation 包括：全部 65 条单边删除、全部 65 条单边加倍、每个脑区全部 incoming edge 删除、每个脑区全部 outgoing edge 删除，以及 1500 个均匀无放回随机 15 边删除 mask。random masks 的前 300 为 pilot，后 1200 为 held-out。"))
    add(("h2", "5.2 anatomical pullback 与结构展开"))
    add(("eq", "Delta J = sum_(a,b) L_ab Delta W_ab,  L_ab = partial J / partial W_ab"))
    add(("p", "每条 FLN source b -> target a 的 L_ab 在 J 中有两个非零项：beta_E mu_EE exc_scale[a]/tau_E 写入 E_target,E_source；beta_I mu_IE inh_scale[a]/tau_I 写入 I_target,E_source。随机方向验证 <K_J,Delta J>_F=<K_SC,Delta W>_F，中位绝对误差 8.76e-9，最大 6.99e-7。"))
    add(("eq", "C^[1](J) = p(J)^T Sigma_z a(J)"))
    add(("eq", "First: C^[1](J+Delta J) approx C^[1](J) + D_J C^[1][Delta J]"))
    add(("eq", "Second: add 1/2 D_J^2 C^[1][Delta J,Delta J]"))
    add(("p", "一阶 anatomical edge kernel 对每条 edge 使用相对权重方向的中心差分 h=0.02。random-15 的一阶预测为 15 条 edge kernel contribution 之和。二阶不构造巨大四阶 tensor，而在实际 Delta J 方向上使用 h=0.05 的中心二阶方向导数。top 15 edge 另用 h=0.03 计算 105 个 mixed edge-pair interaction，用于 interaction heatmap。"))
    add(("p", "每个扰动同时保存 direct C_exact、rho_exact、C^[1](J+Delta J)、一阶/二阶预测、Delta J norm、max_f ||R(f)Delta J||_2、稳定性、有效 trial 数和 edge list。由此将 input expansion error、structural expansion error 和 end-to-end error 分开。"))
    add(("h2", "5.3 baseline 与 held-out 结果"))
    add(("table", ["量", "baseline exact", "input first-order"], [
        ["Cov(P,A)", _fmt(m3["baseline"]["exact"]["C_exact"]), _fmt(m3["baseline"]["C1"][0])],
        ["Pearson rho", _fmt(m3["baseline"]["exact"]["rho_exact"]), _fmt(m3["baseline"]["C1"][1])],
        ["near-separable control rho", _fmt(m3["baseline"]["near_separable_control"]["rho_exact"]), "-"],
    ]))
    add(("p", f"baseline input expansion absolute covariance error={m3['input_expansion_baseline_abs_error']:.6f}。near-separable control 的 rho={m3['baseline']['near_separable_control']['rho_exact']:.4f}、p={m3['baseline']['near_separable_control']['p_value']:.3f}，说明强 baseline coupling 来自所选相关输入分布，而不是 unavoidable network artifact。"))
    add(("table", ["held-out random-15", "一阶", "二阶"], [
        ["Pearson(pred,obs)", _fmt(f3["pearson_r"]), _fmt(s3["pearson_r"])],
        ["Spearman", _fmt(f3["spearman_r"]), _fmt(s3["spearman_r"])],
        ["R2", _fmt(f3["R2"]), _fmt(s3["R2"])],
        ["RMSE", _fmt(f3["RMSE"]), _fmt(s3["RMSE"])],
        ["regression slope", _fmt(f3["slope"]), _fmt(s3["slope"])],
        ["sign accuracy", _fmt(f3["sign_accuracy"]), _fmt(s3["sign_accuracy"])],
        ["top 10% overlap", _fmt(f3["top10pct_overlap"]), _fmt(s3["top10pct_overlap"])],
    ]))
    ci = m3["bootstrap_rmse_improvement_C_95CI"]
    add(("p", f"1200 held-out mask 中，一阶 R2={f3['R2']:.3f}，二阶提高到 {s3['R2']:.3f}；RMSE 从 {f3['RMSE']:.6f} 降至 {s3['RMSE']:.6f}。paired bootstrap 的 RMSE(first)-RMSE(second) 95% CI=[{ci[0]:.6f},{ci[1]:.6f}]，全为正，因此二阶改善稳定。"))
    add(("p", f"对结构内部目标 C^[1](J+Delta J)，一阶/二阶 R2={m3['structural_expansion_first_vs_C1']['R2']:.3f}/{m3['structural_expansion_second_vs_C1']['R2']:.3f}；对 end-to-end direct C_exact 则为 {f3['R2']:.3f}/{s3['R2']:.3f}。两者差异量化了 input Taylor error 与 structural Taylor error。"))
    add(("p", f"单边删除的二阶 R2={m3['single_delete_second']['R2']:.3f}，单边加倍 R2={m3['single_double_second']['R2']:.3f}，说明加倍与删除并非完全反对称。一次删除某区域全部 in/out edges 是大扰动，二阶局部展开仍失败，R2 分别为 {m3['area_incoming_second']['R2']:.3f}/{m3['area_outgoing_second']['R2']:.3f}。"))
    add(("h2", "5.4 九区阈值非线性时域 Delta correlation 验证"))
    add(("p", "从 1200 个 held-out random-15 删除 mask 中预先取前 120 个。生成 64 个与解析实验同分布的输入 trial，使用固定 Fourier 幅度和随机相位合成时域输入；baseline 与每个 mask 使用完全相同的输入信号。九区 threshold-linear 方程以 dt=0.25 ms 积分，burn-in=2.048 s、分析窗=4.096 s，输入 V1 E、读取 PG E。为避免 mask-dependent exclusion 污染 Delta r，最终只用 baseline 与全部 120 mask 共同 peak-valid 的固定 52 个 trial，并在该集合上重新计算 Sigma_z、直接解析结果和一/二阶 kernel。"))
    add(("table", ["预测量 vs nonlinear Delta r", "Pearson r", "slope", "identity R2", "sign accuracy"], [
        ["direct analytic", _fmt(m3n['delta_rho_analytic_direct']['pearson_r']), _fmt(m3n['delta_rho_analytic_direct']['slope']), _fmt(m3n['delta_rho_analytic_direct']['R2']), _fmt(m3n['delta_rho_analytic_direct']['sign_accuracy'])],
        ["first-order kernel", _fmt(m3n['delta_rho_first']['pearson_r']), _fmt(m3n['delta_rho_first']['slope']), _fmt(m3n['delta_rho_first']['R2']), _fmt(m3n['delta_rho_first']['sign_accuracy'])],
        ["second-order expansion", _fmt(m3n['delta_rho_second']['pearson_r']), _fmt(m3n['delta_rho_second']['slope']), _fmt(m3n['delta_rho_second']['R2']), _fmt(m3n['delta_rho_second']['sign_accuracy'])],
    ]))
    add(("p", f"真实 nonlinear Delta r 与 direct/first/second 的 Pearson r={m3n['delta_rho_analytic_direct']['pearson_r']:.3f}/{m3n['delta_rho_first']['pearson_r']:.3f}/{m3n['delta_rho_second']['pearson_r']:.3f}，回归 p 均远小于 0.001。二阶排序最高且符号准确率={m3n['delta_rho_second']['sign_accuracy']:.1%}；但 slope={m3n['delta_rho_second']['slope']:.2f}、identity R2={m3n['delta_rho_second']['R2']:.3f}，说明理论显著低估了时域 Delta r 的幅度。所有网络 rectification fraction 为 0；该结果将问题指向广义 observation layer，而普通独立测量噪声本身不足以解释稳定的尺度放大。"))
    add(("figure", e3n/"figures"/"nonlinear_delta_correlation_predictions.png", "图 12b. 九区真实阈值非线性时域 Delta correlation 对直接解析、一阶 kernel 和二阶展开。固定 52 个共同有效输入 trial；虚线为 identity、实线为 OLS。二阶对排序最好，但绝对尺度仍需经验校准或显式 observation-layer theory。"))
    add(("h2", "5.5 对有限窗解释的进一步评估"))
    add(("p", "总体判断是：报告原有解释在方向上合理，但证据尚不足以把全部解析-时域差异归结为窗口长度。完整 55 区实验中，covariance 对理论的回归 slope=0.741，而 correlation slope=0.385；896 条轨迹的 rectification fraction=0。设有限窗估计特征为："))
    add(("eq", "P_hat = P + epsilon_P,        A_hat = A + epsilon_A."))
    add(("p", "若测量误差近似独立于潜在特征，额外误差会增加两个估计特征的方差，并使观测 correlation 向零 attenuation。因此，实验 1 中 covariance 尚能较好保持、correlation 幅度却明显减小，确实支持 finite-window observation noise；但这仍是相容性证据，而非对窗口长度这一单一机制的识别。"))
    add(("p", "实验 3 呈现不同的尺度模式：二阶预测与 nonlinear Delta r 的 Pearson r=0.9031、符号准确率=90%，但回归 slope=5.616、identity R2=-0.02312；direct analytic 的 slope=8.906。普通、近似独立的测量噪声通常造成 attenuation，而不会稳定地将 Delta correlation 放大约 5.6 倍。更一般地，应把有限时长下的估计写为："))
    add(("eq", "r_hat_T(J) = r(J) + b_T(J) + epsilon_T(J),\nDelta r_hat_T = Delta r + [b_T(J + Delta J) - b_T(J)] + Delta epsilon_T."))
    add(("p", "只有当 Welch、peak 或 exponent estimator 的偏差 b_T(J) 随网络结构系统变化时，才可能出现稳定的尺度放大。因此仍可使用广义的 finite-window feature map 或 observation layer 解释，但不宜将其简化为“窗口太短”。目前至少有四类误差来源尚未排除。第一，解析采用连续时间 transfer R_c(f)=(i2 pi f I-J)^(-1)，而代码使用显式 Euler，dt=0.25 ms。其匹配的离散系统及 transfer 应为："))
    add(("eq", "A_d = I + Delta t J,        B_d = Delta t B,\nR_d(f) = [exp(i 2 pi f Delta t) I - A_d]^(-1) B_d."))
    add(("p", "在 1000 Hz 时每周期仅有 4 个采样点，2 pi f Delta t 约为 1.57，已不处于明显的小步长极限；而 exponent 正是在 100-1000 Hz 拟合，因此 continuous-discrete transfer mismatch 可直接改变高频 slope。第二，实验 3 仅使用 52 个共同 peak-valid trial，Fisher 变换标准误约为 SE[atanh(r_hat)]=1/sqrt(52-3)=0.143。common random numbers 可降低 baseline-perturbation 差值方差，却不能消除 sample correlation 分母波动、有限样本偏差和 feature noise；当理论 Delta r 范围很小时，小的系统误差也会形成很大的回归 slope。"))
    add(("p", "第三，当前 burn-in=2.048 s，仍需对 baseline 及每个 perturbation 计算 tau_slow=-1/max Re lambda(J)，检查 burn-in 是否覆盖多个最慢衰减时间；接近临界稳定边界的删边网络可能保留同时影响低频 peak 与高频 exponent 的 transient。第四，rectification fraction=0 仅有力排除了 firing-rate 穿零导致的 threshold switching，并未排除离散积分误差、输入合成误差、burn-in 不足、Hann spectral leakage、peak/slope estimator bias 与有限 trial correlation bias。"))
    add(("p", "为区分上述来源，可将验证分为五层。以下五层是完整诊断路线图；本次新增实验实际完成了谱估计器/记录长度对照，其余离散 transfer、频域 Monte Carlo 与逐点线性对照仍是后续建议。"))
    add(("bullet", "层 1：保留连续解析 PSD，S_c(f)=|c^T R_c(f)B|^2 Q(f)，作为 latent continuous-time 基准。"))
    add(("bullet", "层 2：计算与显式 Euler 匹配的离散解析 PSD，S_d(f)=|c^T R_d(f)B_d|^2 Q_d(f)，并比较层 1/2 的 peak、exponent、Cov、Var、correlation 和 Delta correlation；若此处已分离，主要来源是 continuous-discrete mismatch。"))
    add(("bullet", "层 3：对 S_d 施加与实际 Hann-Welch 完全一致的 spectral-window convolution，S_bar_d,T(f)=K_Welch,T*S_d(f)，随后运行同一 Gaussian smoothing、peak root finder 与 100-1000 Hz exponent regression，以估计有限窗的确定性 estimator bias。"))
    add(("bullet", "层 4：对每个固定 J,z 直接根据 S_d(f) 生成大量有限长度 Gaussian realization，运行相同 Welch、smoothing、peak、exponent 和 QC，估计 P_T_obs(J,z)=E[P_hat_T|J,z]、A_T_obs(J,z)=E[A_hat_T|J,z] 及两类 feature error covariance。该频域 Monte Carlo 可在不重复积分网络的条件下检验 observation layer 能否产生当前尺度放大。"))
    add(("bullet", "层 5：在相同输入、初始状态和随机相位下逐点比较去掉 rectification 的线性模型与当前 threshold-linear 模型。既然 rectification fraction=0，两者轨迹和 PSD feature 应几乎一致；否则应优先检查状态排序、输入缩放、积分器或实现。"))
    add(("p", "窗口长度扫描应保持 Welch 设置的可比性，并为每个条件使用至少 10-20 组独立 phase realization："))
    add(("compacttable", ["总分析时长", "Welch 段长", "主要目的"], [
        ["4.096 s", "2.048 s", "当前基准"],
        ["16.384 s", "2.048 s", "增加 Welch 平均次数，保持频率分辨率"],
        ["32.768 s", "2.048 s", "检查 estimator variance 是否继续下降"],
        ["32.768 s", "4.096 s", "同时改善低频 peak 分辨率"],
        ["65.536 s", "4.096 s", "长窗参考条件"],
    ]))
    add(("eq", "T increases  =>  slope -> 1,        R2_identity increases,        RMSE decreases."))
    add(("p", "本次直接实现了三个可比 readout：原基准为 4.096 s 总记录、2.048 s Hann-Welch 段和 50% overlap，共 3 个 segment；无 taper 对照使用同一 4.096 s 轨迹的单段 full-record boxcar periodogram，共 1 个 segment；长记录条件使用 32.768 s 总记录、4.096 s Hann-Welch 段和 50% overlap，共 15 个 segment。三者保持 dt=0.25 ms、burn-in=2.048 s、64 个输入 trial、相同输入参数分布和约 0.98 Hz 的 Gaussian smoothing 带宽；实验 3 还保持相同的 120 个 random-15 mask。"))
    fw1 = fw_summary[(fw_summary.experiment == "exp1") & (fw_summary.target == "correlation")].set_index("variant")
    fw1c = fw_summary[(fw_summary.experiment == "exp1") & (fw_summary.target == "covariance")].set_index("variant")
    add(("resulttable", ["实验 1 readout", "cov slope", "cov identity R2", "r slope", "r identity R2", "r Pearson"], [
        ["4.096 s Hann-Welch", _fmt(fw1c.loc['baseline_hann','slope']), _fmt(fw1c.loc['baseline_hann','identity_R2']), _fmt(fw1.loc['baseline_hann','slope']), _fmt(fw1.loc['baseline_hann','identity_R2']), _fmt(fw1.loc['baseline_hann','pearson_r'])],
        ["4.096 s boxcar", _fmt(fw1c.loc['short_boxcar','slope']), _fmt(fw1c.loc['short_boxcar','identity_R2']), _fmt(fw1.loc['short_boxcar','slope']), _fmt(fw1.loc['short_boxcar','identity_R2']), _fmt(fw1.loc['short_boxcar','pearson_r'])],
        ["32.768 s Hann-Welch", _fmt(fw1c.loc['long_hann','slope']), _fmt(fw1c.loc['long_hann','identity_R2']), _fmt(fw1.loc['long_hann','slope']), _fmt(fw1.loc['long_hann','identity_R2']), _fmt(fw1.loc['long_hann','pearson_r'])],
    ]))
    add(("p", f"实验 1 的长记录结果几乎恢复 identity：covariance slope={fw1c.loc['long_hann','slope']:.3f}、identity R2={fw1c.loc['long_hann','identity_R2']:.3f}；correlation slope={fw1.loc['long_hann','slope']:.3f}、identity R2={fw1.loc['long_hann','identity_R2']:.3f}、Pearson r={fw1.loc['long_hann','pearson_r']:.3f}。相较基准，correlation RMSE 从 {fw1.loc['baseline_hann','RMSE']:.3f} 降到 {fw1.loc['long_hann','RMSE']:.3f}。这满足预注册的 slope趋近1、identity R2上升和 RMSE下降三项判据。"))
    add(("figure", fw/"figures"/"finite_window_exp1.png", "图 3d. 实验 1 的有限记录对照。蓝色为原 4.096 s Hann-Welch，红色为同长度单段 boxcar，绿色为 32.768 s 长记录 Hann-Welch。长记录使 covariance 与 correlation 均接近 identity；单段 boxcar 因 periodogram 方差和 leakage 出现反号。"))
    add(("h2free", "5.5.1 实验 1：一阶/二阶与 9 x 9 真实时域系统验证"))
    add(("p", "新增系统实验不再把 resolvent feature evaluation 当作 actual simulation。对九个输入脑区逐一注入输入，在完整 55 区 threshold-linear 网络中显式 Euler 积分，并在同一次积分中保存九个输出脑区 E population 的时间序列。每个输入位置包含 epsilon=0.2 下 14 个输入 condition、每 condition 64 trial，共 8064 条完整长窗网络轨迹。每条 32.768 s 轨迹的前 4.096 s 作为短窗，因此短窗/长窗使用相同 z、相同随机相位和相同动力学 realization。"))
    add(("eq", "F_first(u) = F(0) + grad(F)^T u,\nF_second(u) = F_first(u) + 1/2 u^T H_F u,        F=(P,A)."))
    add(("p", "对每个 input-output-window-condition，在真实时域 PSD feature 上计算 actual covariance 与 actual Pearson r；理论一阶和二阶则在完全相同的 accepted u 子集上计算。线性拟合图同时报告 OLS slope、p、OLS R2=Pearson r^2 和 identity R2。9 x 9 热图使用跨 14 个 condition 的 OLS R2，衡量预测是否解释 condition 间变化；它与检验绝对幅度校准的 identity R2 不同。"))
    e1_v1pg_cor = e1pw_cor[(e1pw_cor.input_area == "V1") & (e1pw_cor.output_area == "PG")].set_index(["window", "order"])
    e1_v1pg_cov = e1pw_cov[(e1pw_cov.input_area == "V1") & (e1pw_cov.output_area == "PG")].set_index(["window", "order"])
    add(("resulttable", ["实验 1 V1->PG", "short first", "short second", "long first", "long second", "长窗变化"], [
        ["covariance OLS R2", _fmt(e1_v1pg_cov.loc[("short","first"),"ols_R2"]), _fmt(e1_v1pg_cov.loc[("short","second"),"ols_R2"]), _fmt(e1_v1pg_cov.loc[("long","first"),"ols_R2"]), _fmt(e1_v1pg_cov.loc[("long","second"),"ols_R2"]), "显著提高"],
        ["correlation OLS R2", _fmt(e1_v1pg_cor.loc[("short","first"),"ols_R2"]), _fmt(e1_v1pg_cor.loc[("short","second"),"ols_R2"]), _fmt(e1_v1pg_cor.loc[("long","first"),"ols_R2"]), _fmt(e1_v1pg_cor.loc[("long","second"),"ols_R2"]), "显著提高"],
        ["correlation identity R2", _fmt(e1_v1pg_cor.loc[("short","first"),"R2"]), _fmt(e1_v1pg_cor.loc[("short","second"),"R2"]), _fmt(e1_v1pg_cor.loc[("long","first"),"R2"]), _fmt(e1_v1pg_cor.loc[("long","second"),"R2"]), "恢复校准"],
    ]))
    add(("p", f"系统结果非常一致。81 对 input-output 的 correlation OLS R2 中位数由短窗一阶 {e1pw_cor[(e1pw_cor.window=='short')&(e1pw_cor.order=='first')].ols_R2.median():.3f}、二阶 {e1pw_cor[(e1pw_cor.window=='short')&(e1pw_cor.order=='second')].ols_R2.median():.3f}，升至长窗一阶 {e1pw_cor[(e1pw_cor.window=='long')&(e1pw_cor.order=='first')].ols_R2.median():.3f}、二阶 {e1pw_cor[(e1pw_cor.window=='long')&(e1pw_cor.order=='second')].ols_R2.median():.3f}。covariance 的对应中位数从 {e1pw_cov[(e1pw_cov.window=='short')&(e1pw_cov.order=='first')].ols_R2.median():.3f}/{e1pw_cov[(e1pw_cov.window=='short')&(e1pw_cov.order=='second')].ols_R2.median():.3f} 升至 {e1pw_cov[(e1pw_cov.window=='long')&(e1pw_cov.order=='first')].ols_R2.median():.3f}/{e1pw_cov[(e1pw_cov.window=='long')&(e1pw_cov.order=='second')].ols_R2.median():.3f}。一阶与二阶几乎相同，说明当前 epsilon=0.2 下主要偏差不是漏掉输入 Hessian，而是短记录对 P、A 及其 correlation 的估计误差。"))
    add(("p", f"有效 trial 数范围为 35-64；最大 threshold rectification fraction={e1pw_qc.rectification_fraction.max():.2e}，只占极少状态更新，最小 E rate={e1pw_qc.min_rate_E.min():.4f} Hz。该结果支持有限窗解释具有跨路径普遍性，但也提示完整 55 区少数输入 realization 极接近阈值，不能把 threshold effect 声称为严格为零。"))
    add(("figure", e1pw/"figures"/"exp1_V1_PG_covariance_short_long_orders.png", "图 3e. 实验 1 V1 输入、PG 输出：短窗/长窗和一阶/二阶 covariance 预测对真实 55 区时域 simulation。虚线为 identity，实线为 OLS；每格标示 slope、p、OLS R2 与 identity R2。"))
    add(("figure", e1pw/"figures"/"exp1_V1_PG_correlation_short_long_orders.png", "图 3f. 实验 1 V1 输入、PG 输出：短窗/长窗和一阶/二阶 correlation 预测对真实 55 区时域 simulation。长窗同时恢复排序与幅度校准。"))
    add(("figure", e1pw/"figures"/"exp1_covariance_9x9_R2.png", "图 3g. 实验 1 九输入乘九输出的 covariance OLS R2。四个面板依次比较短窗一阶、短窗二阶、长窗一阶和长窗二阶；每格均来自真实 55 区时域 readout 的 14 个 condition。"))
    add(("figure", e1pw/"figures"/"exp1_correlation_9x9_R2.png", "图 3h. 实验 1 九输入乘九输出的 correlation OLS R2。长窗在全部 81 条 input-output 路径上接近 1，而一阶与二阶差异很小。"))
    add(("p", f"进一步使用 identity R2 检验绝对校准，而不仅检验线性排序。实验 1 covariance 的 identity R2 中位数由短窗一阶/二阶的 {e1pw_cov[(e1pw_cov.window=='short')&(e1pw_cov.order=='first')].R2.median():.3f}/{e1pw_cov[(e1pw_cov.window=='short')&(e1pw_cov.order=='second')].R2.median():.3f}，提高到长窗的 {e1pw_cov[(e1pw_cov.window=='long')&(e1pw_cov.order=='first')].R2.median():.3f}/{e1pw_cov[(e1pw_cov.window=='long')&(e1pw_cov.order=='second')].R2.median():.3f}；correlation 的中位数则由 {e1pw_cor[(e1pw_cor.window=='short')&(e1pw_cor.order=='first')].R2.median():.3f}/{e1pw_cor[(e1pw_cor.window=='short')&(e1pw_cor.order=='second')].R2.median():.3f} 提高到 {e1pw_cor[(e1pw_cor.window=='long')&(e1pw_cor.order=='first')].R2.median():.3f}/{e1pw_cor[(e1pw_cor.window=='long')&(e1pw_cor.order=='second')].R2.median():.3f}。因此短窗中的正 OLS R2 只表示 condition 排序仍部分保留；负 identity R2 表明幅度 attenuation 严重。长窗同时修复了排序和绝对尺度。"))
    add(("figure", e1pw/"figures"/"exp1_covariance_9x9_identity_R2.png", "图 3i. 实验 1 九输入乘九输出的 covariance identity R2。格内为未截断原始值；统一色轴在 -2.5 处截断。长窗一阶与二阶均在全部路径上接近 1。"))
    add(("figure", e1pw/"figures"/"exp1_correlation_9x9_identity_R2.png", "图 3j. 实验 1 九输入乘九输出的 correlation identity R2。短窗全部路径为负，长窗全部恢复至约 0.98-0.99，说明有限窗主要造成幅度压缩而非路径特异的理论失效。"))
    fw3 = fw_summary[(fw_summary.experiment == "exp3") & (fw_summary.target == "second_order")].set_index("variant")
    add(("resulttable", ["实验 3 单相位 readout（历史结果）", "共同有效 trial", "second slope", "identity R2", "Pearson r", "RMSE"], [
        ["4.096 s Hann-Welch", int(fw3.loc['baseline_hann','n_common_valid_trials']), _fmt(fw3.loc['baseline_hann','slope']), _fmt(fw3.loc['baseline_hann','identity_R2']), _fmt(fw3.loc['baseline_hann','pearson_r']), _fmt(fw3.loc['baseline_hann','RMSE'])],
        ["4.096 s boxcar", int(fw3.loc['short_boxcar','n_common_valid_trials']), _fmt(fw3.loc['short_boxcar','slope']), _fmt(fw3.loc['short_boxcar','identity_R2']), _fmt(fw3.loc['short_boxcar','pearson_r']), _fmt(fw3.loc['short_boxcar','RMSE'])],
        ["32.768 s Hann-Welch", int(fw3.loc['long_hann','n_common_valid_trials']), _fmt(fw3.loc['long_hann','slope']), _fmt(fw3.loc['long_hann','identity_R2']), _fmt(fw3.loc['long_hann','pearson_r']), _fmt(fw3.loc['long_hann','RMSE'])],
    ]))
    add(("p", f"原图 12c 使用一个固定随机相位 realization 和 120 个 mask，得到长窗二阶 slope={fw3.loc['long_hann','slope']:.3f}、identity R2={fw3.loc['long_hann','identity_R2']:.3f}。这一结果曾被解释为长窗恢复校准；但它与原 9 x 9 扫描的 V1->PG 长窗二阶 identity R2={e3pw_metrics[(e3pw_metrics.input_area=='V1')&(e3pw_metrics.output_area=='PG')&(e3pw_metrics.window=='long')&(e3pw_metrics.order=='second')].R2.iloc[0]:.3f} 冲突。逐项核对表明，两套结构预测相关系数大于 0.999999，冲突来自真实时域 PSD 特征对随机 Fourier phase realization 的敏感性，而不是 mask、kernel 或输出索引错误。"))
    add(("p", f"单段 boxcar 的实验 1 correlation slope={fw1.loc['short_boxcar','slope']:.3f}、Pearson r={fw1.loc['short_boxcar','pearson_r']:.3f}；实验 3 二阶 slope={fw3.loc['short_boxcar','slope']:.1f}、Pearson r={fw3.loc['short_boxcar','pearson_r']:.3f}。这不是去窗后的改善，而是单段 raw periodogram 缺少 Welch averaging 后的高方差与 rectangular leakage。"))
    add(("figure", fw/"figures"/"finite_window_exp3.png", "图 12c. 原实验 3 单随机相位的有限记录对照。长窗二阶 identity R2=0.775 仅描述该 realization；后续相位重复显示它不是稳健的跨相位结论，因此本图保留为冲突审计而不再作为主要证据。"))
    add(("h2free", "5.5.2 图 12c 单相位方法的一阶/二阶复算"))
    add(("p", "为避免把9 x 9扫描的另一相位 seed 混入比较，新增复算严格复用图12c的设置：V1 E输入、PG E输出、phase-seed offset=220000、64个输入参数trial、前120个held-out random-15 mask、baseline与全部mask共用相同时域输入。短窗为4.096 s、2.048 s Hann-Welch段；长窗为32.768 s、4.096 s Hann-Welch段。每个窗口分别对baseline与全部120个mask取共同peak-valid输入集合，再在该集合上重算Sigma_z及结构一阶/二阶预测。"))
    e3single_index = e3single_summary.set_index(["window", "order"])
    add(("resulttable", ["图 12c 单相位复算", "short first", "short second", "long first", "long second", "统计层级"], [
        ["共同输入 trial/r", int(e3single_index.loc[("short","first"),"n_common_valid_trials"]), int(e3single_index.loc[("short","second"),"n_common_valid_trials"]), int(e3single_index.loc[("long","first"),"n_common_valid_trials"]), int(e3single_index.loc[("long","second"),"n_common_valid_trials"]), "每个网络内 P-A correlation"],
        ["identity R2", _fmt(e3single_index.loc[("short","first"),"identity_R2"]), _fmt(e3single_index.loc[("short","second"),"identity_R2"]), _fmt(e3single_index.loc[("long","first"),"identity_R2"]), _fmt(e3single_index.loc[("long","second"),"identity_R2"]), "120个mask上的绝对校准"],
        ["OLS R2", _fmt(e3single_index.loc[("short","first"),"ols_R2"]), _fmt(e3single_index.loc[("short","second"),"ols_R2"]), _fmt(e3single_index.loc[("long","first"),"ols_R2"]), _fmt(e3single_index.loc[("long","second"),"ols_R2"]), "120个mask上的线性排序"],
        ["OLS slope", _fmt(e3single_index.loc[("short","first"),"slope"]), _fmt(e3single_index.loc[("short","second"),"slope"]), _fmt(e3single_index.loc[("long","first"),"slope"]), _fmt(e3single_index.loc[("long","second"),"slope"]), "理想值为1"],
        ["regression p", f"{e3single_index.loc[('short','first'),'regression_p_value']:.1e}", f"{e3single_index.loc[('short','second'),'regression_p_value']:.1e}", f"{e3single_index.loc[('long','first'),'regression_p_value']:.1e}", f"{e3single_index.loc[('long','second'),'regression_p_value']:.1e}", "检验非零OLS slope"],
    ]))
    add(("p", f"独立输出目录中的重新积分与原图12c数据逐mask完全一致，actual Delta r、一阶和二阶预测的最大绝对差均为0。该固定单相位下，短窗一阶/二阶 identity R2={e3single_index.loc[('short','first'),'identity_R2']:.3f}/{e3single_index.loc[('short','second'),'identity_R2']:.3f}，长窗提高到 {e3single_index.loc[('long','first'),'identity_R2']:.3f}/{e3single_index.loc[('long','second'),'identity_R2']:.3f}；长窗一阶和二阶均明显改善，二阶更接近identity。这个结果确认图12c内部的计算是可复现的，但其推断范围仍限定在phase seed 220000，不能替代下一节的跨相位稳健性检验。"))
    add(("figure", e3single/"figures"/"single_phase_short_long_first_second_identity_R2.png", "图 12c 扩展. 严格沿用图12c单相位核算方法的短窗/长窗一阶与二阶结果。左侧四格给出120个mask的理论-实际Delta r回归；右侧汇总identity R2。短窗一阶/二阶为0.024/-0.023，长窗为0.675/0.775。"))
    add(("h2free", "5.5.3 图 12c 单相位方法的 9 x 9 系统验证"))
    add(("p", f"系统扩展保持phase-seed offset={e3single9_manifest['phase_seed_offset']}、64个输入参数trial和120个相同held-out mask，依次把九个脑区作为E输入；每次真实九区时域积分同时保存九个E输出。短窗和长窗与图12c一样分别合成各自长度的Fourier输入，而不是把短窗定义为长轨迹前缀。每个input-output-window组合分别取baseline与120个mask的共同peak-valid trial，在同一集合上计算真实r、Delta r及理论一阶/二阶预测。每格identity R2均以120个mask的actual Delta r为因变量计算。"))
    e3s9_sf = e3single9_metrics[(e3single9_metrics.window == "short") & (e3single9_metrics.order == "first")]
    e3s9_ss = e3single9_metrics[(e3single9_metrics.window == "short") & (e3single9_metrics.order == "second")]
    e3s9_lf = e3single9_metrics[(e3single9_metrics.window == "long") & (e3single9_metrics.order == "first")]
    e3s9_ls = e3single9_metrics[(e3single9_metrics.window == "long") & (e3single9_metrics.order == "second")]
    e3s9_v1pg = e3single9_metrics[(e3single9_metrics.input_area == "V1") & (e3single9_metrics.output_area == "PG")].set_index(["window", "order"])
    add(("resulttable", ["12c单相位 9 x 9", "short first", "short second", "long first", "long second", "跨81条路径"], [
        ["identity R2 中位数", _fmt(e3s9_sf.R2.median()), _fmt(e3s9_ss.R2.median()), _fmt(e3s9_lf.R2.median()), _fmt(e3s9_ls.R2.median()), "稳健中心"],
        ["identity R2 均值", _fmt(e3s9_sf.R2.mean()), _fmt(e3s9_ss.R2.mean()), _fmt(e3s9_lf.R2.mean()), _fmt(e3s9_ls.R2.mean()), "受极端负值影响"],
        ["identity R2 > 0 比例", _fmt((e3s9_sf.R2>0).mean()), _fmt((e3s9_ss.R2>0).mean()), _fmt((e3s9_lf.R2>0).mean()), _fmt((e3s9_ls.R2>0).mean()), "优于actual均值基线"],
        ["identity R2 > 0.5 比例", _fmt((e3s9_sf.R2>0.5).mean()), _fmt((e3s9_ss.R2>0.5).mean()), _fmt((e3s9_lf.R2>0.5).mean()), _fmt((e3s9_ls.R2>0.5).mean()), "较强绝对校准"],
        ["OLS R2 中位数", _fmt(e3s9_sf.ols_R2.median()), _fmt(e3s9_ss.ols_R2.median()), _fmt(e3s9_lf.ols_R2.median()), _fmt(e3s9_ls.ols_R2.median()), "mask排序"],
    ]))
    add(("p", f"V1->PG内部对照在多输出积分中得到短窗一阶/二阶identity R2={e3s9_v1pg.loc[('short','first'),'R2']:.3f}/{e3s9_v1pg.loc[('short','second'),'R2']:.3f}、长窗={e3s9_v1pg.loc[('long','first'),'R2']:.3f}/{e3s9_v1pg.loc[('long','second'),'R2']:.3f}，与单输出图12c的0.024/-0.023和0.675/0.775高度一致，验证了相位与核算规则。"))
    add(("p", f"然而，V1->PG不是全局代表。81条路径的identity R2中位数在短窗一阶/二阶为 {e3s9_sf.R2.median():.3f}/{e3s9_ss.R2.median():.3f}，长窗为 {e3s9_lf.R2.median():.3f}/{e3s9_ls.R2.median():.3f}，四者均为负。长窗相对短窗在一阶和二阶各改善50/81条路径，但少数极端恶化路径使均值下降；二阶相对一阶则在短窗改善59/81、长窗改善67/81条路径。故固定seed 220000下，二阶通常有益、长窗对多数路径有小幅改善，但“恢复identity校准”只发生在少数路径，不能由V1->PG推广到整个9 x 9。"))
    add(("p", f"短窗共同有效输入trial范围为 {int(e3single9_results[e3single9_results.window=='short'].n_valid_common.min())}-{int(e3single9_results[e3single9_results.window=='short'].n_valid_common.max())}，中位数 {e3single9_results[e3single9_results.window=='short'].n_valid_common.median():.0f}；长窗为 {int(e3single9_results[e3single9_results.window=='long'].n_valid_common.min())}-{int(e3single9_results[e3single9_results.window=='long'].n_valid_common.max())}，中位数 {e3single9_results[e3single9_results.window=='long'].n_valid_common.median():.0f}。最大rectification fraction={e3single9_qc.rectification_fraction.max():.2e}，不足以解释路径差异。"))
    add(("figure", e3single9/"figures"/"exp3_12c_single_phase_9x9_identity_R2.png", "图 12c 系统扩展. 与图12c相同单相位realization和120个mask下，九输入乘九输出的短窗/长窗一阶、二阶identity R2。横轴是输入脑区，纵轴是输出脑区；格内保留原始值，色轴在-2.5截断。V1输入-PG输出位于PG行、V1列。"))
    add(("h2free", "5.5.4 实验 3：随机相位审计与 9 x 9 相位平均复算"))
    add(("p", "定向敏感性实验使用 V1 输入、PG 输出、前 40 个固定 held-out random-15 mask、256 个输入参数 trial，并对 6 个相互独立的 Fourier phase seed 分别运行九区 threshold-linear 时域积分。每个 phase seed 独立计算 Delta r；同时将同一输入参数、网络与频率点上的 6 份真实时域 PSD 先求均值，再提取 peak 与 exponent，以形成 phase-mean-PSD readout。"))
    e3seed_repeat = e3seed_metrics[~e3seed_metrics.phase_repeat.astype(str).isin(["ensemble_mean", "psd_ensemble"])]
    e3seed_psd = e3seed_metrics[e3seed_metrics.phase_repeat.astype(str) == "psd_ensemble"].set_index(["window", "order"])
    seed_long_second = e3seed_repeat[(e3seed_repeat.window == "long") & (e3seed_repeat.order == "second")]
    add(("p", f"6 个单相位结果的长窗二阶 identity R2 从 {seed_long_second.R2.min():.3f} 到 {seed_long_second.R2.max():.3f}，OLS slope 甚至跨越正负号，说明单相位结论不稳定。先平均 PSD 后，V1->PG 二阶 identity R2 为短窗 {e3seed_psd.loc[('short','second'),'R2']:.3f}、长窗 {e3seed_psd.loc[('long','second'),'R2']:.3f}；长窗没有改善。"))
    add(("figure", e3seed/"figures"/"identity_R2_phase_sensitivity.png", "图 12d. V1 输入、PG 输出的随机相位敏感性。圆点是六个独立 phase seed 的 identity R2；虚线为先对各 seed 的 Delta r 求均值，点线为先平均真实时域 PSD 再提取特征。单 realization 可给出相反结论，phase-mean-PSD 结果不支持长窗恢复。"))
    add(("p", f"随后完成完整相位平均系统复算：9 个输入区 x 41 个网络（baseline 加 40 个固定 mask）x 64 个输入 trial x {e3phase_manifest['n_phase_repeats']} 个 phase realization，共 {9*41*64*e3phase_manifest['n_phase_repeats']:,} 条长窗九区网络轨迹；每次积分同时读取 9 个输出。actual PSD 是 6 个真实阈值非线性时域 simulation 的 PSD 均值，不是 resolvent 直接生成结果。短窗为每条长轨迹的前 4.096 s，长窗为 32.768 s。"))
    add(("eq", "Delta r_first = D_J r[Delta J],\nDelta r_second = Delta r_first + 1/2 D_J^2 r[Delta J,Delta J]."))
    add(("p", "每个 input-output-window 组合取 baseline 与全部 40 个 mask 的共同 peak-valid trial，并在同一 accepted 输入子集上重算 Sigma_z。横轴为结构一阶或二阶预测；纵轴为相位平均 PSD 经同一 peak/exponent pipeline 得到的真实时域 Delta r。OLS R2 衡量 mask 间线性排序；identity R2 检验无需经验斜率或截距的绝对校准。"))
    e3_short_first = e3phase_metrics[(e3phase_metrics.window == "short") & (e3phase_metrics.order == "first")]
    e3_short_second = e3phase_metrics[(e3phase_metrics.window == "short") & (e3phase_metrics.order == "second")]
    e3_long_first = e3phase_metrics[(e3phase_metrics.window == "long") & (e3phase_metrics.order == "first")]
    e3_long_second = e3phase_metrics[(e3phase_metrics.window == "long") & (e3phase_metrics.order == "second")]
    e3_v1pg = e3phase_metrics[(e3phase_metrics.input_area == "V1") & (e3phase_metrics.output_area == "PG")].set_index(["window", "order"])
    add(("resulttable", ["实验 3 相位平均 9 x 9", "short first", "short second", "long first", "long second", "解释"], [
        ["OLS R2 中位数", _fmt(e3_short_first.ols_R2.median()), _fmt(e3_short_second.ols_R2.median()), _fmt(e3_long_first.ols_R2.median()), _fmt(e3_long_second.ols_R2.median()), "排序/线性关系"],
        ["OLS R2 均值", _fmt(e3_short_first.ols_R2.mean()), _fmt(e3_short_second.ols_R2.mean()), _fmt(e3_long_first.ols_R2.mean()), _fmt(e3_long_second.ols_R2.mean()), "跨路径平均"],
        ["identity R2 中位数", _fmt(e3_short_first.R2.median()), _fmt(e3_short_second.R2.median()), _fmt(e3_long_first.R2.median()), _fmt(e3_long_second.R2.median()), "绝对幅度校准"],
        ["正 OLS slope 比例", _fmt((e3_short_first.slope>0).mean()), _fmt((e3_short_second.slope>0).mean()), _fmt((e3_long_first.slope>0).mean()), _fmt((e3_long_second.slope>0).mean()), "方向一致率"],
    ]))
    add(("p", f"校正后的 V1->PG 二阶 identity R2 为短窗 {e3_v1pg.loc[('short','second'),'R2']:.3f}、长窗 {e3_v1pg.loc[('long','second'),'R2']:.3f}，对应 slope={e3_v1pg.loc[('short','second'),'slope']:.3f}/{e3_v1pg.loc[('long','second'),'slope']:.3f}；两者均不支持 identity 校准，且长窗更差。跨 81 条路径，二阶 OLS R2 中位数由 {e3_short_second.ols_R2.median():.3f} 略升到 {e3_long_second.ols_R2.median():.3f}，正 slope 比例由 {(e3_short_second.slope>0).mean():.1%} 升到 {(e3_long_second.slope>0).mean():.1%}；但 identity R2 中位数仅由 {e3_short_second.R2.median():.3f} 变为 {e3_long_second.R2.median():.3f}，仍未达到绝对校准。"))
    add(("p", f"共同有效 trial 数范围为 {int(e3phase_results.n_valid_common.min())}-{int(e3phase_results.n_valid_common.max())}，中位数 {e3phase_results.n_valid_common.median():.0f}；最大 rectification fraction={e3phase_qc.rectification_fraction.max():.3g}。部分长窗路径的 actual Delta r 跨 mask 方差极小，使 identity R2 对很小的绝对误差高度敏感并出现极端负值；因此必须联合报告 OLS R2、slope、identity R2 与 actual Delta r 尺度，不能只看一项。"))
    add(("figure", e3phase/"figures"/"exp3_V1_PG_phase_ensemble_orders.png", "图 12e. 相位平均后的 V1 输入、PG 输出。四格比较短窗/长窗及一阶/二阶；每格标示 slope、p、OLS R2 与 identity R2。长窗保持一定线性排序，但方向和绝对尺度均未恢复。"))
    add(("figure", e3phase/"figures"/"exp3_phase_ensemble_9x9_OLS_R2.png", "图 12f. 相位平均实验 3 的 9 x 9 Delta-correlation OLS R2。二阶通常优于一阶，长窗二阶中位数略高；该指标只表示 mask 排序，不表示 identity calibration。"))
    add(("figure", e3phase/"figures"/"exp3_phase_ensemble_9x9_identity_R2.png", "图 12g. 相位平均实验 3 的 9 x 9 identity R2。格内保留原始值，色轴截断于 -2.5。长窗未产生系统性的绝对校准；极端负值对应 actual Delta r 方差非常小的路径。"))
    add(("figure", e3phase/"figures"/"exp1_exp3_identity_R2_comparison.png", "图 12h. 实验 1 correlation 与校正后实验 3 Delta correlation 的 81 路径 identity R2 分布。实验 1 长窗集中于约 0.99；实验 3 即使相位平均，长窗仍未恢复 identity。"))
    add(("p", "冲突核查后的结论是：有限记录和随机相位确实深刻影响实验 3 的 observation layer，但现有数据不支持“单纯延长窗口即可恢复结构 Delta r 的绝对尺度”。长窗可略改善跨 mask 的总体排序和方向一致率；绝对校准仍受 phase-dependent PSD feature bias、有限输入 trial 的 correlation 分母、continuous-discrete transfer mismatch 及路径特异的微小 signal variance 影响。"))
    add(("p", "长期建议包括：将结构 kernel 扩展到 P_T_obs 与 A_T_obs；分别预测 C_PA(J)、V_P(J)、V_A(J)，再由 r(J)=C_PA(J)/sqrt[V_P(J)V_A(J)] 重构；对 Fisher z=atanh(r) 做结构展开；增加 phase ensemble 与输入 trial 数；比较 Welch 与 multitaper；并避免只用单一随机相位对长短窗作机制判断。"))
    add(("conclusion_keep", f"方法学评估结论：实验 1 的 81 路径短窗 correlation identity R2 中位数约 -2.49，长窗后约 0.99，有限记录解释得到强支持。实验 3 原图 12c 的 V1->PG 长窗二阶 identity R2=0.775 可在同一seed的9 x 9扫描中复现为 {e3s9_v1pg.loc[('long','second'),'R2']:.3f}，但该固定seed下81路径的长窗二阶中位数仅 {e3s9_ls.R2.median():.3f}，说明它是路径特异结果；跨phase平均后V1->PG又变为 {e3_v1pg.loc[('long','second'),'R2']:.3f}。因此实验 3 中有限窗是重要因素，但不是结构预测残差的充分解释。"))
    add(("h2", "5.6 kernel 与简单图指标比较"))
    add(("table", ["predictor", "R2 for single-edge direct Delta C", "Pearson r"], [[name, _fmt(gb.loc[name,"R2"]), _fmt(gb.loc[name,"pearson_r"])] for name in ["structural_kernel","baseline_transfer_magnitude","edge_weight","edge_betweenness","reverse_graph_distance","target_in_degree","source_out_degree","hierarchy_difference"]]))
    add(("p", "结构 kernel 明显优于单纯 edge weight、degree、betweenness 或 hierarchy difference。它的优势来自 resolvent 路径、输入敏感性和 readout 几何，而不仅是边本身的大小。first-order failure 随 ||R Delta J|| 增大、一阶 contribution cancellation 和 edge-pair interaction 增强。全部 1500 个 random-15 网络保持稳定，且 256 个输入 trial 均有效，因此当前 held-out error 不是由 instability 或 peak switching 驱动。"))
    add(("figure", e3/"figures"/"anatomical_kernel_heatmap.png", "图 9. anatomical first-order covariance kernel。横轴 source，纵轴 target；灰格为不存在的边。"))
    add(("figure", e3/"figures"/"single_edge_delete_double.png", "图 10. 全部单边删除与加倍的 direct Delta C 对二阶预测，显示方向与幅度的非对称性。"))
    add(("figure", e3/"figures"/"incoming_outgoing_area.png", "图 11. 删除各区域全部 incoming/outgoing edges 的 direct 与二阶结果。区域级大扰动超出局部展开域。"))
    add(("figure", e3/"figures"/"random15_heldout_predictions.png", "图 12. 1200 个 held-out random-15 mask 的 observed vs first/second prediction。二阶排序和误差均改善，但 slope 仍约 2。"))
    add(("figure", e3/"figures"/"second_order_pair_heatmap.png", "图 13. top 15 edge 的 mixed second-order interaction。非零 pair terms 解释多边扰动为何不能仅由独立 edge effect 相加。"))
    add(("figure", e3/"figures"/"error_vs_resolvent_perturbation.png", "图 14. prediction error 与 max ||R Delta J||_2。resolvent-amplified perturbation 更容易离开局部展开域。"))
    add(("figure", e3/"figures"/"selected_failure_case_psd.png", "图 15. held-out 二阶误差最大的 mask：baseline 与 perturbed PSD。阴影对应 peak band 和 exponent band。"))
    add(("conclusion", f"实验 3 结论：结构 kernel 在解析 held-out 中可预测多边删除效应，并在真实九区阈值非线性时域模拟中保留部分跨 mask 排序信息。原单相位 V1->PG 长窗二阶 identity R2=0.775 不能跨 phase seed 复现；相位平均后为 {e3_v1pg.loc[('long','second'),'R2']:.3f}。因此长窗只解决部分 observation-layer 方差，尚不能证明结构 Delta r 的方向和绝对尺度得到恢复。"))

    add(("h1", "六、综合结论与后续建议"))
    add(("bullet", "输入 sampling geometry 是最强、最清楚的 coupling 来源：同一网络可因 Sigma_z 与抽样约束不同而改变相关符号。"))
    add(("bullet", "Eq.49 在当前全脑 resolvent feature map 上足够准确；高阶展开必须结合 epsilon 和 peak identity 稳定性判断，不能默认更高阶一定更好。"))
    add(("bullet", f"完整 55 区和九区阈值时域验证几乎不触发 rectification。32.768 s 长记录使实验 1 correlation slope 从0.385恢复到0.949；实验3在固定seed 220000下虽有50/81条路径的identity R2改善，但长窗二阶中位数仍为 {e3s9_ls.R2.median():.2f}，六相位复算进一步表明随机phase realization会改变Delta r的方向和尺度。"))
    add(("bullet", f"9 x 9 identity R2 对比表明窗长效应在两个实验中强度不同：实验 1 correlation 中位数从约 -2.49 恢复到 0.99；校正后实验 3 二阶中位数仅从 {e3_short_second.R2.median():.2f} 变为 {e3_long_second.R2.median():.2f}，V1->PG 反而从 {e3_v1pg.loc[('short','second'),'R2']:.2f} 降到 {e3_v1pg.loc[('long','second'),'R2']:.2f}。"))
    add(("bullet", "near-Hopf 正常形的径向统计成立，但 Eq.134 是否可观察取决于 A 对 R^2 的 sensitivity。100-1000 Hz 远离 10 Hz 峰，在这里是低灵敏度 readout。"))
    add(("bullet", "结构 kernel 比简单图指标更有解释力，二阶 edge-pair 项对 random multi-edge perturbation 必要。"))
    add(("bullet", "下一步最有价值的 sensitivity analysis 是预注册一个靠近振荡尾部的辅助 A 频带，并延长窗口/增加 seed，以测试 Eq.134 的机制是否在更可辨识的 feature 上恢复；该分析必须与当前 100-1000 Hz 主结果明确分开。"))
    add(("h2", "复现命令"))
    add(("code", "PYTHONPATH=experiments/pap_theory_validation python -m pap_validation.run --config experiments/pap_theory_validation/configs/full.yaml --experiment all\nPYTHONPATH=experiments/pap_theory_validation python -m pap_validation.run --config experiments/pap_theory_validation/configs/full.yaml --experiment finite_window\nPYTHONPATH=experiments/pap_theory_validation python -m pap_validation.pairwise_nonlinear_window_validation --config experiments/pap_theory_validation/configs/full.yaml --experiment all\nPYTHONPATH=experiments/pap_theory_validation python -m pap_validation.exp3_single_phase_12c_comparison\nPYTHONPATH=experiments/pap_theory_validation python -m pap_validation.exp3_pairwise_single_phase_12c --config experiments/pap_theory_validation/configs/full.yaml\nPYTHONPATH=experiments/pap_theory_validation python -m pap_validation.exp3_pairwise_single_phase_12c_exact --config experiments/pap_theory_validation/configs/full.yaml\nPYTHONPATH=experiments/pap_theory_validation python -m pap_validation.exp3_phase_seed_validation --config experiments/pap_theory_validation/configs/full.yaml\nPYTHONPATH=experiments/pap_theory_validation python -m pap_validation.exp3_pairwise_phase_ensemble --config experiments/pap_theory_validation/configs/full.yaml"))
    add(("p", "原始 trial、预测、condition summary、bootstrap、代表 PSD 和每张 PNG/PDF 图均位于 experiments/pap_theory_validation/outputs/。代码审计、理论映射和假设分别见 code_inventory.md、theory_mapping.md 和 assumptions.md。"))

    add(("pagebreak",))
    add(("h1", "附录 A：实验 2 全部 condition 的训练回归"))
    add(("p", "下表中的 Phi R2 为 dotPhi~mean(R2) 的拟合优度；A R2 为 A~mean(R2) 的拟合优度。方括号为 window-level OLS 95% t 区间。完整系数与 p 值见 outputs/exp2_nonlinear_hopf/condition_regression_stats.csv。"))
    for layer in ["Stuart-Landau", "smooth local E-I", "linearized local E-I"]:
        subset = r2c[r2c.layer == layer]
        rows = []
        for row in subset.itertuples(index=False):
            rows.append([
                _compact_condition(row.condition), str(int(row.n_train)), _fmt(row.phidot_R2_R2),
                f"{_fmt(row.c_i_estimate)} [{_fmt(row.c_i_ci_low)}, {_fmt(row.c_i_ci_high)}]",
                _fmt(row.A_R2_R2),
                f"{_fmt(row.kappa_R_estimate)} [{_fmt(row.kappa_R_ci_low)}, {_fmt(row.kappa_R_ci_high)}]",
            ])
        add(("h2free", layer))
        add(("smalltable", ["condition", "n", "Phi R2", "c_i [95% CI]", "A R2", "kappa_R [95% CI]"], rows))

    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / f"{basename}.md"
    pdf_path = pdf_dir / f"{basename}.pdf"
    _write_markdown(sections, md_path)
    _write_pdf(sections, pdf_path)
    shutil.copy2(pdf_path, report_dir / pdf_path.name)
    return md_path, pdf_path


def _write_markdown(sections, path: Path):
    lines=[]
    for kind,*payload in sections:
        if kind=="title": lines += ["# "+payload[0],""]
        elif kind=="subtitle": lines += ["*"+payload[0]+"*",""]
        elif kind=="h1": lines += ["## "+payload[0],""]
        elif kind in ("h2", "h2free"): lines += ["### "+payload[0],""]
        elif kind in ("p","callout","conclusion","conclusion_keep"): lines += [("**" if kind!="p" else "")+payload[0]+("**" if kind!="p" else ""),""]
        elif kind=="eq": lines += ["```text",payload[0],"```",""]
        elif kind=="code": lines += ["```bash",payload[0],"```",""]
        elif kind=="bullet": lines += ["- "+payload[0]]
        elif kind in ("table", "smalltable", "compacttable", "resulttable"):
            headers,rows=payload; lines += ["|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"]+["|"+"|".join(map(str,r))+"|" for r in rows]+[""]
        elif kind=="figure":
            img,caption=payload; lines += [f"![{caption}]({img})",caption,""]
        elif kind=="pagebreak": lines += ["<div style=\"page-break-after: always;\"></div>",""]
    path.write_text("\n".join(lines))


def _write_pdf(sections, path: Path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(TTFont("STHeiti", "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0))
    styles=getSampleStyleSheet(); font="STHeiti"
    title=ParagraphStyle("title",parent=styles["Title"],fontName=font,fontSize=20,leading=28,textColor=colors.HexColor("#163B4A"),spaceAfter=6*mm)
    subtitle=ParagraphStyle("subtitle",parent=styles["BodyText"],fontName=font,fontSize=10,leading=15,textColor=colors.HexColor("#687078"),alignment=TA_CENTER,spaceAfter=8*mm)
    h1=ParagraphStyle("h1",parent=styles["Heading1"],fontName=font,fontSize=16,leading=23,textColor=colors.HexColor("#164D60"),spaceBefore=7*mm,spaceAfter=3*mm,keepWithNext=True)
    h2=ParagraphStyle("h2",parent=styles["Heading2"],fontName=font,fontSize=12.5,leading=18,textColor=colors.HexColor("#2A6375"),spaceBefore=4*mm,spaceAfter=2*mm,keepWithNext=True)
    h2free=ParagraphStyle("h2free",parent=h2,keepWithNext=False)
    body=ParagraphStyle("body",parent=styles["BodyText"],fontName=font,fontSize=9.1,leading=15,spaceAfter=2.2*mm,textColor=colors.HexColor("#202428"))
    cap=ParagraphStyle("cap",parent=body,fontSize=8.2,leading=12,textColor=colors.HexColor("#555D63"),spaceBefore=2*mm,spaceAfter=4*mm)
    eq=ParagraphStyle("eq",parent=body,fontName="Courier",fontSize=7.2,leading=10.5,leftIndent=3*mm,rightIndent=3*mm,backColor=colors.HexColor("#F2F5F6"),borderPadding=5,spaceBefore=1*mm,spaceAfter=3*mm)
    callout=ParagraphStyle("callout",parent=body,leftIndent=4*mm,rightIndent=4*mm,backColor=colors.HexColor("#EAF3F5"),borderColor=colors.HexColor("#4E8798"),borderWidth=1,borderPadding=7,spaceBefore=2*mm,spaceAfter=4*mm)
    bullet=ParagraphStyle("bullet",parent=body,leftIndent=6*mm,firstLineIndent=-3*mm)
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=17*mm,rightMargin=17*mm,topMargin=16*mm,bottomMargin=16*mm,title="周期-非周期耦合理论数值实验详细总结",author="Codex")
    story=[]
    for kind,*payload in sections:
        if kind=="title": story.append(Paragraph(escape(payload[0]),title))
        elif kind=="subtitle": story.append(Paragraph(escape(payload[0]),subtitle))
        elif kind=="h1": story.append(Paragraph(escape(payload[0]),h1))
        elif kind=="h2": story.append(Paragraph(escape(payload[0]),h2))
        elif kind=="h2free": story.append(Paragraph(escape(payload[0]),h2free))
        elif kind=="p": story.append(Paragraph(escape(payload[0]),body))
        elif kind in ("callout","conclusion"): story.append(Paragraph(escape(payload[0]),callout))
        elif kind=="conclusion_keep": story.append(KeepTogether([Paragraph(escape(payload[0]),callout)]))
        elif kind in ("eq","code"): story.append(Paragraph(escape(payload[0]),eq))
        elif kind=="bullet": story.append(Paragraph("• "+escape(payload[0]),bullet))
        elif kind in ("table", "smalltable", "compacttable", "resulttable"):
            headers,rows=payload; data=[[Paragraph(escape(str(v)),body) for v in headers]]+[[Paragraph(escape(str(v)),body) for v in row] for row in rows]
            if kind == "smalltable":
                small=ParagraphStyle("small",parent=body,fontSize=6.2,leading=8.2,spaceAfter=0)
                data=[[Paragraph(escape(str(v)),small) for v in headers]]+[[Paragraph(escape(str(v)),small) for v in row] for row in rows]
                widths=[35*mm,11*mm,25*mm,39*mm,25*mm,39*mm]
                padding=2
            elif kind == "compacttable":
                compact=ParagraphStyle("compact",parent=body,fontSize=8.2,leading=11.2,spaceAfter=0)
                data=[[Paragraph(escape(str(v)),compact) for v in headers]]+[[Paragraph(escape(str(v)),compact) for v in row] for row in rows]
                widths=[35*mm,35*mm,104*mm]
                padding=3
            elif kind == "resulttable":
                result_style=ParagraphStyle("resulttable",parent=body,fontSize=7.2,leading=9.2,spaceAfter=0)
                data=[[Paragraph(escape(str(v)),result_style) for v in headers]]+[[Paragraph(escape(str(v)),result_style) for v in row] for row in rows]
                widths=[39*mm,23*mm,27*mm,28*mm,28*mm,29*mm]
                padding=2
            else:
                widths=[(A4[0]-34*mm)/len(headers)]*len(headers); padding=5
            tab=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
            tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DDEBF0")),("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#163B4A")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#AAB4B8")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),padding),("RIGHTPADDING",(0,0),(-1,-1),padding),("TOPPADDING",(0,0),(-1,-1),padding),("BOTTOMPADDING",(0,0),(-1,-1),padding)])); story += [tab,Spacer(1,3*mm)]
        elif kind=="figure":
            img,caption=payload
            if not Path(img).exists(): continue
            flow=Image(str(img)); maxw,maxh=174*mm,112*mm; scale=min(maxw/flow.imageWidth,maxh/flow.imageHeight)
            flow.drawWidth=flow.imageWidth*scale; flow.drawHeight=flow.imageHeight*scale; flow.hAlign="CENTER"
            story += [Spacer(1,2*mm),flow,Paragraph(escape(caption),cap)]
        elif kind=="pagebreak": story.append(PageBreak())
    def decorate(canvas,doc):
        canvas.saveState(); canvas.setFont(font,7.5); canvas.setFillColor(colors.HexColor("#6D7478")); canvas.drawString(17*mm,8*mm,"p-ap coupling numerical validation - detailed report"); canvas.drawRightString(A4[0]-17*mm,8*mm,str(doc.page)); canvas.restoreState()
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default=str(BASE/"outputs")); p.add_argument("--report-dir",default=str(BASE/"reports")); p.add_argument("--pdf-dir",default=str(BASE/"output/pdf")); p.add_argument("--basename",default="detailed_experiment_summary_zh_finite_window"); a=p.parse_args()
    print("\n".join(map(str,build(Path(a.output_dir),Path(a.report_dir),Path(a.pdf_dir),a.basename))))


if __name__=="__main__": main()
