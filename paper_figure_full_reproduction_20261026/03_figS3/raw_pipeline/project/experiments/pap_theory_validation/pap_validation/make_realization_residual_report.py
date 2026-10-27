from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .make_detailed_report import BASE, _write_markdown, _write_pdf


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def build() -> tuple[Path, Path]:
    source = BASE / "outputs" / "realization_residual_validation"
    exp2 = BASE / "outputs" / "exp2_window_second_order_validation"
    report_dir = BASE / "reports"
    pdf_dir = BASE / "output" / "pdf"
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((source / "manifest.json").read_text())
    exp1 = pd.read_csv(source / "exp1_condition_results.csv.gz")
    exp1_metrics = pd.read_csv(source / "exp1_9x9_metrics.csv")
    exp1_qc = pd.read_csv(source / "exp1_explicit_recurrence_qc.csv")
    exp2_metrics = pd.read_csv(exp2 / "second_order_prediction_metrics.csv").set_index("prediction")
    exp3 = pd.read_csv(source / "exp3_68edge_long_results.csv").set_index(["network", "order"])
    exp3_qc = pd.read_csv(source / "exp3_explicit_recurrence_qc.csv")

    sections: list[tuple] = []
    add = sections.append
    add(("title", "将同条件 realization 误差纳入 P-A coupling 理论的数值验证"))
    add(("subtitle", "独立补充报告：Experiment 1、2、3 的 calibration/held-out 检验"))
    add(("p", "本报告是新建文件，不覆盖此前的 three_experiment_summary_zh.pdf。核心问题是：理论给出由输入参数或网络结构变化产生的 latent peak-frequency P 与 aperiodic exponent A，而有限记录中即使输入参数和网络结构完全相同，不同 Fourier phase realization 仍会使观测到的 P 与 A 波动。若忽略这一层，理论与单次真实 simulation 的 correlation 不是同一个统计对象。"))

    add(("h1", "一、统一的 observation-layer 模型"))
    add(("p", "对 trial i 和 realization q，定义二维 feature 向量 F=(P,A)。理论预测的是 latent feature mu_i；有限窗谱估计给出的观测为："))
    add(("eq", "F_iq(obs) = mu_i(theory) + epsilon_iq(realization)"))
    add(("p", "在每个 condition 内使用3个独立 calibration phase。先对同一 trial 跨phase中心化，再池化估计 realization covariance。该操作不会把不同输入参数造成的 signal variance 混入误差项："))
    add(("eq", "epsilon_iq = F_iq - mean_q(F_iq)\nSigma_eps = sum_i sum_q epsilon_iq epsilon_iq' / [N (Q-1)]"))
    add(("p", "理论总协方差和修正后的相关性为："))
    add(("eq", "Sigma_total = Sigma_signal + Sigma_eps\nr_corrected = Sigma_total(P,A) / sqrt[Sigma_total(P,P) Sigma_total(A,A)]"))
    add(("p", "另用5个完全独立的 held-out phase 验证。主目标是5个单-realization correlation的均值，因为它与将Sigma_eps加入理论后的预测对象一致；同时报告先跨phase平均feature后再求相关性的phase-ensemble结果，后者的realization方差会按平均次数下降，不能与单-realization预测混用。所有分析只使用理论、一阶、二阶以及8个phase都可测的冻结共同trial集合。"))

    add(("h1", "二、Experiment 1：完整55脑区 baseline 网络"))
    add(("h2", "2.1 方法"))
    add(("p", f"使用完整55脑区E-I threshold-linear网络；输入和读取脑区均从{', '.join(manifest['exp1']['input_areas'])}中选择，共9 x 9=81条路径。每条路径有{manifest['exp1']['conditions']}个输入分布condition，每个condition {manifest['exp1']['trials_per_condition']}个参数trial。输入参数z=(bump中心、bump幅度、beta、aperiodic幅度)，输入PSD为背景幂律与Gaussian bump之和。"))
    add(("eq", "S_in(f;z) = a_noise (f/10)^beta + a_bump exp[-(f-f0)^2/(2 width^2)]"))
    add(("p", f"每个trial先生成同一目标PSD但随机phase不同的时序，再驱动完整55区网络。时间步长{manifest['exp1']['dt_ms']} ms，burn-in 2.048 s，分析窗{manifest['exp1']['analysis_s']} s；PSD使用4.096 s Hann-Welch段、50%重叠，并对log PSD平滑。P来自5-20 Hz最高内部峰，A为100-1000 Hz log-log回归的正exponent。模型在该regime几乎不触及阈值，因此主扫描使用与Euler递推严格对应的离散transfer加速，并用显式55区递推作QC。"))
    add(("p", "一阶和二阶latent feature分别为："))
    add(("eq", "F_1(u)=F(0)+G u\nF_2(u)=F(0)+G u + 1/2 [u' H_P u, u' H_A u]'"))
    add(("p", "对每个input-output-condition分别从3个calibration realization估计Sigma_eps，再将其加入该condition的一阶或二阶signal covariance。最终对14个condition比较理论r与5个held-out simulation r的均值，并分别计算OLS R2、slope、RMSE和identity R2。identity R2以y=x为固定模型，不允许通过重新缩放掩盖校准偏差。"))

    add(("h2", "2.2 结果"))
    for prediction, label in (
        ("first_latent_r", "一阶latent"),
        ("first_residual_corrected_r", "一阶+realization"),
        ("second_latent_r", "二阶latent"),
        ("second_residual_corrected_r", "二阶+realization"),
    ):
        subset = exp1_metrics[exp1_metrics.prediction == prediction]
        add(("bullet", f"{label}：81条路径identity R2中位数={_fmt(subset.identity_R2.median())}，OLS R2中位数={_fmt(subset.ols_R2.median())}，slope中位数={_fmt(subset.slope.median())}，RMSE中位数={_fmt(subset.RMSE.median())}。"))
    pair = exp1_metrics[(exp1_metrics.input_area == "V1") & (exp1_metrics.output_area == "PG")].set_index("prediction")
    add(("resulttable", ["V1 input -> PG output", "identity R2", "OLS R2", "slope", "RMSE"], [
        [label, _fmt(pair.loc[pred, "identity_R2"]), _fmt(pair.loc[pred, "ols_R2"]), _fmt(pair.loc[pred, "slope"]), _fmt(pair.loc[pred, "RMSE"])]
        for pred, label in (
            ("first_latent_r", "一阶latent"),
            ("first_residual_corrected_r", "一阶+realization"),
            ("second_latent_r", "二阶latent"),
            ("second_residual_corrected_r", "二阶+realization"),
        )
    ]))
    add(("figure", source / "figures" / "exp1_realization_residual_validation.png", "图 R1. Experiment 1 realization-residual验证。前四图以V1输入、PG读取为代表，比较未修正与加入Sigma_eps后的一阶、二阶理论；后两图展示81条输入-输出路径的修正后identity R2。颜色表示输入扰动幅度epsilon。"))
    add(("p", f"显式55区Euler递推QC中rectification count为{int(exp1_qc.rectification_count_all_states.max())}；timeseries r最低为{exp1_qc.timeseries_r.min():.6f}，relative RMSE最高约{100 * exp1_qc.relative_rmse.max():.1f}%。因此离散传递函数加速保留了该regime的完整网络线性递推，剩余小差异在报告中保留为数值误差边界。"))
    corrected_gain_first = (
        exp1_metrics[exp1_metrics.prediction == "first_residual_corrected_r"].set_index(["input_area", "output_area"]).identity_R2
        - exp1_metrics[exp1_metrics.prediction == "first_latent_r"].set_index(["input_area", "output_area"]).identity_R2
    )
    corrected_gain_second = (
        exp1_metrics[exp1_metrics.prediction == "second_residual_corrected_r"].set_index(["input_area", "output_area"]).identity_R2
        - exp1_metrics[exp1_metrics.prediction == "second_latent_r"].set_index(["input_area", "output_area"]).identity_R2
    )
    add(("conclusion_keep", f"Experiment 1结论：显式加入同条件realization covariance后，一阶identity R2在{(corrected_gain_first > 0).mean():.1%}的路径改善，中位改变量{_fmt(corrected_gain_first.median())}；二阶在{(corrected_gain_second > 0).mean():.1%}的路径改善，中位改变量{_fmt(corrected_gain_second.median())}。这检验的是单-realization correlation，而不是phase平均后的低噪声极限。二阶是否优于一阶仍由input Taylor非线性决定，realization层修正与Taylor阶数是两个独立问题。"))

    add(("h1", "三、Experiment 2：新版稳定滤波器 P-A correlation"))
    add(("h2", "3.1 方法与原图"))
    add(("p", "Experiment 2新版程序原本就严格采用本报告的方法：每个S、sigma_beta、floor ratio和input sigma condition有256个参数trial；用3个独立phase估计condition-specific realization residual covariance，用5个独立phase作held-out；时间步长0.25 ms、burn-in 2.048 s、分析窗32.768 s、Welch段4.096 s。因此本部分不重复随机模拟，完全复用已冻结的原图和数据，避免不必要的Monte Carlo漂移。"))
    add(("p", "理论层比较exact mapping、beta局部一阶、beta二阶，以及分别加Sigma_eps后的预测。它直接回答：稳定滤波器由beta变化产生的latent P-A coupling，在有限窗measurement noise存在时能否预测真实simulation correlation。"))
    add(("figure", exp2 / "figures" / "exp2_second_order_identity.png", "图 R2. Experiment 2原始新版结果。各线性回归图标示回归系数和p值；比较exact、一阶、二阶及其realization-residual修正与5-phase held-out nonlinear simulation。该图未重算。"))
    table_rows = []
    for pred in ("exact_mapping_r", "first_order_r", "second_order_r", "first_order_residual_r", "second_order_residual_r"):
        if pred not in exp2_metrics.index:
            continue
        row = exp2_metrics.loc[pred]
        table_rows.append([pred, str(int(row["n"])), _fmt(row["R2"]), _fmt(row["pearson_r"]), _fmt(row["slope"]), _fmt(row["RMSE"])])
    add(("resulttable", ["Experiment 2 prediction", "n", "identity R2", "Pearson", "slope", "RMSE"], table_rows))
    add(("conclusion_keep", "Experiment 2结论：这一实验已经证明，仅有latent beta->(P,A)映射并不足以预测有限窗simulation；condition-specific Sigma_eps是统计对象匹配所必需的组成。若修正后仍有偏差，则应归因于条件均值映射偏差、peak switching/invalidity或非加性误差，而不能再笼统归咎于phase realization。"))

    add(("h1", "四、Experiment 3：68条重要路由边删除"))
    add(("h2", "4.1 方法"))
    add(("p", f"输入固定在{manifest['exp3']['input_area']}，读取{manifest['exp3']['output_area']}。删边网络删除V2除A6Va外的全部入边，并删除A6Va除V2外的全部出边，保留A6Va->V2，共{manifest['exp3']['n_deleted_edges']}条。4096个输入参数trial来自冻结的四维相关Gaussian，epsilon={manifest['exp3']['epsilon']}。本次把分析窗从此前32.768 s再延长到{manifest['exp3']['analysis_s']} s；dt={manifest['exp3']['dt_ms']} ms、burn-in 2.048 s、Welch段仍为4.096 s。"))
    add(("p", "对baseline和deleted网络分别估计Sigma_eps，并分别把误差层加入一阶、二阶和exact-resolvent latent covariance；最后再取相关性差值："))
    add(("eq", "r_base,corr = Corr(Sigma_signal,base + Sigma_eps,base)\nr_del,corr = Corr(Sigma_signal,del + Sigma_eps,del)\nDelta r_corr = r_del,corr - r_base,corr"))
    add(("p", "这里不能直接把两个Sigma_eps相减后塞入一次correlation，因为correlation是协方差矩阵的非线性函数。baseline与deleted使用配对的held-out phase，因此Delta r的realization SD由每个phase内r_del-r_base直接估计。"))

    add(("h2", "4.2 结果"))
    table_rows = []
    for order, label in (("first", "一阶"), ("second", "二阶"), ("exact", "exact resolvent")):
        table_rows.append([
            label,
            _fmt(exp3.loc[("baseline", order), "latent_r"], 5),
            _fmt(exp3.loc[("baseline", order), "residual_corrected_r"], 5),
            _fmt(exp3.loc[("deleted", order), "latent_r"], 5),
            _fmt(exp3.loc[("deleted", order), "residual_corrected_r"], 5),
            _fmt(exp3.loc[("deleted_minus_baseline", order), "residual_corrected_r"], 5),
        ])
    table_rows.append([
        "held-out simulation",
        "-", _fmt(exp3.loc[("baseline", "exact"), "heldout_r_mean"], 5),
        "-", _fmt(exp3.loc[("deleted", "exact"), "heldout_r_mean"], 5),
        _fmt(exp3.loc[("deleted_minus_baseline", "exact"), "heldout_r_mean"], 5),
    ])
    add(("resulttable", ["68边结果", "base latent", "base + eps", "deleted latent", "deleted + eps", "Delta corrected"], table_rows))
    add(("figure", source / "figures" / "exp3_realization_residual_validation.png", "图 R3. 65.536 s长窗下的68边Experiment 3。前三图比较latent、加入realization covariance后的一阶/二阶/exact理论与独立held-out simulation；右下显示5个held-out realization的baseline、deleted和paired Delta r。阴影为held-out realization间SD。"))
    add(("pagebreak",))
    add(("resulttable", ["显式递推QC", "rectification count", "timeseries r", "relative RMSE", "max abs error"], [
        [row.network, str(int(row.rectification_count)), _fmt(row.timeseries_r, 8), f"{row.relative_rmse:.2e}", f"{row.max_abs_error:.2e}"]
        for row in exp3_qc.itertuples(index=False)
    ]))
    old = pd.read_csv(BASE / "outputs" / "exp3_emergent_second_order_validation" / "correlation_validation.csv").set_index(["method", "network"])
    add(("p", f"旧32.768 s六相位ensemble simulation的Delta r={old.loc[('simulation','deleted_minus_baseline'),'r']:.5f}；本次65.536 s下5个独立单-realization Delta r均值={exp3.loc[('deleted_minus_baseline','exact'),'heldout_r_mean']:.5f}，realization间SD={exp3.loc[('deleted_minus_baseline','exact'),'heldout_r_sd']:.5f}。两者统计对象不同，因此本报告不把数值相等当作必要条件；真正的检验是修正后的理论是否落在独立单-realization均值及其波动尺度附近。"))
    add(("conclusion_keep", "Experiment 3结论：结构效应必须在baseline与deleted两个网络中各自完成observation-layer修正，再做Delta r。延长窗长降低了realization方差，但不会自动消除两网络共同的系统性feature bias；若exact+Sigma_eps仍偏离held-out Delta，剩余问题更可能是条件依赖的有限窗均值偏差、peak检测非线性或baseline/deleted误差的配对结构，而不是Taylor展开阶数。"))

    add(("h1", "五、总讨论"))
    add(("bullet", "此前理论预测的是latent parameter/structure-induced covariance；真实有限窗simulation测得的是latent covariance与同条件realization covariance之和。两者必须显式对齐。"))
    add(("bullet", "Sigma_eps必须在每个condition内、按trial跨realization中心化估计。直接把所有trial flatten后求协方差会把输入参数信号混入误差，并且使用3N而非N(Q-1)会产生有偏估计。"))
    add(("bullet", "calibration和held-out phase必须独立。否则理论修正与验证共享噪声，会夸大拟合效果。"))
    add(("bullet", "单-realization correlation与phase-ensemble correlation是不同estimand。前者使用Sigma_signal+Sigma_eps；若平均M个独立realization，理想情况下误差项近似变为Sigma_eps/M。"))
    add(("bullet", "一阶/二阶处理input-to-feature映射的局部非线性；realization residual处理observation noise。二者可以同时改善，也可以只改善其中一个，不能互相替代。"))
    add(("bullet", "对Experiment 3做差时，小Delta r仍可能是病态统计量。加入realization covariance改善了点预测的统计对象匹配，但稳健性仍应由held-out phase间SD和冻结共同trial集合判断。"))

    md_path = report_dir / "realization_residual_validation_summary_zh.md"
    pdf_path = pdf_dir / "realization_residual_validation_summary_zh.pdf"
    _write_markdown(sections, md_path)
    _write_pdf(sections, pdf_path)
    return md_path, pdf_path


if __name__ == "__main__":
    print("\n".join(map(str, build())))
