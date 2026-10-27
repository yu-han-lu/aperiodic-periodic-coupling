from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = ROOT / "experiments" / "pap_theory_validation"
SOURCE = EXP_ROOT / "outputs" / "paper_figure_node_deletion"
OUTPUT = ROOT / "output" / "pdf" / "paper_figure_node_deletion_summary_zh_s32_longwindow131.pdf"
REPORT = EXP_ROOT / "reports" / "paper_figure_node_deletion_summary_zh_s32_longwindow131.txt"
FONT_NAME = "ArialUnicodeEmbedded"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")


def _styles():
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyZH",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=9.2,
        leading=14,
        textColor=colors.HexColor("#20242A"),
        spaceAfter=5,
    )
    title = ParagraphStyle(
        "TitleZH",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=19,
        leading=25,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#15324A"),
    )
    h1 = ParagraphStyle(
        "H1ZH",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=14,
        leading=19,
        textColor=colors.HexColor("#15324A"),
        spaceBefore=7,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2ZH",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=11.2,
        leading=16,
        textColor=colors.HexColor("#8C2F39"),
        spaceBefore=5,
        spaceAfter=4,
    )
    caption = ParagraphStyle(
        "CaptionZH",
        parent=body,
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#4D5660"),
        spaceBefore=3,
        spaceAfter=7,
    )
    small = ParagraphStyle("SmallZH", parent=body, fontSize=7.6, leading=10.5)
    return body, title, h1, h2, caption, small


def _image(path: Path, max_width: float, max_height: float):
    with PILImage.open(path) as im:
        width, height = im.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _table(rows, widths, font_size=8):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 3),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8EF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#15324A")),
                ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#AAB4BD")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7F8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_NAME, 7.5)
    canvas.setFillColor(colors.HexColor("#66717A"))
    canvas.drawString(18 * mm, 10 * mm, "P-A coupling numerical validation | independent summary")
    canvas.drawRightString(192 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def _source_index():
    names = [
        "A_baseline_input_output_psd",
        "B_exp1_V1_PG_theory_simulation",
        "C_exp1_9x9_correlation_matrix",
        "D_exp2_eigenvalue_transitions",
        "E_exp2_three_regime_psd",
        "F_exp2_nearcritical_input_output_psd",
        "G_exp2_1000trial_theory_simulation",
        "H_exp3_baseline_deleted_J_matrices",
        "I_exp3_node_deletion_correlations",
    ]
    rows = []
    for name in names:
        rows.append((name.split("_")[0], str((SOURCE / "panels" / f"{name}.png").resolve()), str((SOURCE / "panels" / f"{name}.pdf").resolve())))
    rows.append(("总图", str((SOURCE / "figures" / "paper_figure_A_to_I.png").resolve()), str((SOURCE / "figures" / "paper_figure_A_to_I.pdf").resolve())))
    index = pd.DataFrame(rows, columns=["panel", "png_path", "pdf_path"])
    index.to_csv(SOURCE / "source_image_index.csv", index=False)
    return index


def build():
    body, title, h1, h2, caption, small = _styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    source_index = _source_index()
    exp3 = pd.read_csv(SOURCE / "exp3_node_deletion_long_results_robust_v3_corrected.csv").set_index("network")
    dense_source = SOURCE.parent / "paper_figure_dense_revision"
    exp1_dense = pd.read_csv(dense_source / "exp1_dense_conditions.csv")
    exp2_source = SOURCE.parent / "exp2_pap_s32_window_131p072"
    exp2 = pd.read_csv(exp2_source / "selected_1000_condition_results_rlt0p6.csv")
    pilot_65 = json.loads((SOURCE.parent / "exp2_pap_s32_window_65p536" / "metrics_100.json").read_text())
    pilot_131 = json.loads((exp2_source / "metrics_100.json").read_text())
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    v6 = json.loads((SOURCE / "exp3_selected_observation_matched_v6.json").read_text())
    v9 = pd.read_csv(SOURCE / "exp3_observation_matched_nullspace_refinement_v9.csv")

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
        title="全脑模型周期-非周期耦合数值实验图与结果总结",
        author="Codex",
    )
    story = []
    story.append(Paragraph("全脑模型周期-非周期耦合：论文图与数值结果总结", title))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "本文件是独立于既有 summary 的新报告，汇总 baseline PSD、Experiment 1 的 1000 条件长窗验证、Experiment 2 的局部 E-I 临界扫描与 S32 残差修正方法的 1000 条件验证，以及 Experiment 3 的节点删除复核。所有最终 panel、总图、数值表和筛选审计均保存在结果目录。",
        body,
    ))
    story.append(Paragraph("关键结论", h1))
    conclusions = [
        "Experiment 1：V1 input、PG readout 的 1000 个输入条件中，每条件模拟 64 trial；一阶理论加 realization covariance 后覆盖 -0.956 到 0.950，simulation 覆盖 -0.964 到 0.962，identity R²=0.9945。",
        "Experiment 2：I-to-E 增益约 2.082 时首先出现稳定复特征值；固定该增益后，E-to-E 增益约 4.339 才发生 Hopf crossing。复模态出现与 Hopf 失稳不是同一临界点。",
        f"Experiment 2：100-condition pilot 中，65.536 s 与 131.072 s 的 identity R² 分别为 {pilot_65['identity_R2']:.3f} 和 {pilot_131['identity_R2']:.3f}，因此最终 G 使用 131.072 s。正式结果只纳入实际 residual-corrected theory r<0.6 的 1000 个 condition；其 identity R²={manifest['panel_G']['identity_R2']:.3f}，Pearson r={manifest['panel_G']['pearson_r']:.3f}。",
        "Experiment 3：复核发现旧结果表在组装 signal、realization error 与 signal-error cross covariance 时使用了过时公式。以同一份 65.536 s 缓存按完整公式重算后，baseline 和 node-deleted 的理论值均与 held-out simulation 接近，且 Δr 约 0.10。",
        "进一步 observation-matched 与零空间筛选显示：理论上可得到 baseline 小、deleted r 大的组合，但其删节点后 peak SD 仅约 0.06-0.10 Hz；放大至可观测水平时 baseline correlation 同时升高。当前参数和 feature 定义下未找到通过独立 holdout 的强 emergent-correlation 节点删除 regime。",
    ]
    for item in conclusions:
        story.append(Paragraph("• " + item, body))
    story.append(_image(SOURCE / "figures" / "paper_figure_A_to_I.png", 178 * mm, 130 * mm))
    story.append(Paragraph("总图 A-I。图 I 已用完整 signal + error + signal-error cross covariance 公式重算；它验证了该节点删除条件下理论与长窗 held-out simulation 的一致性。", caption))

    story.append(PageBreak())
    story.append(Paragraph("1. 模型、输入与共同分析流程", h1))
    story.append(Paragraph(
        "全模型包含 55 个皮层脑区，每区有 E/I 两个状态，因此 Jacobian J 为 110×110。长程 FLN 只由各区 E population 发出，并投射到目标区 E/I population；局部 E-I 参数由标准 marmoset 模型给定。输入注入指定脑区 E population。",
        body,
    ))
    rows = [
        ["项目", "设定"],
        ["输入谱", "Q(f;z)=a_ap(f/10)^β+a_b exp[-(f-f0)^2/(2·2^2)]"],
        ["输入参数中心", "z0=(f0=10 Hz, a_b=6, β=-0.5, a_ap=1)"],
        ["峰频", "在指定 peak band 内对平滑 log PSD 的内部局部极大值进行识别"],
        ["非周期斜率", "对 100-1000 Hz 的 log PSD-log f 做线性回归；图中标示回归斜率"],
        ["时间步长", "0.25 ms"],
        ["长窗", "Experiment 1: 32.768 s；Experiment 3: 65.536 s/realization"],
        ["误差层", "3 个 calibration realization 估计；5 个 held-out realization 独立验证"],
    ]
    story.append(_table(rows, [42 * mm, 132 * mm]))
    story.append(Paragraph("Panel A：baseline 输入与输出 PSD", h2))
    story.append(_image(SOURCE / "panels" / "A_baseline_input_output_psd.png", 176 * mm, 72 * mm))
    story.append(Paragraph(
        "输入由 10 Hz 附近 bump 与 β=-0.5 的 aperiodic background 叠加。两图的黑色 input/exact PSD 均为粗虚线；左图浅蓝粗实线是 aperiodic fit，右图同色更粗实线是 100-1000 Hz slope fit。V1 注入、PG readout 的代表性输出在 8.52 Hz 捕捉到峰，拟合斜率为 -6.42。",
        caption,
    ))

    story.append(PageBreak())
    story.append(Paragraph("2. Experiment 1：输入变化与 55 脑区长窗验证", h1))
    story.append(Paragraph(
        "固定完整 55 脑区网络，输入 V1、读取 PG。新 Panel B 构造 1000 个输入条件，每条件 64 trial；bump center 与 beta 的样本相关从 -0.995 到 0.995 等距扫描，epsilon=0.2。每条轨迹使用 32.768 s 长窗。simulation 由完整 55 区 Jacobian 的精确离散传递函数生成，该加速器此前已与显式递推逐点核验；一阶预测加入既有 3-phase calibration pool 估计的 realization covariance。",
        body,
    ))
    story.append(_image(SOURCE / "panels" / "B_exp1_V1_PG_theory_simulation.png", 85 * mm, 76 * mm))
    story.append(Paragraph(
        "Panel B：V1 input、PG readout，仅展示一阶理论。1000 个条件在相关轴上近似均匀铺开；每条件模拟 64 trial。红色虚线为 identity，identity R²=0.9945；OLS 线已从图中移除。",
        caption,
    ))
    story.append(_image(SOURCE / "panels" / "C_exp1_9x9_correlation_matrix.png", 100 * mm, 86 * mm))
    story.append(Paragraph(
        "Panel C：9 个输入区×9 个 readout 区的长窗 held-out phase-ensemble correlation。这里恢复为此前使用的 rank-one、epsilon=0.2 条件；每条轨迹分析窗为 32.768 s，5 个 held-out realization 先对 feature 求均值后计算 Pearson r。81 格范围为 0.975-0.996，均达到 p<0.05。此前约 0.6 的版本并非窗长不足，而是误用了 correlated-Gaussian、rho=-0.6 条件。",
        caption,
    ))

    story.append(PageBreak())
    story.append(Paragraph("3. Experiment 2：复模态、Hopf 与 P-A correlation", h1))
    story.append(Paragraph(
        "局部非线性 E-I 系统分两阶段扫描。阶段 1 固定 E-to-E=1，只改变 I-to-E multiplier；阶段 2 将 I-to-E 固定在目标值 6.694，再改变 E-to-E multiplier。特征值由局部 Jacobian 直接计算，复特征值出现条件为判别式跨零，Hopf 条件为主导特征值实部跨零。",
        body,
    ))
    story.append(_image(SOURCE / "panels" / "D_exp2_eigenvalue_transitions.png", 176 * mm, 70 * mm))
    story.append(Paragraph(
        "Panel D：稳定复模态在 s_EI≈2.082 出现；Hopf crossing 在 s_EE≈4.339。前者只表示共轭复极点形成，后者才表示稳定性边界。",
        caption,
    ))
    story.append(_image(SOURCE / "panels" / "E_exp2_three_regime_psd.png", 176 * mm, 48 * mm))
    story.append(Paragraph(
        "Panel E：同一 aperiodic input 下的三个 regime。实特征值与稳定复特征值条件均没有形成明确的低频输出峰；越过 Hopf 后输出出现约 22.7 Hz 峰。",
        caption,
    ))
    story.append(_image(SOURCE / "panels" / "F_exp2_nearcritical_input_output_psd.png", 176 * mm, 70 * mm))
    story.append(Paragraph(
        "Panel F：在稳定复特征值（stable-complex）条件，输入仅含 aperiodic component，输出出现约 25.9 Hz 的网络峰。右图以浅蓝粗实线标示 100-1000 Hz fitted slope，黑色虚线为 stable-complex PSD。",
        caption,
    ))

    story.append(PageBreak())
    story.append(Paragraph("4. Experiment 2：S32 方法的 1000 条件理论-模拟匹配", h1))
    story.append(Paragraph(
        "这里严格沿用原图 S32 右图的统计定义，但将有效散点扩展到 1000 个参数 condition。pilot 使用原参数立方体；正式运行根据 pilot 对候选进行低-r 富集，范围为 S=4.5-6.694、sigma_beta=0.05-0.25、white-floor ratio=0.30-0.50，并对每个 condition 完成真实 nonlinear simulation 后施加 residual-corrected theory r<0.6 的硬筛选。input sigma 固定为 0.02。每个 condition 内生成 256 个 beta trial，beta 服从均值 0、标准差 sigma_beta、截断于 [-0.8,0.8] 的正态分布。",
        body,
    ))
    story.append(Paragraph(
        "非线性 E-I 网络使用 tanh 激活和 Heun 积分，dt=0.25 ms，burn-in=2.048 s。先对 65.536 s 和 131.072 s 各做 100-condition pilot，并按 identity R² 选择 131.072 s 用于最终 1000-condition 模拟。PSD 使用 4.096 s Hann-Welch 窗与 50% overlap；峰频在 5-100 Hz 识别，aperiodic slope 在 100-1000 Hz 拟合。每个 condition 使用 3 个 calibration realization 估计 feature-level residual covariance，再将它加到理论 signal covariance；横轴是由总 covariance 换算的 residual-corrected theory r，纵轴是 5 个独立 held-out nonlinear realization 中 Pearson r 的平均值。",
        body,
    ))
    pilot_rows = [
        ["analysis window", "conditions", "identity R²", "Pearson r"],
        ["65.536 s", "100", f"{pilot_65['identity_R2']:.3f}", f"{pilot_65['pearson_r']:.3f}"],
        ["131.072 s", "100", f"{pilot_131['identity_R2']:.3f}", f"{pilot_131['pearson_r']:.3f}"],
    ]
    story.append(_table(pilot_rows, [45*mm, 38*mm, 45*mm, 45*mm], font_size=8))
    story.append(_image(SOURCE / "panels" / "G_exp2_1000trial_theory_simulation.png", 100 * mm, 75 * mm))
    story.append(Paragraph(
        f"Panel G：131.072 s、residual-corrected theory r<0.6 的 1000 个 condition-level 散点。identity R²={manifest['panel_G']['identity_R2']:.3f}，OLS R²={manifest['panel_G']['ols_R2']:.3f}，Pearson r={manifest['panel_G']['pearson_r']:.3f}，回归 p={manifest['panel_G']['p']:.3g}。红色虚线是 identity，黑色实线是自由截距 OLS；这些指标仅写在正文，图内不再重复。",
        caption,
    ))
    exp2_rows = [["量", "minimum", "median", "maximum"]]
    for column, label in (
        ("S", "I-to-E multiplier S"),
        ("sigma_beta", "SD(beta)"),
        ("floor_ratio", "white-floor ratio"),
        ("n_valid_common", "common valid trials"),
        ("residual_corrected_r", "theory+residual r"),
        ("nonlinear_heldout_r_mean", "held-out nonlinear r"),
    ):
        values = exp2[column]
        exp2_rows.append([label, f"{values.min():.3f}", f"{values.median():.3f}", f"{values.max():.3f}"])
    story.append(_table(exp2_rows, [55*mm, 38*mm, 38*mm, 38*mm], font_size=7.5))

    story.append(PageBreak())
    story.append(Paragraph("5. Experiment 3：节点删除、J 矩阵与 held-out 复核", h1))
    story.append(Paragraph(
        "节点删除定义为将指定区域在长程 FLN/J 中的全部 in-degree 与 out-degree 置零，局部 E-I block 保留。Panel H/I 使用 v3 候选：删除 A46D、A47L、A4c、A6DC、A6DR、AuA1、MIP、OPt、PE、PG、V1、V4T；input=A6Va，readout=V2。最终长窗使用 512 个 trial、每 trial 65.536 s、3 calibration+5 held-out realization。",
        body,
    ))
    story.append(_image(SOURCE / "panels" / "H_exp3_baseline_deleted_J_matrices.png", 176 * mm, 78 * mm))
    story.append(Paragraph(
        "Panel H：55×55 长程 E-to-E coupling block。右图中灰色表示因节点删除而被置零的原非零连接；彩色部分是保留的连接。",
        caption,
    ))
    story.append(_image(SOURCE / "panels" / "I_exp3_node_deletion_correlations.png", 112 * mm, 88 * mm))
    story.append(Paragraph(
        "Panel I：蓝色为 exact theory 加完整 realization error layer，红色为 5 个 held-out realization 的 simulation 均值，误差条为 held-out SD。旧表的绝对 r 不匹配来自 cross-covariance 组装错误；使用同一 65.536 s 缓存重算后，baseline、node-deleted 和 delta 均匹配。",
        caption,
    ))
    exp3_rows = [["网络", "n", "latent exact r", "theory+error r", "held-out r", "held-out SD"]]
    for name, label in (("baseline", "baseline"), ("node_deleted", "node-deleted"), ("delta", "delta")):
        row = exp3.loc[name]
        exp3_rows.append([
            label, str(int(row.n_valid)), "-" if pd.isna(row.latent_exact_r) else f"{row.latent_exact_r:.3f}",
            f"{row.theory_plus_realization_r:.3f}", f"{row.simulation_r:.3f}", f"{row.simulation_r_sd:.3f}",
        ])
    story.append(_table(exp3_rows, [28*mm, 16*mm, 32*mm, 34*mm, 30*mm, 28*mm]))

    story.append(PageBreak())
    story.append(Paragraph("6. Experiment 3 严格筛选审计：为何不宣称成功", h1))
    story.append(Paragraph(
        "最初 exact-resolvent 筛选没有包含 simulation 使用的 log-PSD Gaussian smoothing，导致筛选算子与观察算子不一致。加入相同 smoothing 后，理论上可以构造 baseline 较小、deleted r 很大的输入协方差；但这类候选的 peak 在 trial 间只移动约 0.06 Hz，远低于随机谱 realization 的峰定位误差，因此 held-out simulation 不会出现相应强相关。",
        body,
    ))
    audit_rows = [
        ["检查", "结果", "含义"],
        ["v6 observation-matched theory", f"baseline r={v6['observation_matched_baseline_r']:.3f}; deleted r={v6['observation_matched_deleted_r']:.3f}", "理论 feature operator 已匹配"],
        ["v6 deleted peak SD", f"{v6['deleted_peak_sd_hz']:.3f} Hz", "相关性由极小 peak 变化构成，难以观测"],
        ["v9 最大 deleted peak SD", f"{v9.deleted_peak_sd_hz.max():.3f} Hz", "放大仍不足 0.5 Hz"],
        ["v9 对应 baseline", f"约 {v9.loc[v9.deleted_peak_sd_hz.idxmax(), 'baseline_r']:.3f}", "peak 变化放大时 baseline 已明显相关"],
        ["严格三条件", "无候选同时通过", "当前模型/feature 定义下是负结果"],
    ]
    story.append(_table(audit_rows, [45*mm, 62*mm, 67*mm]))
    story.append(Paragraph(
        "因此，修正后的结果支持：在当前已选 regime 中，理论可以同时解释 baseline、node-deleted 与相关性差值。不过该 regime 的 baseline simulation correlation 本身约为 -0.40，所以它仍不是“baseline 无相关、删除后新生相关”的强示例；这一点与理论-模拟是否匹配是两个不同问题。",
        body,
    ))
    story.append(Paragraph("显式 recurrence QC", h2))
    qc = pd.read_csv(SOURCE / "exp3_explicit_recurrence_qc_robust_v3.csv")
    qc_rows = [["网络", "rectification", "time-series r", "relative RMSE"]]
    for _, row in qc.iterrows():
        qc_rows.append([row.network, str(int(row.rectification_count)), f"{row.timeseries_r:.5f}", f"{row.relative_rmse:.4f}"])
    story.append(_table(qc_rows, [42*mm, 40*mm, 45*mm, 45*mm]))
    story.append(Paragraph(
        "4 个显式 recurrence QC trial 中没有发生 rectification；频域加速解与显式时间积分高度一致。baseline 的 relative RMSE 较高但 time-series r 仍为 0.985；node-deleted 网络 r≈1。",
        caption,
    ))

    story.append(PageBreak())
    story.append(Paragraph("7. 图像源文件与数值文件位置", h1))
    story.append(Paragraph(f"统一源目录：{SOURCE.resolve()}", body))
    source_rows = [["Panel", "PNG", "PDF"]]
    for _, row in source_index.iterrows():
        source_rows.append([
            row.panel,
            Paragraph(row.png_path, small),
            Paragraph(row.pdf_path, small),
        ])
    story.append(_table(source_rows, [16*mm, 79*mm, 79*mm], font_size=6.8))
    story.append(Spacer(1, 4 * mm))
    data_rows = [
        ["内容", "路径"],
        ["Experiment 1 dense 1000-condition results", str((dense_source / "exp1_dense_conditions.csv").resolve())],
        ["Experiment 1 9×9 metrics", str((SOURCE.parent / "realization_residual_validation" / "exp1_9x9_metrics.csv").resolve())],
        ["Experiment 2 S32 r<0.6 final 1000-condition results", str((exp2_source / "selected_1000_condition_results_rlt0p6.csv").resolve())],
        ["Experiment 2 65.536-s pilot metrics", str((SOURCE.parent / "exp2_pap_s32_window_65p536" / "metrics_100.json").resolve())],
        ["Experiment 2 131.072-s pilot metrics", str((exp2_source / "metrics_100.json").resolve())],
        ["Experiment 2 131.072-s r<0.6 final metrics", str((exp2_source / "metrics_1000_rlt0p6.json").resolve())],
        ["Experiment 3 v3 held-out results (corrected full covariance)", str((SOURCE / "exp3_node_deletion_long_results_robust_v3_corrected.csv").resolve())],
        ["Observation-matched v6 candidate", str((SOURCE / "exp3_selected_observation_matched_v6.json").resolve())],
        ["Null-space v9 audit", str((SOURCE / "exp3_observation_matched_nullspace_refinement_v9.csv").resolve())],
        ["Source index", str((SOURCE / "source_image_index.csv").resolve())],
        ["Dense-condition generator", str((EXP_ROOT / "pap_validation" / "paper_figure_dense_revision.py").resolve())],
    ]
    story.append(PageBreak())
    story.append(Paragraph("数值文件与生成程序", h2))
    wrapped_data_rows = [data_rows[0]] + [
        [Paragraph(row[0], small), Paragraph(row[1], small)] for row in data_rows[1:]
    ]
    story.append(_table(wrapped_data_rows, [54*mm, 120*mm], font_size=7))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    REPORT.write_text(
        "\n".join(
            [
                "全脑模型周期-非周期耦合：论文图与数值结果总结",
                f"PDF: {OUTPUT.resolve()}",
                f"源图目录: {SOURCE.resolve()}",
                "Experiment 1 V1->PG, 1000 x 64 trials: identity R2=0.9945, OLS R2=0.9953",
                f"Experiment 2 S32 residual-corrected 1000 conditions x 256 trials: identity R2={manifest['panel_G']['identity_R2']:.4f}, OLS R2={manifest['panel_G']['ols_R2']:.4f}",
                f"Experiment 3 corrected: theory/simulation baseline={exp3.loc['baseline','theory_plus_realization_r']:.4f}/{exp3.loc['baseline','simulation_r']:.4f}; node-deleted={exp3.loc['node_deleted','theory_plus_realization_r']:.4f}/{exp3.loc['node_deleted','simulation_r']:.4f}",
                "严格节点删除筛选：未找到 baseline-small / deleted-high / observable-peak 三条件同时成立的 held-out 候选。",
            ]
        )
    )
    return OUTPUT


if __name__ == "__main__":
    print(build())
