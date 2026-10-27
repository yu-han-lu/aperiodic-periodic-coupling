from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE=Path(__file__).resolve().parents[1]


def _load(path): return json.loads(path.read_text()) if path.exists() else {}
def _v(d,*keys,default=float("nan")):
    for k in keys:
        d=d.get(k,{})
    return d if d!={} else default


def build_summary(output_root: Path, report_dir: Path):
    m1=_load(output_root/"exp1_input_sampling"/"metrics.json"); m2=_load(output_root/"exp2_nonlinear_hopf"/"metrics.json"); m3=_load(output_root/"exp3_structural_kernel"/"metrics.json")
    e1best=max((k for k in m1 if isinstance(m1[k],dict)),key=lambda k:m1[k].get("R2",-1),default="C49")
    f1=m1.get(e1best,{}); f2=_v(m3,"heldout_C_first",default={}); s2=_v(m3,"heldout_C_second",default={})
    ci=_v(m3,"bootstrap_rmse_improvement_C_95CI",default=[float("nan"),float("nan")])
    p1=pd.read_csv(output_root/"exp1_input_sampling"/"theory_predictions.csv"); s1=pd.read_csv(output_root/"exp1_input_sampling"/"condition_summary.csv")
    t1=pd.read_csv(output_root/"exp1_input_sampling"/"trials.csv.gz",low_memory=False); dv=pd.read_csv(output_root/"exp1_input_sampling"/"derivative_validation.csv")
    eps_errors=p1.assign(err=lambda d:abs(d.predicted_covariance-d.observed_covariance)).groupby(["epsilon","method"]).err.mean().unstack()
    skew=p1[p1.design=="skewed"].assign(err=lambda d:abs(d.predicted_covariance-d.observed_covariance)).groupby("method").err.mean()
    d2=pd.read_csv(output_root/"exp2_nonlinear_hopf"/"radial_moments.csv"); p2=pd.read_csv(output_root/"exp2_nonlinear_hopf"/"theory_predictions.csv")
    radial_rel=float(np.median(abs(d2.E_R2_emp-d2.E_R2)/d2.E_R2)); radial_var_r=float(d2[["Var_R2_emp","Var_R2_theory"]].corr().iloc[0,1])
    sl=m2.get("Stuart-Landau",{}).get("C_semi",{}); ei=m2.get("smooth local E-I",{}).get("C_semi",{})
    gb=pd.read_csv(output_root/"exp3_structural_kernel"/"graph_baselines.csv").set_index("predictor")
    lines=[
"# p-ap coupling 理论数值验证总结",
"", "## 摘要",
f"本项目在同一套可复现代码中检验输入抽样、近 Hopf 非线性以及结构扰动三条 p-ap coupling 路径。输入理论和结构 kernel 得到定量支持；Eq.（134）在本项目预注册的 100-1000 Hz exponent 上没有得到跨条件支持。周期特征 P 为 5-20 Hz 内三次样条导数确定的连续峰值，A 为 100-1000 Hz 固定频带 log-log 斜率的相反数。实验 1 有 {len(t1):,} 行 trial，结构实验有 1500 个随机 15 边 mask。",
"", "## 复用的模型与约定",
"实验复用了 Li, Li & Wang (2025) 的 55 区 marmoset E-I Jacobian、FLN、层级参数和标准参数。状态按全部 E 后全部 I 排列；FLN[target, source] 表示 source -> target，一条 FLN 边同时映射到 J[E_target,E_source] 与 J[I_target,E_source]。实验 1 使用完整 55 区网络，实验 3 使用九区诱导子网络以完成 1500 个 mask 的 held-out 验证。",
"所有解析 PSD 使用 R(f)=(i2πf/1000 I-J)^-1。输入为周期 bump 与幂律背景之和，z=(bump center, bump amplitude, beta, aperiodic amplitude)。越界样本明确裁剪并保存标志；所有结构网络共享同一组 z。peak boundary、竞争峰和不稳定状态均有 QC 字段。",
"", "## 实验 1：输入抽样",
f"56 个 sampling condition 各使用 4096 trial；峰有效率为 {t1.peak_valid.mean():.3%}，裁剪率为 {t1.clipped.mean():.3%}。仅改变输入联合分布就使 covariance 覆盖 [{s1.covariance.min():.3f}, {s1.covariance.max():.3f}]，Pearson r 覆盖 [{s1.pearson_r.min():.3f}, {s1.pearson_r.max():.3f}]，因此同一个固定网络可以出现正、负或近零 coupling。",
f"跨全部条件，式（49）R2={m1['C49']['R2']:.6f}、RMSE={m1['C49']['RMSE']:.4g}；式（48）R2={m1['C48']['R2']:.6f}；quadratic R2={m1['Cquad']['R2']:.6f}；式（64）R2={m1['C64']['R2']:.4f}。式（49）的 mean absolute error 从 epsilon=0.05 的 {eps_errors.loc[.05,'C49']:.3g} 增至 epsilon=0.4 的 {eps_errors.loc[.4,'C49']:.3g}。偏斜输入上式（48）的 MAE={skew['C48']:.3g}，没有优于式（49）的 {skew['C49']:.3g}；这说明三阶修正在本特征映射中很小，而 epsilon=0.4 时遗漏的高阶项更重要。Gaussian 式（64）的 linear-cubic 修正在强变化尺度上过冲，不能被视为无条件改进。",
f"50 个随机点的导数交叉步长验证中，一阶、二阶和三阶张量的中位相对误差分别为 {dv.grad_rel_error.median():.2e}、{dv.hessian_rel_error.median():.2e}、{dv.third_rel_error.median():.2e}；三阶误差最大值为 {dv.third_rel_error.max():.2g}，对应接近强曲率/peak identity 变化的少数点，已保留而未静默删除。",
"", "## 实验 2：近 Hopf 非线性",
f"Stuart-Landau 层使用偶数 seed 估计 c_i、kappa_R，奇数 seed held-out。修正快速旋转的 Euler 径向偏差后，式（136）对 E[R2] 的中位相对误差为 {radial_rel:.2%}，理论与经验 Var(R2) 跨条件相关 r={radial_var_r:.3f}，所以径向平稳分布得到支持。",
f"但主分析的 Eq.（134）结果为 R2={sl.get('R2',float('nan')):.3f}、RMSE={sl.get('RMSE',float('nan')):.4g}、符号准确率={sl.get('sign_accuracy',float('nan')):.3f}，不支持跨条件预测。原因不是 radial density 失败，而是 100-1000 Hz 已远离约 10 Hz 的振荡峰，A 对 R2 的 kappa_R 很弱且窗口估计噪声占主导；同时低振幅时 phase noise 使训练 c_i 的估计不稳定。Eq.（134）预测的是 instantaneous-frequency-A coupling，不能在这里无条件替代 PSD-peak-A coupling。",
f"原模型局部层中，单独增大 E-E 会先到实特征值不稳定而非 Hopf。扫描全部脑区后选择所需 cross-coupling 调整最小的 PG，锁定 I-to-E multiplier={m2.get('PG_I_to_E_balance_multiplier',float('nan')):.3f}，随后在 E-E gain={m2.get('hopf_gain',float('nan')):.4f} 处获得复共轭 Hopf。平滑 E-I held-out Eq.（134）仅 R2={ei.get('R2',float('nan')):.3f}；这同样提示 finite-band feature estimation 和正常形外项不可忽略。",
"", "## 实验 3：结构 kernel",
f"random-15 held-out 一阶预测：R2={f2.get('R2',float('nan')):.3f}，RMSE={f2.get('RMSE',float('nan')):.4g}；二阶预测：R2={s2.get('R2',float('nan')):.3f}，RMSE={s2.get('RMSE',float('nan')):.4g}。配对 bootstrap 的 RMSE(first)-RMSE(second) 95% CI 为 [{ci[0]:.4g}, {ci[1]:.4g}]。单边删除/加倍、区域入边/出边删除和 1500 个随机 15 边 mask 均使用同一输入 trial 集。二阶方向导数避免构造四阶结构 Hessian。",
f"对理论内部量 C^[1](J+Delta J)，一阶/二阶结构展开 R2 分别为 {m3['structural_expansion_first_vs_C1']['R2']:.3f}/{m3['structural_expansion_second_vs_C1']['R2']:.3f}；这与 end-to-end exact 结果分开报告。end-to-end top 10% effect overlap 从 {f2.get('top10pct_overlap',float('nan')):.3f} 提高到 {s2.get('top10pct_overlap',float('nan')):.3f}，符号准确率从 {f2.get('sign_accuracy',float('nan')):.3f} 提高到 {s2.get('sign_accuracy',float('nan')):.3f}。",
f"单边删除样本中，结构 kernel 与直接 Delta C 的 R2={gb.loc['structural_kernel','R2']:.3f}，高于 edge weight 的 {gb.loc['edge_weight','R2']:.3f}、edge betweenness 的 {gb.loc['edge_betweenness','R2']:.3f} 和 hierarchy difference 的 {gb.loc['hierarchy_difference','R2']:.3f}。一阶失败主要随 ||R Delta J||、一阶抵消和 edge-pair interaction 增强；所有 1500 个删除网络保持稳定且 256 个输入 trial 均有效，因此本批次没有 instability/peak-switching 驱动的失败。",
"", "## 适用范围与限制",
"解析 resolvent 实验验证的是固定点附近线性响应；near-Hopf 实验使用独立的平滑非线性局部层。九区结构实验保留准确局部参数和 FLN-to-J 映射，但不等同于完整 55 区结构 kernel。Eq.（134）的失败说明“正常形正确”与“特定 PSD exponent estimator 能看到该机制”是两件事。线性 E-I 对照多数没有合格内点峰，这是预期的无自持振荡负对照。直接 PSD feature 与 specparam/FOOOF 拟合误差未混合。",
"", "## 结论",
"1. 输入 sampling geometry 本身足以决定 coupling 的符号和大小，式（49）在本实验范围内最稳健。",
"2. 式（136）的径向统计成立，但式（134）对预注册 100-1000 Hz A 的 held-out 预测不成立；机制存在的代数条件不能替代 feature 可辨识性。",
"3. anatomical structural kernel 对随机多边删除有 held-out 预测力；二阶 edge-pair 项稳定提高 R2、降低 RMSE，但 regression slope 仍约 2，说明大幅删除超出局部展开的定量校准范围。",
"", "## 重要图索引",
"- 实验 1：`outputs/exp1_input_sampling/figures/`",
"- 实验 2：`outputs/exp2_nonlinear_hopf/figures/`",
"- 实验 3：`outputs/exp3_structural_kernel/figures/`",
"", "## 复现命令",
"`PYTHONPATH=experiments/pap_theory_validation python -m pap_validation.run --config experiments/pap_theory_validation/configs/full.yaml --experiment all`",
]
    report_dir.mkdir(parents=True,exist_ok=True); md=report_dir/"summary_zh.md"; md.write_text("\n".join(lines))
    key=pd.DataFrame([
        {"experiment":"input sampling","comparison":e1best,"observed":"across conditions","predicted":"theory","uncertainty":"bootstrap per condition","R2":f1.get("R2"),"RMSE":f1.get("RMSE"),"interpretation":"sampling distribution changes coupling"},
        {"experiment":"nonlinear Hopf","comparison":"Eq.134 Stuart-Landau held-out","observed":"Cov(dotPhi,A)","predicted":"-ci*kappa*Var(R2)","uncertainty":"even/odd seed split","R2":sl.get("R2"),"RMSE":sl.get("RMSE"),"interpretation":"not supported for 100-1000 Hz A"},
        {"experiment":"nonlinear Hopf","comparison":"Eq.136 radial moments","observed":f"median relative error {radial_rel:.4g}","predicted":"stationary radial density","uncertainty":"90 parameter conditions","R2":radial_var_r**2,"RMSE":None,"interpretation":"radial density supported"},
        {"experiment":"structural kernel","comparison":"first order held-out","observed":"Delta C","predicted":"kernel","uncertainty":"20/80 split","R2":f2.get("R2"),"RMSE":f2.get("RMSE"),"interpretation":"linear structural response"},
        {"experiment":"structural kernel","comparison":"second order held-out","observed":"Delta C","predicted":"directional Hessian","uncertainty":str(ci),"R2":s2.get("R2"),"RMSE":s2.get("RMSE"),"interpretation":"pair/nonlinear correction"},
    ]); key.to_csv(report_dir/"key_results.csv",index=False)
    _pdf(lines,output_root,report_dir/"summary_zh.pdf")
    return md,report_dir/"summary_zh.pdf"


def _pdf(lines,output_root,path):
    from html import escape
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    font_path="/System/Library/Fonts/STHeiti Medium.ttc"
    pdfmetrics.registerFont(TTFont("STHeiti",font_path,subfontIndex=0))
    styles=getSampleStyleSheet(); font="STHeiti"
    title=ParagraphStyle("CNTitle",parent=styles["Title"],fontName=font,fontSize=20,leading=28,textColor=colors.HexColor("#17202A"),spaceAfter=10*mm)
    h2=ParagraphStyle("CNH2",parent=styles["Heading2"],fontName=font,fontSize=14,leading=20,textColor=colors.HexColor("#1F4E5F"),spaceBefore=5*mm,spaceAfter=2.5*mm,keepWithNext=True)
    body=ParagraphStyle("CNBody",parent=styles["BodyText"],fontName=font,fontSize=9.4,leading=16,spaceAfter=2.2*mm,textColor=colors.HexColor("#202428"))
    bullet=ParagraphStyle("CNBullet",parent=body,leftIndent=5*mm,firstLineIndent=-3*mm)
    code=ParagraphStyle("CNCode",parent=body,fontName="Courier",fontSize=7.5,leading=11,leftIndent=4*mm,backColor=colors.HexColor("#F3F5F6"),borderPadding=5)
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=17*mm,title="p-ap coupling 理论数值验证总结",author="Codex numerical validation")
    story=[Paragraph("p-ap coupling 理论数值验证总结",title)]
    for line in lines:
        if line.startswith("# ") or not line: continue
        if line.startswith("## "): story.extend([Paragraph(escape(line[3:]),h2)])
        elif line.startswith("- "): story.append(Paragraph("• "+escape(line[2:]),bullet))
        elif line.startswith("`") and line.endswith("`"): story.append(Paragraph(escape(line.strip("`")),code))
        elif line[0:2] in ("1.","2.","3.") and len(line)>3: story.append(Paragraph(escape(line),bullet))
        else: story.append(Paragraph(escape(line),body))
    images=[output_root/"exp1_input_sampling/figures/observed_vs_predicted.png",output_root/"exp2_nonlinear_hopf/figures/eq134_observed_predicted.png",output_root/"exp3_structural_kernel/figures/anatomical_kernel_heatmap.png",output_root/"exp3_structural_kernel/figures/random15_heldout_predictions.png"]
    for img in images:
        if not img.exists(): continue
        story.extend([PageBreak(),Paragraph(escape(img.stem.replace("_"," ")),h2),Spacer(1,4*mm)])
        flow=Image(str(img)); maxw,maxh=174*mm,225*mm; scale=min(maxw/flow.imageWidth,maxh/flow.imageHeight)
        flow.drawWidth=flow.imageWidth*scale; flow.drawHeight=flow.imageHeight*scale; flow.hAlign="CENTER"; story.append(flow)
    def decorate(canvas,doc):
        canvas.saveState(); canvas.setFont(font,8); canvas.setFillColor(colors.HexColor("#687078"));
        canvas.drawString(18*mm,9*mm,"p-ap coupling theory validation"); canvas.drawRightString(A4[0]-18*mm,9*mm,str(doc.page)); canvas.restoreState()
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default=str(BASE/"outputs")); p.add_argument("--report-dir",default=str(BASE/"reports")); a=p.parse_args()
    print("\n".join(map(str,build_summary(Path(a.output_dir),Path(a.report_dir)))))


if __name__=="__main__": main()
