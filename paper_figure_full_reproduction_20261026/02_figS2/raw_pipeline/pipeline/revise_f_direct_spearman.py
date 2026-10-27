"""Replace only panel f's correction with an observed-minus-predicted rho.

Contract: retain final 3x3 evidence figure; f is a prediction-residual diagnostic,
not an estimate of a recovered latent correlation. All eight other axes and the
raw FOOOF/SPLIT distributions remain identical. No new simulations or fitting.
"""
from pathlib import Path
import hashlib
import io
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from lxml import etree as ET
import fitz

sys.path.insert(0, '/PORTABLE_HOME/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

SRC = Path('/PORTABLE_HOME/Documents/New project/output/fooof_split_broad_total_covariance_3x3_2026-09-13')
OLD = 'fooof_split_broad_total_covariance_nature_neuroscience_3x3'
OUT = Path(__file__).resolve().parent / 'outputs/fooof_split_direct_spearman_difference_20260914'
OUT.mkdir(parents=True, exist_ok=True)
STEM = 'fooof_split_3x3_direct_spearman_difference'
NS = {'s': 'http://www.w3.org/2000/svg'}
groups = pd.read_csv(SRC / 'all_group_results.csv')
g = groups.loc[groups.band.eq('high gamma')].copy()
g['fooof_direct_difference'] = g.fooof_emp - g.fooof_pred
assert len(g) == 20 and np.isfinite(g[['fooof_emp', 'fooof_pred', 'split_emp']]).all().all()
arrays = [g.fooof_emp.to_numpy(), g.fooof_direct_difference.to_numpy(), g.split_emp.to_numpy()]
g.to_csv(OUT / 'panel_f_source_data.csv', index=False)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 6.2,
    'axes.labelsize': 6.4, 'axes.linewidth': .65,
    'xtick.labelsize': 5.8, 'ytick.labelsize': 5.8,
    'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, axes = plt.subplots(3, 3, figsize=(518.4/72, 470.16/72))
layout = json.loads((SRC / f'{OLD}.alignment.json').read_text())['layout']
for ax, panel in zip(axes.flat, layout['panels']):
    x0,y0,x1,y1 = panel['bbox_pt']
    ax.set_position([x0/518.4,y0/470.16,(x1-x0)/518.4,(y1-y0)/470.16])
ax = axes[2,2]
v = ax.violinplot(arrays, positions=[0,1,2], widths=.76, showmeans=True, showextrema=False)
for body, color in zip(v['bodies'], ['#247BA0','#2A9D8F','#D1495B']):
    body.set_facecolor(color); body.set_edgecolor('black'); body.set_linewidth(.6); body.set_alpha(.8)
v['cmeans'].set_color('black'); v['cmeans'].set_linewidth(1.)
ax.axhline(0, color='0.35', lw=.6)
ax.set(ylim=(-.5,.5), xticks=[0,1,2],
    xticklabels=['FOOOF','FOOOF\nobserved − predicted','SPLIT'], ylabel='Spearman ρ or Δρ')
ax.text(-.14,1.035,'f',transform=ax.transAxes,fontsize=8,fontweight='bold',ha='left',va='bottom')
ax.tick_params(width=.6,length=2.5,pad=1.5)
ax.spines[['top','right']].set_visible(False)
for t in ax.get_xticklabels(): t.set_linespacing(1.25)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=OUT/f'{STEM}.alignment.json', strict=True)
buf=io.BytesIO(); fig.savefig(buf, format='svg')
for other in axes.flat:
    if other is not ax: other.set_visible(False)
fpdf=io.BytesIO(); fig.savefig(fpdf,format='pdf'); plt.close(fig)
fresh = ET.fromstring(buf.getvalue())
original_bytes = (SRC / f'{OLD}.svg').read_bytes()
root = ET.fromstring(original_bytes)

def byid(tree, name):
    result = tree.xpath('//*[@id=$name]', name=name)
    assert len(result)==1, name
    return result[0]

untouched = [ET.tostring(byid(root, f'axes_{i}')) for i in range(1,9)]
oldax = byid(root, 'axes_9'); newax=byid(fresh, 'axes_9')
newbody = newax.xpath('./s:g[starts-with(@id,"FillBetweenPolyCollection_")]', namespaces=NS)
oldbody = oldax.xpath('./s:g[starts-with(@id,"FillBetweenPolyCollection_")]', namespaces=NS)
assert len(oldbody)==len(newbody)==3
for idx in (0,2):
    # Exact agreement in untouched distributions proves original violin geometry.
    import re
    oldd=oldbody[idx].find('.//s:defs/s:path',NS).get('d')
    newd=newbody[idx].find('.//s:defs/s:path',NS).get('d')
    oldxy=np.array([float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',oldd)])
    newxy=np.array([float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',newd)])
    print('Unchanged violin coordinate tolerance:',idx,np.max(np.abs(oldxy-newxy)))
    assert np.allclose(oldxy,newxy,rtol=0,atol=2e-6)
oldbody[1].find('.//s:defs/s:path',NS).set('d',newbody[1].find('.//s:defs/s:path',NS).get('d'))
oldmean = byid(root,'LineCollection_1').findall('s:path',NS)
newmean = byid(fresh,'LineCollection_1').findall('s:path',NS)
assert len(oldmean)==len(newmean)==3
for i in (0,2): assert oldmean[i].get('d')==newmean[i].get('d')
oldmean[1].set('d',newmean[1].get('d'))
label=byid(root,'text_132').findall('s:text',NS)[1]
assert label.text=='corrected'
label.text='observed − predicted'
label.attrib.pop('transform')
label.set('x','445.426218'); label.set('y','454.093295')
label.set('style',label.get('style')+';text-anchor:middle')
byid(root,'text_139').find('s:text',NS).text='Spearman ρ or Δρ'
assert all(ET.tostring(byid(root,f'axes_{i}'))==untouched[i-1] for i in range(1,9))
svg=ET.tostring(root,xml_declaration=True,encoding='utf-8')
(OUT/f'{STEM}.svg').write_bytes(svg)

# Assemble the QA PDF using the original PDF and newly rendered Matplotlib f.
# MuPDF's SVG reader cannot reproduce this source's CSS clips/transparency;
# never use its SVG rendering for QA. The delivered SVG remains fully vector.
doc=fitz.open(SRC/f'{OLD}.pdf')
fdoc=fitz.open(stream=fpdf.getvalue(),filetype='pdf')
pdf=fitz.open(); page=pdf.new_page(width=518.4,height=470.16)
for rect in [fitz.Rect(0,0,518.4,321),fitz.Rect(0,321,357,470.16)]:
    page.show_pdf_page(rect,doc,0,clip=rect)
rect=fitz.Rect(357,321,518.4,470.16)
page.show_pdf_page(rect,fdoc,0,clip=rect)
pdf.save(OUT/f'{STEM}.pdf')
pdf[0].get_pixmap(matrix=fitz.Matrix(3,3)).save(OUT/f'{STEM}_preview.png')
pdf.close();doc.close();fdoc.close()
stats={'definition':'rho_held minus rho_pred; direct Spearman difference, no copula transform, no clipping',
       'n_groups':len(g), 'source_data':str(SRC/'all_group_results.csv'),
       'source_svg':str(SRC/f'{OLD}.svg'),
       'source_sha256':hashlib.sha256(original_bytes).hexdigest(),
       'unchanged_panels':'a, b1, b2, b3, b4, c, d, e (SVG element equality verified)',
       'test':'two-sided exact Wilcoxon signed-rank; unadjusted',
       'results':{}}
for name, arr in zip(['FOOOF','FOOOF observed minus predicted','SPLIT'], arrays):
    stats['results'][name]={'mean':float(np.mean(arr)), 'median':float(np.median(arr)),
        'p_against_zero':float(wilcoxon(arr,alternative='two-sided',method='exact').pvalue)}
stats['paired_difference_vs_SPLIT_p']=float(wilcoxon(arrays[1],arrays[2],method='exact').pvalue)
(OUT/'statistics_and_provenance.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2))
print(OUT/f'{STEM}.svg')
