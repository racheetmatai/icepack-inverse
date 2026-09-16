"""Plot saved, support-corrected median-control results; no physical solves."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter

p = argparse.ArgumentParser()
p.add_argument('metrics', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
configs = [f'CFG{i:02d}' for i in range(1, 7)]
def select(path):
    df = pd.read_csv(path)
    return df.loc[df.experiment.str.fullmatch(r'SQ\d{2}') &
                  df.population.eq('central_50km') & df.support_stratum.eq('all') &
                  df.control_kind.eq('median')].sort_values(['experiment','configuration'])
df = select(args.metrics)
assert len(df) == 60 and not df.duplicated(['experiment','configuration']).any()
fields = ['vector_rmse_m_per_a','uniform_vector_rmse_m_per_a','inversion_vector_rmse_m_per_a']
for field in fields[1:]:
    assert (df.groupby('experiment')[field].nunique() == 1).all()
pivot = df.pivot(index='experiment', columns='configuration', values=fields[0]).reindex(columns=configs)
base = df.groupby('experiment')[fields[1:]].first().reindex(pivot.index)
values = np.column_stack([base[fields[2]], base[fields[1]], pivot])
assert values.shape == (10,8) and np.isfinite(values).all() and (values > 0).all()
colors = ['#775599', '#333333', '#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                     'axes.labelsize':14, 'xtick.labelsize':11.5,
                     'ytick.labelsize':12, 'legend.fontsize':11,
                     'pdf.fonttype':42, 'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(9.1,5.6), constrained_layout=True)
x = np.arange(8)
ax.fill_between(x, values.min(axis=0), values.max(axis=0), color='0.72', alpha=.22, lw=0)
for row in values:
    ax.plot(x,row,color='0.68',lw=.8,alpha=.7,zorder=1)
    for j,c in enumerate(colors):
        ax.scatter(j,row[j],s=28,facecolor='white',edgecolor=c,linewidth=1.1,zorder=2)
for j,(m,c) in enumerate(zip(np.median(values,axis=0),colors)):
    ax.plot([j-.19,j+.19],[m,m],color=c,lw=3.5,solid_capstyle='butt',zorder=4)
ax.axvline(1.5,color='0.6',ls='--',lw=.8)
ax.set_yscale('log')
ax.set_ylim(values.min()*.65, values.max()*1.45)
ticks = [v for v in [.1,.2,.5,1,2,5,10,20,50,100,200,500,1000] if ax.get_ylim()[0] <= v <= ax.get_ylim()[1]]
ax.yaxis.set_major_locator(FixedLocator(ticks))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:g}'))
ax.set_xticks(x, ['Inversion\nreference','Uniform $C$']+configs)
ax.set_ylabel(r'Velocity RMSE (m a$^{-1}$; logarithmic scale)')
ax.tick_params(axis='x',length=0,pad=8)
ax.grid(axis='y',which='major',color='0.90',lw=.6)
ax.spines[['top','right']].set_visible(False)
ax.legend(handles=[Patch(facecolor='0.72',alpha=.22,label='Range across ten squares'),
                   Line2D([0],[0],color='0.68',marker='o',markerfacecolor='white',lw=.8,label='One line per square'),
                   Line2D([0],[0],color='0.25',lw=3.5,label='Median across squares')],
          loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,columnspacing=1.3,handlelength=1.8)
args.output.mkdir(parents=True,exist_ok=True)
for suffix in ['pdf','png']:
    fig.savefig(args.output/f'figure3_square_velocity_rmse.pdf'.replace('.pdf','.'+suffix),dpi=220,facecolor='white')
plt.close(fig)
pd.DataFrame(values,index=pivot.index,columns=['inversion_reference','uniform_C']+configs).to_csv(args.output/'figure3_values.csv',index_label='square')
record={'inputs':{str(args.metrics):hashlib.sha256(args.metrics.read_bytes()).hexdigest()},
        'verification':'All 60 median results and both baselines are present in the corrected evaluation table.',
        'range_m_per_a':[float(values.min()),float(values.max())],
        'inversion_range_m_per_a':[float(values[:,0].min()),float(values[:,0].max())],
        'median_inversion_m_per_a':float(np.median(values[:,0])),
        'physical_solves':0}
(args.output/'figure3_verification.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
