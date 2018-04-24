import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from matplotlib import gridspec
import scikit_posthocs as sp


sns.set()
sns.set_context("notebook")

metrics = ['SDR', 'SIR', 'SAR', 'ISR']
targets = ['vocals', 'accompaniment', 'drums', 'bass', 'other']
selected_targets = ['vocals', 'accompaniment']
oracles = [
    'IBM1', 'IBM2', 'IRM1', 'IRM2', 'MWF'
]

df = pd.read_pickle("sisec18_mus.pandas")
df['oracle'] = df.method.isin(oracles)

# aggregate methods by mean using median by track
df = df.groupby(
    ['method', 'track', 'target', 'metric']
).median().reset_index()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.unicode'] = 'True'

sns.set()
sns.set_context("paper")

params = {
    'backend': 'ps',
    'axes.labelsize': 18,
    'font.size': 15,
    'legend.fontsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 15,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'ptmrr8re',
    'text.latex.unicode': True
}

sns.set_style("darkgrid", {
    'pgf.texsystem': 'xelatex',  # pdflatex, xelatex, lualatex
    "axes.facecolor": "0.925",
    'text.usetex': True,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'font.size': 14,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 17,
    'font.serif': [],
    'text.latex.unicode': True
})
plt.rcParams.update(params)

f = plt.figure(figsize=(22, 20))
# resort them by median SDR
# Get sorting keys (sorted by median of SDR:vocals score)
df_voc = df[(df.target == 'vocals') & (df.metric == "SDR")]
df_acc = df[(df.target == 'accompaniment') & (df.metric == "SDR")]

targets_by_voc_sdr = df_voc.score.groupby(
    df_voc.method
).median().sort_values().index.tolist()

targets_by_acc_sdr = df_acc.score.groupby(
    df_acc.method
).median().sort_values().index.tolist()

targets_by_voc_sdr_acc = [x for x in targets_by_voc_sdr if x in targets_by_acc_sdr]

# get the two sortings
df_voc['method'] = df_voc['method'].astype('category', categories=targets_by_voc_sdr, ordered=True)
df_acc['method'] = df_acc['method'].astype('category', categories=targets_by_acc_sdr, ordered=True)

# prepare the pairwise plots
pc_voc = sp.posthoc_conover(df_voc, val_col='score', group_col='method')
pc_acc = sp.posthoc_conover(df_acc, val_col='score', group_col='method')

f = plt.figure(figsize=(10, 10))
# Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
cmap = ['1', '#ff2626',  '#ffffff', '#fcbdbd', '#ff7272']
heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.90, 0.35, 0.04, 0.3]}
sp.sign_plot(pc_voc, **heatmap_args)

f.tight_layout()
f.savefig(
    "pairwise_vocals.pdf",
    bbox_inches='tight',
    dpi=300
)

f = plt.figure(figsize=(8.709677419, 8.709677419))

# Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
cmap = ['1', '#ff2626',  '#ffffff', '#fcbdbd', '#ff7272']
heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.90, 0.35, 0.04, 0.3]}
sp.sign_plot(pc_acc, **heatmap_args)

f.tight_layout()
f.savefig(
    "pairwise_acc.pdf",
    bbox_inches='tight',
    dpi=300
)
