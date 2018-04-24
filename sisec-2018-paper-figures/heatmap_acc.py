import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from matplotlib import gridspec


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
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'ptmrr8re',
    'text.latex.unicode': False
}

sns.set_style("darkgrid", {
    'pgf.texsystem': 'xelatex',  # pdflatex, xelatex, lualatex
    "axes.facecolor": "0.925",
    'text.usetex': False,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'font.size': 14,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 17,
    'font.serif': [],
    'text.latex.unicode': False
})
plt.rcParams.update(params)

target = 'accompaniment'
f, ax = plt.subplots(1, 1, figsize=(16, 10))

df_target = df[(df.target == target) & (df.metric == 'SDR')]

targets_by_score = df_target.score.groupby(
    df_target.method
).median().sort_values().index.tolist()

tracks_by_score = df_target.score.groupby(
    df_target.track
).median().sort_values().index.tolist()

pivoted = pd.pivot_table(df_target, values='score', index='method', columns='track')

pivoted = pivoted.reindex(index=targets_by_score[::-1], columns=tracks_by_score[::-1])
sns.heatmap(
    pivoted, square=True, ax=ax, cmap='viridis', vmin=np.percentile(pivoted, 10), vmax=np.percentile(pivoted, 90)
)
for label in ax.get_yticklabels():
    label.set_rotation(0)
f.savefig(
    "heatmap_acc.pdf",
    bbox_inches='tight',
    dpi=300
)
