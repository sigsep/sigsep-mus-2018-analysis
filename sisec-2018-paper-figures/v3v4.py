import json
import numpy as np
import argparse
from pathlib import Path
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import musdb


class Framing:
    """helper iterator class to do overlapped windowing"""
    def __init__(self, window, hop, length):
        self.current = 0
        self.window = window
        self.hop = hop
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.nwin:
            raise StopIteration
        else:
            start = self.current * self.hop
            if np.isnan(start) or np.isinf(start):
                start = 0
            stop = min(self.current * self.hop + self.window, self.length)
            if np.isnan(stop) or np.isinf(stop):
                stop = self.length
            result = slice(start, stop)
            self.current += 1
            return result

    @property
    def nwin(self):
        if self.window < self.length:
            return int(
                np.floor((self.length - self.window + self.hop) / self.hop)
            )
        else:
            return 1

    next = __next__


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    color_tuples = map(tuple, color_list.tolist())
    return base.from_list(cmap_name, color_list, N), color_tuples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate Folder')
    parser.add_argument(
        'submission_dirs',
        help='directories of submissions',
        nargs='+',
        type=str
    )

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['text.latex.unicode'] = 'True'

    sns.set()
    sns.set_context("paper")

    # Get this from LaTeX using \showthe\columnwidth
    fig_width_pt = 244.6937
    # Convert pt to inch
    inches_per_pt = 1.0 / 72.27
    # Aesthetic ratio
    golden_mean = (math.sqrt(5) - 1.0) / 2.0
    # width in inches
    fig_width = fig_width_pt * inches_per_pt
    # height in inches
    fig_height = fig_width * golden_mean
    fig_size = np.array([fig_width*2, fig_height*2])

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
        'figure.figsize': fig_size
    }

    sns.set_style("darkgrid", {
        "axes.facecolor": "0.925",
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 17,
        'font.serif': 'ptmrr8re',
    })
    plt.rcParams.update(params)
    args = parser.parse_args()
    data = []
    mus = musdb.DB()
    v3 = []
    v4 = []
    plot_target = 'vocals'
    for path in args.submission_dirs:
        p = Path(path)
        if p.exists():
            json_paths = p.glob('**/*.json')
            for json_path in json_paths:
                with open(json_path) as json_file:
                    print(json_path.stem)
                    json_string = json.loads(json_file.read())
                    track = mus.load_mus_tracks(tracknames=[json_path.stem])[0]
                    (nsampl, nchan) = track.targets[plot_target].audio.shape
                    framer = Framing(track.rate, track.rate, nsampl)
                    nwin = framer.nwin
                    audio = track.targets[plot_target].audio
                    energy = [
                        np.sqrt(np.mean(audio[win, :]**2))
                        for win in framer
                    ]

                    vocal_scores = list(
                        filter(
                            lambda target: target['name'] == plot_target,
                            json_string['targets']
                        )
                    )
                    frames = vocal_scores[0]['frames']
                    for s, e in zip(frames, energy):
                        if "v3" in str(json_path):
                            v3.append([e, s['metrics']['SIR']])
                        else:
                            v4.append([e, s['metrics']['SIR']])

    fig, ax1 = plt.subplots(figsize=fig_size)
    v3 = np.array(v3)
    v4 = np.array(v4)
    ax1.scatter(v3[:, 0], v3[:, 1], label="v3", s=3, alpha=0.5)
    ax1.scatter(v4[:, 0], v4[:, 1], label="v4", s=3, alpha=0.5)
    ax1.legend()
    ax1.set_xlabel('Framewise RMS (%s)' % plot_target.capitalize())
    ax1.set_ylabel('Framewise SIR in dB (%s)' % plot_target.capitalize())
    ax1.set_ylim([-50, 100])
    ax1.set_xlim([0, 0.175])
    fig.set_tight_layout(True)
    fig.savefig(
        "timeplot_sir_%s.pdf" % plot_target.capitalize(),
        bbox_inches='tight',
        dpi=300
    )
