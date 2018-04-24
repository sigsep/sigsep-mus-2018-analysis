import os
import glob
import yaml
import pandas as pd
import json
import scipy.io
import numpy as np
import difflib
import csv
import argparse
from pathlib import Path
from pandas.io.json import json_normalize
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import BlendedGenericTransform
import matplotlib as mpl
from matplotlib import gridspec
import soundfile as sf


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
        help='method folder',
        nargs='+',
        type=str
    )

    parser.add_argument(
        '--track_name',
        help='method folder',
        type=str,
        default="AM Contra - Heart Peripheral"
    )

    args = parser.parse_args()

    oracles = [
        'IBM1', 'IBM2', 'IRM1', 'IRM2', 'MWF', 'IMSK'
    ]

    data = []
    color_list, color_tuples = discrete_cmap(
        len(args.submission_dirs) + 2,
        'cubehelix_r'
    )
    fig, ax1 = plt.subplots()
    scores = []
    data = []
    for k, path in enumerate(args.submission_dirs):
        p = Path(path)
        is_oracle = p.stem in oracles
        p = p / 'test' / (args.track_name + '.json')
        with open(p) as json_file:
            json_string = json.loads(json_file.read())
            try:
                scores = list(
                    filter(
                        lambda target: target['name'] == 'vocals',
                        json_string['targets']
                    )
                )
                frames = scores[0]['frames']
            except IndexError:
                continue

            score = np.array([s['metrics']['SDR'] for s in frames])
            t = np.array([s['time'] for s in frames])
            if is_oracle:
                ax1.plot(t, score, lw=2, color='red', alpha=0.4)
            else:
                data.append(score)
                ax1.plot(t, score, lw=2, color='gray', alpha=0.4)

    import musdb
    mus = musdb.DB()
    track = mus.load_mus_tracks(tracknames=[p.stem])[0]
    (nsampl, nchan) = track.audio.shape
    framer = Framing(track.rate, track.rate, nsampl)
    nwin = framer.nwin
    audio = track.targets['vocals'].audio
    energy = [
        np.sqrt(np.mean(audio[win, :]**2))
        for win in framer
    ]
    ax2 = ax1.twinx()
    # ax2.step(range(nwin), energy, alpha=0.5, color='red')
    ax2.plot(range(nwin), energy, alpha=0.5, color='blue')
    # ax2.set_ylim([0.1, .3])
    D = np.array(data)
    U = np.argmax(np.nanmean(D, axis=1))
    print("best method:", np.max(np.nanmean(D, axis=1)))
    print("upper_bound:", np.mean(np.nanmax(D, axis=0)))
    L = np.argmin(np.nanmean(D, axis=1))
    lower_bound = np.nanmin(D, axis=0)
    upper_bound = np.nanmax(D, axis=0)
    # ax1.plot(t, D[U], lw=2, label='Best Method', color='green')
    ax1.fill_between(t, lower_bound, upper_bound, facecolor='gray', alpha=0.1)
    ax1.legend()
    plt.show()
