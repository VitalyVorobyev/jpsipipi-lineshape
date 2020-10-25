#! /usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 14
})

from bevent import Jpsi2piKEvent
from phist import PoissonHist
from convolution import apply_resolution

def make_x_mass_plot(data:dict, lo=3.850, hi=3.895, bins=100, smeared=False, sigma=0.0027):
    fcn = lambda x: x if not smeared else apply_resolution(x, sigma)
    transform = lambda item: PoissonHist(fcn(item.mjpsipipi), lo, hi , bins, False, item.pdf)
    hists = {key: transform(item) for key, item in data.items()}

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))
    maxval = 0
    for key, hist in hists.items():
        maxval = max(maxval, hist.data.max())
        hist.plot_on(ax[0], key, 2)
        if key != 'no':
            diff = hists['no'] - hist
            diff.plot_on(ax[1], key, 2)

    ax[1].set_xlabel(r'$m(J/\psi\pi^+\pi^-)$ (GeV)', fontsize=18)
    ax[0].set_xlim((lo, hi))
    ax[0].set_ylim((0, 1.05*maxval))

    for a in ax:
        a.legend(fontsize=18)
        a.minorticks_on()
        a.grid(which='major')
        a.grid(which='minor', linestyle=':')
    fig.tight_layout()
    plt.show()

def main():
    data = {
        'no': Jpsi2piKEvent.deserialize('../data/data_t1_None.npz'),
        '0': Jpsi2piKEvent.deserialize('../data/data_t1_0.npz'),
        '90': Jpsi2piKEvent.deserialize('../data/data_t1_90.npz'),
        '180': Jpsi2piKEvent.deserialize('../data/data_t1_180.npz'),
        '270': Jpsi2piKEvent.deserialize('../data/data_t1_270.npz'),
    }
    make_x_mass_plot(data, lo=3.855, hi=3.890, bins=200, smeared=True)


if __name__ == "__main__":
    main()
