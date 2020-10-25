#! /usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 14
})

from bevent import Jpsi2piKEvent
from phist import PoissonHist

def mx_vs_mkpipi(events):
    fig, ax = plt.subplots(ncols=2, figsize=(14, 7), sharey=True)

    N = 3*10**4
    ax[0].scatter(events.mkpipi[:N], events.mjpsipipi[:N], marker='o', s=0.4)
    ax[1].scatter(events.mpipi[:N], events.mjpsipipi[:N], marker='o', s=0.4)

    for a in ax:
        a.minorticks_on()
        a.grid(which='major')
        a.grid(which='minor', linestyle=':')

    fig.tight_layout()
    plt.show()

def slice_plot(events, ref, cb, conditions):
    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = 3.850, 3.895

    for cnd in conditions:
        ph = PoissonHist(events.mjpsipipi[cnd(cb(events))], dens=True, lo=lo, hi=hi)
        phref = PoissonHist(ref.mjpsipipi[cnd(cb(events))], dens=True, lo=lo, hi=hi)
        diff = ph - phref
        diff.plot_on(ax)

    ax.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')
    fig.tight_layout()

    plt.show()

def main():
    events_ref = Jpsi2piKEvent.deserialize('../data/data_t1_None.npz')
    events = Jpsi2piKEvent.deserialize('../data/data_t1_0.npz')
    mx_vs_mkpipi(events_ref)

    cb = lambda x: x.mkpipi
    conditions = [
        lambda x: x < 1.6,
        lambda x: (x > 1.6) & (x < 1.8),
        lambda x: x > 1.8,
    ]
    slice_plot(events, events_ref, cb, conditions)


if __name__ == "__main__":
    main()
