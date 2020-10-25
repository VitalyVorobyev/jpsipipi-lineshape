#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

from bevent import Jpsi2piKEvent
from phist import PoissonHist

def plot(data0, data1, lo=3.850, hi=3.895, bins=100):
    h0, h1 = [PoissonHist(item, lo, hi, bins) for item in [data0, data1]]

    fig, ax = plt.subplots(figsize=(8,6))
    h0.plot_on(ax, 'generated', 3)
    h1.plot_on(ax, 'measured', 3)

    ax.set_xlabel(r'$m(J/\psi\pi^+\pi^-)$ (GeV)', fontsize=18)
    ax.set_xlim((lo, hi))
    ax.set_ylim((0, 1.05*h0.data.max()))
    ax.legend(fontsize=18)
    ax.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')

    fig.tight_layout()
    plt.show()

def apply_resolution(data, sigma, gen=None):
    gen = gen if gen is not None else np.random.default_rng()
    return data + sigma * gen.standard_normal(data.shape)

def main():
    events = Jpsi2piKEvent.deserialize('../data/data_t1_None.npz').mjpsipipi
    print(events.shape)
    measurements = apply_resolution(events, 0.0027)
    plot(events, measurements, lo=3.855, hi=3.890)

if __name__ == '__main__':
    main()
