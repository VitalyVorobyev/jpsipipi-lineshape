#! /usr/bin/env python

from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

from pdftools import Pdf
from phist import PoissonHist

def make_minuit(fcn, params, errdef):
    kwargs = {}
    for name, val, err, rng, fixed in params:
        kwargs.update({
            name: val,
            f'error_{name}': err,
            f'limit_{name}': rng,
            f'fix_{name}': fixed
        })
    return Minuit(fcn, errordef=errdef, **kwargs)

class Fitter:
    def __init__(self, pdf, lo, hi):
        self.lo, self.hi = lo, hi
        self.pdf = pdf

    def fcn(self, m, w, f, b):
        self.pdf.setParams(m, w, f, b, 0)
        loglh = self.loglh()
        print(f'loglh {loglh:.2f}, m {m:.4f}, w {w*1000:.2f}, f {f:.3f}, b {b:.2f}, c {0:.2f}')
        return loglh

    def loglh(self):
        return -np.sum(np.log(self.pdf(self.data))) +\
            np.log(np.sum(self.pdf(self.normdata))) * self.data.size

    def fitTo(self, data:np.ndarray, normdata:np.ndarray):
        self.data = data[(data > self.lo) & (data < self.hi)]
        self.normdata = normdata[(normdata > self.lo) & (normdata < self.hi)]

        print(self.data.size, self.normdata.size)
        params = [
            ['m', 3.872,  0.001, (3.870, 3.875), False],
            ['w', 0.0012, 0.001, (0.000, 0.005), False],
            ['f', 0.36,    0.1,   (0., 1.),       False],
            ['b', 4.5,    1.,   (-20, 20),    False],
            # ['c', 50.,    50.,   (-10, 20),    False],
        ]
        minimizer = make_minuit(self.fcn, errdef=0.5, params=params)

        fmin, param = minimizer.migrad()
        return (fmin, param, minimizer.matrix(correlation=True))

def fit_plot(data, pdf, rng):
    phist = PoissonHist(data, *rng)

    x = np.linspace(*rng, 250)
    f = pdf(x) * phist.num_entries * phist.bin_size

    fig, ax = plt.subplots(figsize=(8, 6))
    phist.plot_on(ax)
    ax.plot(x, f)
    ax.set_xlabel(r'$m(J/\psi\pi^+\pi^-)$ (GeV)', fontsize=18)
    ax.set_xlim(rng)
    ax.set_ylim((0, 1.05*phist.data.max()))
    ax.legend(fontsize=18)
    ax.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')

    fig.tight_layout()
    plt.show()

def main():
    from convolution import apply_resolution
    from bevent import Jpsi2piKEvent

    seed = 56
    gen = np.random.default_rng(seed)

    sigma = 0.0027
    lof, hif = 3.855, 3.890
    loc, hic = 3.845, 3.900
    b_init = 4.5
    c_init = 1
    f_init = 0.36

    events = Jpsi2piKEvent.deserialize('../data/data_t1_270.npz').mjpsipipi
    measurements = apply_resolution(events, sigma, gen)
    pdf = Pdf((loc, hic), sigma, norm_range=(lof, hif),
        b=b_init, f=f_init, c=c_init)


    fitter = Fitter(pdf, lof, hif)
    normdata = gen.uniform(lof, hif, 3*10**6)
    fmin, param, corrmtx = fitter.fitTo(measurements, normdata)
    print(fmin)
    print(param)

    pdf.setParams(
        param[0].value,
        param[1].value,
        param[2].value,
        param[3].value,
        0
    )

    fit_plot(measurements, pdf, (lof, hif))

if __name__ == "__main__":
    main()
