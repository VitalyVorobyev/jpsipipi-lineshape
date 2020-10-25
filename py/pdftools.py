#! /usr/bin/env python

import numpy as np
from scipy import integrate, signal, stats, interpolate

def normed(f, lo, hi):
    return lambda x: f(x) / integrate.quad(f, lo, hi)[0]

def rbw(e, m, w):
    """ Relativistic Breit-Wigner lineshape """
    ampl = 1./(e**2 - m**2 + 1j*m*w)
    return ampl.real**2 + ampl.imag**2

def poly2d(e, b, c):
    return np.clip(e**2 + b*e + c, 0.01, None)

def poly1d(e, b):
    return b*e + 1

def make_normed_pdfs(m, w, b, c, f, lo, hi):
    sigpdf = normed(lambda x0: rbw(x0, m, w), lo, hi)
    bkgpdf = normed(lambda x1: poly2d(x1, b, c), lo, hi)
    return (
        lambda x: f * sigpdf(x),  # signal pdf
        lambda x: (1-f) * bkgpdf(x),  # background pdf
        lambda x: f * sigpdf(x) + (1-f) * bkgpdf(x)  # full pdf
    )

def smeared_pdf(pdf, sigma, lo, hi, grid_size=500):
    x = np.linspace(lo, hi, grid_size)
    xi = np.linspace(-(hi - lo) / 2, (hi - lo) / 2, grid_size)
    spdf = signal.fftconvolve(pdf(x), stats.norm.pdf(xi, 0, sigma), 'same') * (x[1] - x[0])
    return interpolate.interp1d(x, spdf, kind='cubic')

class Pdf:
    def __init__(self, def_range, sigma, norm_range=None, m=3.872, w=0.0012, f=0.3, b=10., c=50.):
        if norm_range is None:
            norm_range = def_range
        self.lo, self.hi = def_range
        self.nlo, self.nhi = norm_range
        self.sigma = sigma
        self.setParams(m, w, f, b, c)

    def init(self):
        self.sig_pdf = normed(lambda x: rbw(x, self.m, self.w), self.nlo, self.nhi)
        # self.bkg_pdf = normed(lambda x: poly2d(x-self.lo, self.b, self.c), self.nlo, self.nhi)
        self.bkg_pdf = normed(lambda x: poly1d(x-self.lo, self.b), self.nlo, self.nhi)
        self.full_pdf = lambda x: self.f * self.sig_pdf(x) + (1-self.f) * self.bkg_pdf(x)
        self.smeared_pdf = smeared_pdf(self.full_pdf, self.sigma, self.lo, self.hi)

    def __call__(self, x):
        return self.smeared_pdf(x)

    def setParams(self, m, w, f, b, c):
        self.m = m
        self.w = w
        self.f = f
        self.b = b
        self.c = c
        self.init()

def main():
    import matplotlib.pyplot as plt

    m = 3.872
    w = 0.0012
    sigma = 0.0027
    lof, hif = 3.855, 3.890
    loc, hic = 3.845, 3.900
    b, c = 10, 50
    f = 0.3  # signal fraction

    sigpdf, bkgpdf, pdf = make_normed_pdfs(m, w, b, c, f, lof, hif)
    smearpdf = smeared_pdf(pdf, sigma, loc, hic)
    x = np.linspace(lof, hif, 250)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, pdf(x), label='full')
    ax.plot(x, smearpdf(x), label='smeared')
    # ax.plot(x, spdf(x), ':', label='signal')
    # ax.plot(x, bpdf(x), '.', label='background')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
