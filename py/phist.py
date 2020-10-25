import numpy as np
import typing

class PoissonHist:
    """ Binned data with symmetric Poisson error bars """

    def __init__(self, data:typing.Iterable=None, lo=None, hi=None, nbins=100, dens=False, wght=None):
        if data is not None:
            lo = min(data) if lo is None else lo
            hi = max(data) if hi is None else hi
            self.data, edges = np.histogram(data, bins=nbins, range=(lo, hi), normed=dens, weights=wght)
            self.bins = 0.5 * (edges[1:] + edges[:-1])
            norm = self.data.sum() / data.sum()
            self.errors = np.sqrt(self.data) * norm

    @property
    def nbins(self):
        return self.data.size if hasattr(self, 'data') else None

    @property
    def num_entries(self):
        return self.data.sum() if hasattr(self, 'data') else 0

    @property
    def bin_size(self):
        return self.bins[1] - self.bins[0] if hasattr(self, 'bins') else None

    def __add__(self, rhs):
        assert isinstance(rhs, PoissonHist)
        assert self.nbins == rhs.nbins
        result = PoissonHist()
        result.data = self.data + rhs.data
        result.bins = self.bins
        result.errors = np.sqrt(self.errors**2 + rhs.errors**2)
        return result

    def __sub__(self, rhs):
        assert isinstance(rhs, PoissonHist)
        assert self.nbins == rhs.nbins
        result = PoissonHist()
        result.data = self.data - rhs.data
        result.bins = self.bins
        result.errors = np.sqrt(self.errors**2 + rhs.errors**2)
        return result

    def plot_on(self, ax, label='', markersize=4):
        ax.errorbar(self.bins, self.data, yerr=self.errors,
            marker='o', linestyle='none', markersize=markersize, label=label)
