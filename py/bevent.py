from typing import NamedTuple
import numpy as np

def mass_sq(lv: np.array):
    return lv[:,0]**2 - np.sum(lv[:,1:]**2, axis=1)

def mass_sq_arr(*args):
    return mass_sq(sum(args))


class Jpsi2piKEvent(NamedTuple):
    """ B+ -> Jpsi pi+ pi- K+ event """
    de: np.ndarray
    mbc: np.ndarray
    pi1_mom: np.ndarray
    pi1_mom_mc: np.ndarray
    pi2_mom: np.ndarray
    pi2_mom_mc: np.ndarray
    jpsi_mom: np.ndarray
    jpsi_mom_mc: np.ndarray
    k_mom: np.ndarray
    k_mom_mc: np.ndarray
    k_charge: np.ndarray
    pdf: np.ndarray
    bdaug: np.ndarray
    foxWolfram: np.ndarray

    def serialize(self, path):
        np.savez(path, **{key: value for key, value in zip(self._fields, self)})

    @classmethod
    def deserialize(cls, path):
        data = np.load(path)
        return Jpsi2piKEvent(*[data[key] for key in cls._fields])

    @property
    def mjpsipipi(self):
        return np.sqrt(mass_sq_arr(self.pi1_mom, self.pi2_mom, self.jpsi_mom))

    @property
    def mjpsipipimc(self):
        return np.sqrt(mass_sq_arr(self.pi1_mom_mc, self.pi2_mom_mc, self.jpsi_mom_mc))

    @property
    def mkpipi(self):
        return np.sqrt(mass_sq_arr(self.pi1_mom, self.pi2_mom, self.k_mom))

    @property
    def mkpipimc(self):
        return np.sqrt(mass_sq_arr(self.pi1_mom_mc, self.pi2_mom_mc, self.k_mom_mc))

    @property
    def mpipi(self):
        return np.sqrt(mass_sq_arr(self.pi1_mom, self.pi2_mom))

    @property
    def mpipimc(self):
        return np.sqrt(mass_sq_arr(self.pi1_mom_mc, self.pi2_mom_mc))
