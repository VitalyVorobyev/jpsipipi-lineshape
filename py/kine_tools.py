import numpy as np

# 3-vector

def ptot_sq(vec):
    return np.sum(vec**2, axis=1)

def ptot(vec):
    return np.sqrt(ptot_sq(vec))

# Lorentz vector

def p3tot_sq(lvec):
    return np.sum(lvec[:,1:]**2, axis=1)

def p3tot(lvec):
    return np.sqrt(p3tot_sq(lvec))

def mass_sq(lvec: np.array):
    return lvec[:,0]**2 - p3tot_sq(lvec)

def mass_sq_arr(*args):
    return mass_sq(sum(args))

def gamma(lvec):
    """ """
    return lvec[:,0] / p3tot(lvec)

# Boost vector

def boost_vector(lvec):
    """ """
    return -lvec[:,1:] / lvec[:,0]

def boost_beta_sq(bvec):
    return ptot_sq(bvec)

def boost_beta(bvec):
    return np.sqrt(boost_beta_sq)

def boost_gamma(bvec):
    return 1./np.sqrt(1 - boost_beta_sq(bvec))

def boost_direction(bvec):
    return bvec / np.sqrt(np.sum(bvec**2, axis=1))

# Lorentz transformation

def vecdot(v1, v2):
    """ [Nx3] x [N*3] -> [Nx1] """
    return np.sum(v1 * v2, axis=1)

def boosted_time(lvec, bvec, nr=None):
    nr = nr if nr is not None else vecdot(boost_direction(bvec), lvec[:,1:])
    return gamma(lvec) * (lvec[:,0] - boost_beta(bvec) * nr)

def boosted_space(lvec, bvec, nr=None):
    nr = nr if nr is not None else vecdot(boost_direction(bvec), lvec[:,1:])
    gam = gamma(lvec)
    bdir = boost_direction(bvec)
    return lvec[:,1:] + (gam - 1) * nr * bdir -\
        gam * lvec[:,0] * ptot(bvec) * bdir


def boosted_to(lvec, bvec):
    nr = vecdot(boost_direction(bvec), lvec[:,1:])
    return np.stack([
        boosted_time(lvec, bvec, nr),
        boosted_space(lvec, bvec, nr)
    ])
