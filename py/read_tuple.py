#! /usr/bin/env python

import os
import sys
import uproot
import numpy as np

from bevent import Jpsi2piKEvent

BASE_PATH = '/home/vitaly/work/lhcb/Kirill/fit_data_unbinned_jpsipippimkp'
BASE_PATH_NEW = '/media/vitaly/4759e668-4a2d-4997-8dd2-eb4d25313d90/vitaly/work/fit_data_unbinned_jpsipippimkp'
DATA_FILE = 'res.root'

def get_path(phase=None):
    if phase is None:
        folder = 'run_signal_belle1_chic13872_vorobyev_bondar_no_interference'
    else:
        folder = f'run_signal_belle1_chic13872_vorobyev_bondar_interference_{phase}'
    
    return os.path.join(BASE_PATH_NEW, folder, DATA_FILE)

def get_path_new(phase=None):
    suffix = {0: 'phase_0',90: 'phase_90',180: 'phase_180',270: 'phase_270','none': 'no_interference'}
    phase = phase if phase is not None else 'none'
    return os.path.join(BASE_PATH_NEW, f'toy_mc_{suffix[phase]}.root')

def read_tree(phase=None, router=get_path_new, tree='t_mc'):
    file = uproot.open(router(phase))
    print(file.keys())
    events = file[f'{tree}/event']
    # pdf = file[f'{tree}/pdf']

    print(events.keys())

    data = Jpsi2piKEvent(
        events['deltaE'].array(),
        events['mbc'].array(),
        events['pi']['pi.momentum[4]'].array(),
        events['pi']['pi.momentumMC[4]'].array(),
        events['pi2']['pi2.momentum[4]'].array(),
        events['pi2']['pi2.momentumMC[4]'].array(),
        events['psi2s']['psi2s.jpsi.momentum[4]'].array(),
        events['psi2s']['psi2s.jpsi.momentumMC[4]'].array(),
        events['k']['k.momentum[4]'].array(),
        events['k']['k.momentumMC[4]'].array(),
        events['k']['k.kp.charge'].array(),
        np.ones(events['mbc'].array().shape),
        # pdf.array(),
        events['bDaughters[5]'].array(),
        events['foxWolfram[5]'].array(),
    )

    data.serialize(f'data_{tree}_{phase}')


def main():
    phase = int(sys.argv[1]) if len(sys.argv) > 1 else None
    read_tree(phase, tree='t1')


if __name__ == "__main__":
    main()

