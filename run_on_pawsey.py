from transient_search import *
import pandas as pd
import os

obslist = pd.read_csv('obsids_shif.txt')
path = '~/candidates/{0}/{0}_{1}'
obs_idx = 0

filters = [
    Filter('tcg'  , 5.5, 8, True , fil.Correlator, (1,1,1), (25,1,1)),
    Filter('spike', 5.5, 8, False, fil.Spike, 4),
    Filter('rms'  , 2.0, 3, True , fil.RMS)]

for obsid in obslist.obsids:
    print('--------------', obsid, '------', obs_idx)
    obs_idx += 1
    cube_fname = path.format(obsid, 'transient.hdf5')
    if not os.path.isfile(cube_fname):
        os.system('mkdir ~/candidates/{0}'.format(obsid))
        os.system('scp ubuntu@146.118.68.233:/mnt/gxarchive/Archived_Obsids/{0}/{0}_transient.hdf5 {1}'.format(obsid, cube_fname))
        if os.path.isfile(cube_fname):
            cands = TransientSearch(path, obsid, filters, 'mod', False)
            if len(cands) > 10:
                os.system(f'echo {obsid}, {len(cands)} >> bad_obsids.txt')
            os.system(f'rm {cube_fname}')