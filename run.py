from transient_search import *
import pandas as pd
import os
import numpy as np

obslist = pd.read_csv('/media/septagonic/CORSAIR/gxarchive/obs_data.csv')
path = '/media/septagonic/CORSAIR/gxarchive/{0}/{0}_{1}'
obs_idx = 0

freqs = np.unique(obslist.freq)
obsids = []
obsdats = []
for freq in freqs:
    obsids.append(obslist.obsid[obslist.freq == freq].values[:20])
    obsdats.append(obslist[obslist.freq == freq].iloc[:20])
obsids = np.concatenate(obsids)
obslist = pd.concat(obsdats)
obslist.to_csv('100_obs.csv')

# filters = [
#     Filter('tcg'  , 8, True , fil.Correlator, (5,1,1), (25,1,1)),
#     Filter('spike', 7, False, fil.Spike, 4),
#     Filter('rms'  , 3, True , fil.RMS)]

# filters = [
#     Filter('tcg0.1', 7.5, True , fil.Correlator, (0.1,1,1), (25,1,1)),
#     Filter('tcg1'  , 7.5, True , fil.Correlator, (1  ,1,1), (25,1,1)),
#     Filter('tcg5'  , 7.5, True , fil.Correlator, (5  ,1,1), (25,1,1))]

# NEW NEW ULTRA EPIC FILTER SETTING !!!!!!!!
filters = [
    Filter('tcg'  , 5.5, 8, True , fil.Correlator, (1,1,1), (25,1,1)),
    Filter('spike', 5.5, 8, False, fil.Spike, 4),
    Filter('rms'  , 2.0, 3, True , fil.RMS)]

for obsid in obsids:
    print('--------------', obsid, '------', obs_idx)
    obs_idx += 1
    cands = TransientSearch(path, obsid, filters, 'mod', False)
    os.system(f'cp {path.format(obsid, "*.png")} {path.format(obsid, "*.gif")} ./candidates')
    if len(cands) > 10:
        os.system(f'echo {obsid}, {len(cands)} >> bad_obsids.txt')