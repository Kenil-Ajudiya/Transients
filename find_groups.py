import numpy as np
import FileIO as io
import pandas as pd
from astropy.coordinates import match_coordinates_sky, search_around_sky
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.table import vstack
import matplotlib.pyplot as plt
import matplotlib.colors as col
import os
import sys

# from sklearn.cluster import DBSCAN


# obslist = pd.read_csv('/media/septagonic/CORSAIR/gxarchive/obs_data.csv')
# fname_list = [f'/media/septagonic/CORSAIR/gxarchive/{obsid}/{obsid}_islands_selected.fits' for obsid in obslist.obsid]

# fname_list = ['/home/septagonic/Documents/Transients/islands_setonix/' + fname for fname in os.listdir('/home/septagonic/Documents/Transients/islands_setonix/')]
if len(sys.argv == 3):
    fname_list = [f'{sys.argv[1]}/{fname}/{fname}_{sys.argv[2]}' for fname in os.listdir(sys.argv[1])]
else:
    fname_list = [sys.argv[1]]

# obslist = pd.read_csv('100_obs.csv')
# fname_list = [f'/media/septagonic/CORSAIR/gxarchive/{x}/{x}_4_islands.fits' for x in obslist.obsid]

invalid_bool_names = ['invalid_beam', 'invalid_majmin', 'scintil_dist', 'scintil_corr', 'close_to_ateam', 'close_to_bright']
filter_bool_names = ['valid_spike', 'valid_tcg', 'valid_rms']

table_list = []
for fname in fname_list:
    try:
        data = Table.read(fname, format='fits')
        # data['valid_tcg'] = data['tcg_norm'] > 0.8
        # data['valid_rms'] = data['rms_norm'] > 0.8
        # data['invalid_beam'] = data['beam_norm'] < 0.25
        # data = data[np.logical_or.reduce([data[name].value for name in filter_bool_names])]
        # data = data[~np.logical_or.reduce([data[name].value for name in invalid_bool_names])]
        print(f'{fname} read')
        table_list.append(data)
    except:
        print(f'{fname} not found')

data = vstack(table_list)
# data.write('combined.fits', format='fits')
# data = data[data['obs_cent_freq'] > 100e6]
# data = data[data['nks1_sep_deg'] > 0.1]
# data = data[data['nks2_sep_deg'] > 0.1]
cat = SkyCoord(data['ra_deg'], data['dec_deg'], unit=(u.deg, u.deg), frame="fk5")
# idx, sep, _ = match_coordinates_sky(cat, cat, nthneighbor=2)
marked = np.zeros(len(data), dtype=bool)
groups = []
match_sep = 1*u.arcmin

# def cluster(data, epsilon,N): #DBSCAN, euclidean distance
#     db     = DBSCAN(eps=epsilon, min_samples=N).fit(data)
#     labels = db.labels_ #labels of the found clusters
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #number of clusters
#     clusters   = [data[labels == i] for i in range(n_clusters)] #list of clusters
#     return clusters, n_clusters

# centers = [[1, 1,1], [-1, -1,1], [1, -1,1]]
# cluster(X,epsilon,N)
    

# Finding groups of candidates

idx1, idx2, sep, _ = search_around_sky(cat, cat, match_sep)
sort_idx = np.argsort(idx1)
idx1 = idx1[sort_idx]
idx2 = idx2[sort_idx]
sep = sep[sort_idx]
cur_idx = -1
group = []

for i in range(len(idx1)):
    if cur_idx != idx1[i]:
        groups.append(group)
        group = []
        if marked[idx1[i]]:
            continue
        group = [{'cand':data[idx1[i]], 'sep':0}]
        marked[idx1[i]] = True
        cur_idx = idx1[i]
    if not marked[idx2[i]]:
        marked[idx2[i]] = True
        group.append({'cand':data[idx2[i]], 'sep':sep[i].arcmin})
        



# for i in range(len(data)):
#     if not marked[i]:
#         marked[i] = True
#         group = [{'cand':data[i], 'sep':0}]
#         j = i
#         # While I have not visited you before, and you are close, add to group
#         while (not marked[idx[j]]) and (sep[j] < match_sep):
#             j = idx[j]
#             group.append({'cand':data[j], 'sep':sep[j].arcmin})
#             marked[j] = True
#         groups.append(group)


group_lengths = [len(np.unique([cand['cand']['obs_id'] for cand in group])) for group in groups]
sort_idx = np.argsort(group_lengths)
groups = [groups[i] for i in sort_idx]
for group in groups:
    if len(group) > 0:
        for row in group:
            cand = row['cand']
            coord = SkyCoord(ra=cand['ra_deg'], dec=cand['dec_deg'], unit='deg', frame='fk5')
            print_vals = [cand['obs_id'], cand['cand_id'], int(cand['obs_cent_freq']/1e6), cand['area_pix'], cand['peak_flux'], cand['spike'], row['sep'], coord.ra.to_string(u.hour), coord.dec.to_string(u.degree)]
            print('%10s %5d %4d MHz %4d pix %10.4f Jy %10.4f std %10.4f arcmin %20s %20s' % tuple(print_vals))
        print('--------------------------')

# valid = data['obs_cent_freq'] > 120e6
# valid = (data['cand_id'] < 1000) & (data['nks_sep_deg'] > 0.5) & (data['nks2_sep_deg'] > 0.5)
# data = data[valid]
# flux_sort = np.argsort(data['peak_flux'])
# for i in flux_sort:
#     print(data[i]['obs_id'], data[i]['cand_id'], data[i]['peak_flux'])

exit()

ncands = np.array([len(x) for x in table_list])
bins=np.arange(np.max(ncands)+2)
plt.hist(ncands, log=True, bins=bins)
plt.xlabel('Number of candidates')
plt.ylabel('Number of observations')

plt.figure()
freq_list = np.unique(obslist['freq'])
bins=np.arange(np.max(ncands)+2)
hist, _ = np.histogram(ncands, bins=bins, density=True)
plt.stairs(hist, edges=bins, label=f'Overall', alpha=0.5, fill=True, lw=2)
norm = col.Normalize()
colors = plt.cm.jet(norm(freq_list))
for i in range(len(freq_list)):
    hist, _ = np.histogram(ncands[obslist['freq'] == freq_list[i]], bins=bins, density=True)
    plt.stairs(hist, edges=bins, label=f'{int(freq_list[i]/1e6)} MHz', lw=1, color=colors[i])
plt.legend()
plt.yscale('log')
plt.xlabel('Number of candidates per observation')
plt.ylabel('Fraction of observations at frequency')
plt.yticks(ticks=[0.01, 0.1, 1], labels=['1%', '10%', '100%'])

plt.show()
