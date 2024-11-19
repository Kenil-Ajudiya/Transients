import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import find_groups as fg
from astropy.table import Table
import pickle as pkl
from transient_search import Filter
import filters as fil
import os

filters = [ # ------------------------------------------- SAM
    Filter('tcg'  , 5.5, 7.0, True , fil.Correlator, (1,1,1), (125,1,1)),
    Filter('spike', 5.5, 7.5, False, fil.Spike, 3),
    Filter('rms'  , 2.0, 2.25, True , fil.RMS)]

if False:
    fname = '/media/septagonic/CORSAIR/candidates/try_1_islands_selected.fits'
    data = fg.ReadTablesConcat([fname])

    # Apply max pixel area
    max_area = 20
    data.add_column(Table.Column(name='invalid_area', data=data['area_pix'] > max_area))
    data = data[~data['invalid_area']]

    # tcg_norm = tcg / tcg_cut_high = tcg / (obs_rms * cut_high_unscaled)
    # obs_rms = tcg / (tcg_norm * cut_high_unscaled
    data.add_column(Table.Column(name='obs_rms', data=data['tcg'].data/(data['tcg_norm'].data*filters[0].cut_high_unscaled)))
    data.add_column(Table.Column(name='peak_sn', data=data['peak_flux'].data/data['obs_rms'].data))

    # with open('groups.pkl', 'rb') as f:
    #     data, _ = pkl.load(f)

    groups = fg.FindGroups(data)
    with open('groups.pkl', 'wb') as f:
        pkl.dump([data, groups], f)
else:
    with open('groups.pkl', 'rb') as f:
        data, groups = pkl.load(f)

print('Number of candidates:', len(data))
print('Number of candidates in a group:', np.count_nonzero(data['group_len'] > 1))

group_sizes = np.array([len(x) for x in groups])
group_max_sn = np.array([np.max(x['peak_sn']) for x in groups])
group_max_ds = np.array([np.max(x['det_stat']) for x in groups])
group_mean_sn = np.array([np.mean(x['peak_sn']) for x in groups])
group_mean_ds = np.array([np.mean(x['det_stat']) for x in groups])

ng = data[data['group_len'] == 1]
group_sizes = np.concatenate([group_sizes, np.ones(len(ng))])
group_max_sn = np.concatenate([group_max_sn, ng['peak_sn'].data])
group_mean_sn = np.concatenate([group_mean_sn, ng['peak_sn'].data])
group_max_ds = np.concatenate([group_max_ds, ng['det_stat'].data])
group_mean_ds = np.concatenate([group_mean_ds, ng['det_stat'].data])

in_group = data['group_len'] > 1
plt.scatter(data[in_group]['ra_deg'], data[in_group]['dec_deg'], c=data[in_group]['group_len'])
plt.xlabel('RA')
plt.ylabel('Dec')
plt.colorbar()

for group in groups:
    for cand in group[1:]:
        plt.plot([group['ra_deg'][0], cand['ra_deg']], [group['dec_deg'][0], cand['dec_deg']], c='blue')

plt.show()

fig, axs = plt.subplots(3, 2)

axs[0,0].hist(data['peak_sn'], bins=20)
axs[0,0].set_xlabel('S/N')
axs[0,0].set_ylabel('Number of candidates')

axs[1,0].hist(data['group_len'], bins=20)
axs[1,0].set_xlabel('Group size')
axs[1,0].set_ylabel('Number of candidates')
axs[1,0].set_yscale('log')

axs[2,0].scatter(group_sizes, group_mean_sn, label='Mean')
axs[2,0].scatter(group_sizes, group_max_sn, label='Max')
axs[2,0].set_xlabel('Groups size')
axs[2,0].set_ylabel('Group S/N')
axs[2,0].set_xscale('log')
axs[2,0].set_yscale('log')
axs[2,0].legend()

cutoff = 10
x = np.arange(1,140)
y = cutoff/np.sqrt(x)
axs[2,0].plot(x,y,color='black')
selection = data['group_mean_sn']>cutoff/np.sqrt(data['group_len'])
print('Number of down-selected groups:', len(np.unique(data[selection]['group_idx'].data)))
print('Number of down-selected candidates:', len(data[selection]))

axs[0,1].hist(data['det_stat'], bins=20)
axs[0,1].set_xlabel('det_stat')
axs[0,1].set_ylabel('Number of candidates')

# axs[1,1].hist(data['group_len'], bins=20)
# axs[1,1].set_xlabel('Group size')
# axs[1,1].set_ylabel('Number of candidates')
# axs[1,1].set_yscale('log')

axs[2,1].scatter(group_sizes, group_mean_ds, label='Mean')
axs[2,1].scatter(group_sizes, group_max_ds, label='Max')
axs[2,1].set_xlabel('Groups size')
axs[2,1].set_ylabel('Group det_stat')
axs[2,1].set_xscale('log')
axs[2,1].set_yscale('log')
axs[2,1].legend()

stem = '/media/septagonic/CORSAIR'
src = stem+'/candidates'
dest = stem+'/groups_sort_named'
for row in data[selection]:
    for ext in ['png', 'gif']:
        fname_src = f"{row['obs_id']}_candidate_{row['cand_id']}.{ext}"
        fname_dest = f"{row['group_len']:03}_{row['group_idx']:05}_{row['obs_id']}_tcg_spike_rms_{row['cand_id']:03}.{ext}"
        os.system(f"cp {src}/{fname_src} {dest}/{fname_dest}")
    # if row['cand_id'] == 52:
    #     print(row['group_len'], fname)

name_changes = {'peak_flux'    : 'can_peak_flux'   ,
        'beam'         : 'can_beam'        ,
        'det_stat'     : 'can_det_stat'    ,
        'nks_flux_rat' : 'can_nks_flux_rat'}
for key in name_changes:
    data.rename_column(key, name_changes[key])

# data[selection].write('/media/septagonic/CORSAIR/candidates_group_selected/group_islands_selected.fits', format='fits', overwrite=True)

plt.show()