import numpy as np
import FileIO as io
import pandas as pd
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.table import vstack
import matplotlib.pyplot as plt
import matplotlib.colors as col

obslist = pd.read_csv('/media/septagonic/CORSAIR/gxarchive/obs_data.csv')
table_list = []

for obsid in obslist.obsid:
    table_list.append(Table.read(f'/media/septagonic/CORSAIR/gxarchive/{obsid}/{obsid}_islands_selected.fits', format='fits'))

data = vstack(table_list)
cat = SkyCoord(data['ra_deg'], data['dec_deg'], unit=(u.deg, u.deg), frame="fk5")
idx, sep, _ = match_coordinates_sky(cat, cat, nthneighbor=2)
marked = np.zeros(len(data), dtype=bool)
groups = []
match_sep = 0.0666667 * 2

# Finding groups of candidates
for i in range(len(data)):
    if not marked[i]:
        marked[i] = True
        group = [data[i]]
        j = i
        # While I have not visited you before, and you are close, add to group
        while (not marked[idx[j]]) and (sep[j] < match_sep*u.deg):
            j = idx[j]
            group.append(data[j])
            marked[j] = True
        # Add group to list of groups if they originate from at least 2 obsids
        # group_obsids = [x['obs_id'] for x in group]
        # if len(np.unique(group_obsids)) == len(group):
        #     groups.append(group)
        groups.append(group)

group_lengths = [len(x) for x in groups]
sort_idx = np.argsort(group_lengths)
groups = [groups[i] for i in sort_idx]
for group in groups:
    if len(group) > 1:
        for row in group:
            print(row['obsid'], row['can_idx'], int(row['cent_freq']/1e6))
        print('--------------------------')

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
