import numpy as np
import FileIO as io
import pandas as pd
from astropy.coordinates import search_around_sky
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.table import vstack
import matplotlib.pyplot as plt
import matplotlib.colors as col
import sys

def ReadTablesConcat(
    fname_list,
    invalid_bool_names = ['invalid_beam', 'invalid_majmin', 'scintil_dist', 'scintil_corr', 'close_to_ateam', 'close_to_bright'],
    filter_bool_names = ['valid_spike', 'valid_tcg', 'valid_rms']):
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
            data.add_column(Table.Column(name='fname', data=[fname for _ in range(len(data))]))
            table_list.append(data)
        except:
            print(f'{fname} not found')

    data = vstack(table_list)
    return data

def FindGroups(data, match_sep = 1*u.arcmin, sn_name='peak_sn'):
    if 'group_idx' not in data.columns:
        data.add_column(Table.Column(name='group_idx', data=np.zeros(len(data), dtype=np.int64)))
    # if 'group_sep' not in data.columns:
    #     data.add_column(Table.Column(name='group_sep', data=np.zeros(len(data), dtype=np.float64)))
    if 'group_len' not in data.columns:
        data.add_column(Table.Column(name='group_len', data=np.ones(len(data), dtype=np.int64)))


    data.add_column(Table.Column(name='group_max_sn', data=np.ones(len(data), dtype=np.float64)))
    data.add_column(Table.Column(name='group_max_ds', data=np.ones(len(data), dtype=np.float64)))
    data.add_column(Table.Column(name='group_mean_sn', data=np.ones(len(data), dtype=np.float64)))
    data.add_column(Table.Column(name='group_mean_ds', data=np.ones(len(data), dtype=np.float64)))

    cat = SkyCoord(data['ra_deg'], data['dec_deg'], unit=(u.deg, u.deg), frame="fk5")
    idx1, idx2, sep, _ = search_around_sky(cat, cat, match_sep)
    sort_idx = np.argsort(idx1)
    idx1 = idx1[sort_idx]
    idx2 = idx2[sort_idx]
    sep = sep[sort_idx]

    group_idx = np.full(len(data), -1, dtype=np.int64)
    group_count = 0
    groups = [set() for _ in range(len(data))]

    for i in range(len(idx1)):
        groups[idx1[i]].add(idx2[i])

    for i in range(len(groups)-1, 0, -1):
        concat = list(groups[i])
        while len(concat) > 0:
            if concat[0] != i:
                concat += list(groups[concat[0]].difference(groups[i]))
                groups[i] = groups[i].union(groups[concat[0]])
                groups[concat[0]] = set()
            del concat[0]

    # print(groups)
    # exit()

    new_groups = []
    group_count = 0
    for group in groups:
        group = list(group)
        if len(group) > 0:
            data['group_idx'][group] = group_count
            data['group_len'][group] = len(group)

            data['group_max_sn'][group] = np.max(data[sn_name][group])
            data['group_max_ds'][group] = np.max(data['det_stat'][group])
            data['group_mean_sn'][group] = np.mean(data[sn_name][group])
            data['group_mean_ds'][group] = np.mean(data['det_stat'][group])

            group_count += 1
            
        if len(group) > 1:
            new_groups.append(vstack(data[group]))

    # for i in range(len(idx1)):
    #     if idx1[i] >= idx2[i]:
    #         if group_idx[idx1[i]] == -1:
    #             group_idx[idx1[i]] = group_count
    #             groups.append([{'cand':data[idx1[i]], 'sep':sep[i].arcmin, 'row':idx1[i]}])
    #             group_count += 1
    #         if idx1[i] != idx2[i]:
    #             if group_idx[idx2[i]] == -1:
    #                 group_idx[idx2[i]] = group_idx[idx1[i]]
    #                 groups[group_idx[idx1[i]]].append({'cand':data[idx2[i]], 'sep':sep[i].arcmin, 'row':idx2[i]})
                

    # Finding groups of candidates
    '''
    groups = []
    marked = np.zeros(len(data), dtype=bool)
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
    '''

    # new_groups = []
    # for i in range(len(groups)):
    #     for row in groups[i]:
    #         row['cand']['group_idx'] = i
    #         row['cand']['group_sep'] = sep[i].arcmin
    #         row['cand']['group_len'] = len(groups[i])
        
    #     if len(groups[i]) > 1:
    #         new_groups.append(vstack([x['cand'] for x in groups[i]]))

    return new_groups

def PrintGroups(groups):

    # PRINTING

    group_lengths = [len(np.unique([cand['cand']['obs_id'] for cand in group])) for group in groups]
    sort_idx = np.argsort(group_lengths)
    group_lengths_2 = [group_lengths[i] for i in sort_idx]
    group_lengths = group_lengths_2
    groups = [groups[i] for i in sort_idx]
    i = 0
    obsids = []
    cands = []
    for group in groups:
        if len(group) > 1 and len(group) == group_lengths[i]:

            times = np.array([int(row['cand']['obs_id']) + row['cand']['peak_frame']*4 for row in group])
            times = np.sort(times)
            test_period = times[-1] - times[0]
            period = 0
            residual = 100000
            while test_period > 120:
                test_residual = np.mean((times[1:-1] - times[0]) % test_period)
                if test_residual < residual:
                    residual = test_residual
                    period = test_period
                test_period /= 2

            if False: #residual > 16:
                continue
            else:
                print('period:', period, '    residual:', residual)

            image_link = 'feh '

            for row in group:
                cand = row['cand']
                coord = SkyCoord(ra=cand['ra_deg'], dec=cand['dec_deg'], unit='deg', frame='fk5')
                print_vals = [cand['obs_id'], cand['cand_id'], int(cand['obs_cent_freq']/1e6), cand['fname'], cand['area_pix'], cand['peak_flux'], cand['spike'], row['sep'], coord.ra.to_string(u.hour), coord.dec.to_string(u.degree), cand['spike_norm'], cand['tcg_norm'], cand['rms_norm']]
                print('%10s %5d %4d MHz %9.9s %4d pix %10.4f Jy %10.4f std %10.4f arcmin %20s %20s %10.4f spike %10.4f tcg %10.4f rms' % tuple(print_vals))
                image_link += '%9.9s_candidates/%s_candidate_%d.png ' % (cand['fname'], cand['obs_id'], cand['cand_id'])
                obsids.append(cand['obs_id'])
                cands.append(cand)
            print(image_link)
            print('--------------------------')
        i += 1

    print('\n'.join(np.unique(obsids)))

if __name__ == '__main__':
    fname_list = sys.argv[1:]
    data = ReadTablesConcat(fname_list)
    groups = FindGroups(data)
    PrintGroups(groups)

'''
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
'''