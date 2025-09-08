import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
import find_groups as fg
from astropy.table import Table
import pickle as pkl
from transient_search import Filter
import filters as fil
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
import os

from scipy.stats import norm

def Fixup(in_fname, out_fname, plot=False):

    filters = [ # ------------------------------------------- SAM
        Filter('tcg'  , 5.5, 7.0, True , fil.Correlator, (1,1,1), (125,1,1)),
        Filter('spike', 5.5, 7.5, False, fil.Spike, 3),
        Filter('rms'  , 2.0, 2.25, True , fil.RMS)]

    log_scale = True

    if log_scale:
        bin_func = np.geomspace
    else:
        bin_func = np.linspace

    # sn_name = 'tcg_norm'
    # sn_name = 'peak_sn'
    sn_name = 'fluence_sn'
    # sn_name = 'det_stat'
    match_sep = 4.0*u.arcmin

    if True:
        if not isinstance(in_fname, list):
            in_fname = [in_fname]
        data = fg.ReadTablesConcat(in_fname)

        # Apply max pixel area and max mean
        max_area = 20
        data.add_column(Table.Column(name='mean', data=np.mean(data['curve'].data, axis=1)))
        # data.add_column(Table.Column(name='fluence', data=np.sum(np.abs(data['curve'].data), axis=1) * 4.0))
        data.add_column(Table.Column(name='fluence', data=np.sum(np.abs(data['curve'].data), axis=1) * 4.0))
        # data.add_column(Table.Column(name='fluence', data=np.sum(np.abs(data['curve'].data - np.min(data['curve'].data, axis=1, keepdims=True)), axis=1) * 4.0))

        # tcg_norm = tcg / tcg_cut_high = tcg / (obs_rms * cut_high_unscaled)
        # obs_rms = tcg / (tcg_norm * cut_high_unscaled)
        data.add_column(Table.Column(name='obs_rms', data=data['tcg'].data/(data['tcg_norm'].data*filters[0].cut_high_unscaled)))
        data.add_column(Table.Column(name='peak_sn', data=data['peak_flux'].data/data['obs_rms'].data))
        data.add_column(Table.Column(name='fluence_sn', data=data['fluence'].data/(4.0 * data['obs_rms'].data * np.sqrt(data['curve'].data.shape[1]))))

        data.add_column(Table.Column(name='invalid_area', data=data['area_pix'].data > max_area))
        data.add_column(Table.Column(name='invalid_mean', data=np.abs(data['mean'].data) > data['obs_rms'].data))
        data.add_column(Table.Column(name='invalid_freq', data=data['obs_cent_freq'].data < 100e6))
        width, height = 2400, 2400
        margin = 10
        data.add_column(Table.Column(name='on_edge', data=
            (data['x_pix'].data < margin) | (data['x_pix'].data > width-margin) |
            (data['y_pix'].data < margin) | (data['y_pix'].data > height-margin)))
        
        data['peak_sn'] = np.abs(data['peak_sn'])
        data['fluence_sn'] = np.abs(data['fluence_sn'])
        # data['det_stat'][:] = 0.0
        # data['det_stat'][data['valid_tcg']]   += data['tcg_norm'][data['valid_tcg']]**2
        # data['det_stat'][data['valid_spike']] += data['spike_norm'][data['valid_spike']]**2
        # data['det_stat'][data['valid_rms']]   += data['rms_norm'][data['valid_rms']]**2

        # See if any known transients are in here
        if False:
            cat = SkyCoord(data['ra_deg'], data['dec_deg'], unit=(u.deg, u.deg), frame="fk5")
            known_lpt = np.array([
                ['GLEAM-X J162759.5-523504.3', '16h27m59.50s', '-52°35′04.30″'],
                ['GPM J1839-10'              , '18h39m02.00s', '-10°31′49.37″'],
                ['GCRT J1745-3009'           , '17h45m05.00s', '-30°09′52.00″'],
                ['ASKAP J193505.1+214841.0'  , '19h35m05.13s', ' 21°48′41.05″'],
                ['ASKAP J175534.9-252749.1'  , '17h55m34.87s', '-25°27′49.10″'],
                ['CHIME J0630+25'            , '06h30m43.00s', ' 25°23′24.00″'],
                ['ILT J110160.52+552119.62'  , '11h01m50.50s', ' 55°21′19.60″'],
                ['GLEAM-X J0704-37'          , '07h04m07.51s', '-37°05′40.56″'],
                ['ASKAP J1839-0756'          , '18h39m50.40s', '-07°56′35.38″'],
                ['ASKAP J1832-0911'          , '18h32m48.46s', '-09°11′15.30″'],
                ['ASKAP J175154.89-255135.3' , '17h51m54.89s', '-25°51′35.30″'],
                ['ASKAP J172755-343119'      , '17h27m55.26s', '-34°31′19.41″']])
            search = SkyCoord(ra=known_lpt[:,1], dec=known_lpt[:,2], frame='fk5')
            idx1, idx2, sep, _ = search_around_sky(search, cat, 1*u.arcmin)
            print('Search results:')
            for i in range(len(idx1)):
                row = data[idx2[i]]
                cand_fname = f"{row['obs_id']}_candidate_{row['cand_id']}.png"
                print(f'{known_lpt[idx1[i],0]:30s} {idx2[i]} {sep[i].arcsec:.3f}" {cand_fname}')
                for key in row.colnames:
                    print(f'{key:20s}{row[key]}')

        # Find groups
        data = data[~(data['invalid_area'] | data['invalid_mean'] | data['on_edge'] | data['invalid_freq'])]
        groups = fg.FindGroups(data, match_sep=match_sep, sn_names=['peak_sn', 'fluence_sn'])

        # Dump group data
        with open('groups2.pkl', 'wb') as f:
            pkl.dump([data, groups], f)
    else:
        with open('groups2.pkl', 'rb') as f:
            data, groups = pkl.load(f)

    print('Number of candidates:', len(data))
    print('Number of candidates in a group:', np.count_nonzero(data['group_len'] > 1))

    # print(groups[0].keys())

    group_sizes = np.array([len(x) for x in groups])
    group_max_sn = np.array([np.max(x[sn_name]) for x in groups])
    group_max_ds = np.array([np.max(x['det_stat']) for x in groups])
    group_mean_sn = np.array([np.mean(x[sn_name]) for x in groups])
    group_mean_ds = np.array([np.mean(x['det_stat']) for x in groups])
    group_ids = np.array([x['group_idx'][0] for x in groups])
    group_ra_deg = np.array([x['ra_deg'][0] for x in groups])
    group_dec_deg = np.array([x['dec_deg'][0] for x in groups])

    ng = data[data['group_len'] == 1]
    group_sizes = np.concatenate([group_sizes, np.ones(len(ng))])
    group_max_sn = np.concatenate([group_max_sn, ng[sn_name].data])
    group_mean_sn = np.concatenate([group_mean_sn, ng[sn_name].data])
    group_max_ds = np.concatenate([group_max_ds, ng['det_stat'].data])
    group_mean_ds = np.concatenate([group_mean_ds, ng['det_stat'].data])
    group_ids = np.concatenate([group_ids, ng['group_idx'].data])
    group_ra_deg = np.concatenate([group_ra_deg, ng['ra_deg'].data])
    group_dec_deg = np.concatenate([group_dec_deg, ng['dec_deg'].data])

    # Pulsars:
    pulsar_group_ids = np.array([13487, 13257, 7911, 12802])
    lpt_group_ids = np.array([13142])
    # pulsar_cand_ids = np.array([158, 50, 32, 67, 30, 79])
    # lpt_cand_ids = np.array([33])
    # pulsar_group_ids = data['group_idx'].data[np.isin(data['cand_id'].data, pulsar_cand_ids)]
    # lpt_group_ids = data['group_idx'].data[np.isin(data['cand_id'].data, lpt_cand_ids)]
    is_pulsar  = np.isin(group_ids, pulsar_group_ids)
    is_lpt = np.isin(group_ids, lpt_group_ids)

    known_lpt = np.array([
        ['GLEAM-X J162759.5-523504.3', '16h27m59.50s', '-52°35′04.30″'],
        ['GPM J1839-10'              , '18h39m02.00s', '-10°31′49.37″'],
        ['GCRT J1745-3009'           , '17h45m05.00s', '-30°09′52.00″'],
        ['ASKAP J193505.1+214841.0'  , '19h35m05.13s', ' 21°48′41.05″'],
        ['ASKAP J175534.9-252749.1'  , '17h55m34.87s', '-25°27′49.10″'],
        ['CHIME J0630+25'            , '06h30m43.00s', ' 25°23′24.00″'],
        ['ILT J110160.52+552119.62'  , '11h01m50.50s', ' 55°21′19.60″'],
        ['GLEAM-X J0704-37'          , '07h04m07.51s', '-37°05′40.56″'],
        ['ASKAP J1839-0756'          , '18h39m50.40s', '-07°56′35.38″'],
        ['ASKAP J1832-0911'          , '18h32m48.46s', '-09°11′15.30″'],
        ['ASKAP J175154.89-255135.3' , '17h51m54.89s', '-25°51′35.30″'],
        ['ASKAP J172755-343119'      , '17h27m55.26s', '-34°31′19.41″']])
    lpt_coords = SkyCoord(ra=known_lpt[:,1], dec=known_lpt[:,2], frame='fk5')
    psrs = fits.open(os.getenv('ATNF_PULSAR_CAT', "~/Documents/MWA-GPM-data/atnf_pulsar_cat.fits"))[1].data
    psr_coords = SkyCoord(psrs["RAJ2000"], psrs["DEJ2000"], unit=(u.deg, u.deg), frame='fk5')
    group_coords = SkyCoord(group_ra_deg, group_dec_deg, unit=(u.deg, u.deg), frame='fk5')
    _, sep, _ = group_coords.match_to_catalog_sky(psr_coords)
    is_pulsar |= sep < 2*u.arcmin
    _, sep, _ = group_coords.match_to_catalog_sky(lpt_coords)
    is_lpt = sep < 2*u.arcmin

    print('Pulsar group IDs:', group_ids[is_pulsar])

    in_group = data['group_len'] > 1
    # plt.scatter(data[in_group]['ra_deg'], data[in_group]['dec_deg'], c=data[in_group]['group_len'])
    # plt.xlabel('RA')
    # plt.ylabel('Dec')
    # plt.colorbar()

    # for group in groups:
    #     for cand in group[1:]:
    #         plt.plot([group['ra_deg'][0], cand['ra_deg']], [group['dec_deg'][0], cand['dec_deg']], c='blue')

    # plt.show()

    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios':[0.3,0.7], 'width_ratios':[0.7, 0.3]}, figsize=(5.45,3.8))
    fig.subplots_adjust(left=0.12, bottom=0.14, right=0.96, top=0.96, wspace=0, hspace=0)

    axs[0,0].hist(group_sizes, bins=int(np.max(group_sizes))-1)
    axs[0,0].set_ylabel('No. groups')
    axs[0,0].set_xticks([])
    axs[0,0].set_xscale('log')
    axs[0,0].set_yscale('log')

    axs[1,1].hist(group_mean_sn, bins=bin_func(np.nanmin(group_mean_sn), np.nanmax(group_mean_sn), 20), orientation='horizontal', density=False)
    axs[1,1].set_xlabel('No. groups')
    axs[1,1].set_xscale('log')
    if log_scale:
        axs[1,1].set_yscale('log')
    axs[1,1].set_yticks([])

    axs[0,1].axis('off')

    def ColumnNormedHist2d(x, y, axs):
        valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        x = x[valid]
        y = y[valid]
        xbins = bin_func(np.min(x), np.max(x), 20, endpoint=True)
        ybins = np.linspace(np.min(y), np.max(y), 100, endpoint=True)
        img, xbins, ybins = np.histogram2d(x, y, bins=(xbins, ybins))
        # print(xbins[0], xbins[-1], ybins[0], ybins[-1])
        img /= np.sum(img, axis=1, keepdims=True)
        axs.pcolor(xbins, ybins, img.T, norm=matplotlib.colors.LogNorm())

    axs[1,0].scatter(group_sizes, group_mean_sn, label='Candidates', s=10, zorder=10, color='blue', marker='x')
    axs[1,0].scatter(group_sizes[is_pulsar], group_mean_sn[is_pulsar], label='Pulsars', s=40, zorder=10, color='red', marker='x')
    axs[1,0].scatter(group_sizes[is_lpt], group_mean_sn[is_lpt], label='LPTs', s=100, zorder=10, marker="*", color='red')
    '''
    axs[1,0].text(2.2, 15.68891971 * 0.8, "a", verticalalignment="center") # "PSR J0630-2834"  
    axs[1,0].text(6.6, 6.67617366  * 0.8, "b", verticalalignment="center") # "PSR J0031-57"    
    axs[1,0].text(2.2, 25.27475862 * 0.8, "c", verticalalignment="center") # "PSR J0437-4715"  
    axs[1,0].text(3.3, 7.46751458  * 0.8, "d", verticalalignment="center") # "PSR J0410-31"    
    axs[1,0].text(22 , 15.36910853 * 0.8, "e", verticalalignment="center") # "PSR J2048-1616"  
    axs[1,0].text(53 , 25.95448242 * 0.8, "f", verticalalignment="center") # "PSR J0034-0721"  
    axs[1,0].text(1.1, 3.6028995   * 0.8, "g", verticalalignment="center") # "PSR J2241-5236"  
    axs[1,0].text(1.1, 2.66011573  * 0.8, "h", verticalalignment="center") # "PSR J0502-6617"  
    axs[1,0].text(1.1, 4.66265338  * 0.8, "i", verticalalignment="center") # "PSR J1244-1812"  
    axs[1,0].text(1.1, 10.8442     * 0.8, "j", verticalalignment="center") # "GLEAM-X J0704-37"
    '''
    axs[1,0].text(2.2,  9.0, "a", verticalalignment="center") # "PSR J0630-2834"  
    axs[1,0].text(6.6,  8.6, "b", verticalalignment="center") # "PSR J0031-57"    
    axs[1,0].text(2.2, 10.5, "c", verticalalignment="center") # "PSR J0437-4715"  
    axs[1,0].text(3.3,  9.6, "d", verticalalignment="center") # "PSR J0410-31"    
    axs[1,0].text(22 ,  7.5, "e", verticalalignment="center") # "PSR J2048-1616"  
    axs[1,0].text(53 , 13.3, "f", verticalalignment="center") # "PSR J0034-0721"  
    axs[1,0].text(1.1,  4.2, "g", verticalalignment="center") # "PSR J2241-5236"  
    axs[1,0].text(1.1,  3.0, "h", verticalalignment="center") # "PSR J0502-6617"  
    axs[1,0].text(1.1,  4.9, "i", verticalalignment="center") # "PSR J1244-1812"  
    axs[1,0].text(1.1,  5.9, "j", verticalalignment="center") # "GLEAM-X J0704-37"

    axs[1,0].set_xlabel('Group size')
    # axs[1,0].set_ylabel('Group mean fluence S/N')
    axs[1,0].set_ylabel('Group mean peak S/N')
    # axs[1,0].set_ylabel('Group mean TCG / RMS')
    axs[1,0].set_xscale('log')
    if log_scale:
        axs[1,0].set_yscale('log')
    xlim, ylim = axs[1,0].get_xlim(), axs[1,0].get_ylim()
    # print(ylim)

    def CutoffFunction(sizes, cutoff, P0):
        return cutoff / np.sqrt(sizes)
        # return -norm.ppf((norm.cdf(-cutoff))**(1/sizes))
        # return -norm.ppf((((1-P0)*norm.cdf(-cutoff) + P0)**(1/sizes) - P0) / (1-P0))

    cutoff = 7
    x = bin_func(1, 140, 1000)
    P0 = 0.001
    y = CutoffFunction(x, cutoff, P0)
    axs[1,0].plot(x,y,color='black', label='Cutoff', zorder=1000)
    # axs[1,0].hlines(5, 1, 70, color='black', ls='--', label='S/N Cutoff 2', zorder=1000)
    axs[1,0].set_xlim(xlim)
    axs[1,0].set_ylim(ylim)
    axs[1,0].legend()
    axs[1,0].legend(bbox_to_anchor=(0.97, 0.98), bbox_transform=fig.transFigure)
    # print('Number of candidates with S/N greater than 10.87:', np.count_nonzero(data['fluence_sn'] > 10.87))
    selection = data['group_mean_fluence_sn'] > CutoffFunction(data['group_len'], cutoff, P0)
    sel_1_ngroups = np.unique(data['group_idx'][selection]).size
    selection_2 = data['group_mean_peak_sn'] > CutoffFunction(data['group_len'], cutoff, P0)
    sel_2_ngroups = np.unique(data['group_idx'][selection_2]).size
    group_selection = selection | selection_2
    data.add_column(Table.Column(name='valid_group_mean_peak_sn', data=selection))
    data.add_column(Table.Column(name='valid_group_mean_fluence_sn', data=selection_2))

    print(sel_1_ngroups, sel_2_ngroups)
    print(np.unique(data['group_idx'][(~selection) & selection_2]))

    print('Candidates above cutoff:', np.count_nonzero(selection))
    print('Candidates below cutoff:', np.count_nonzero(~selection))

    print('Valid group IDs:', repr(np.unique(data['group_idx'].data[selection])))

    # group_selection = group_mean_sn > CutoffFunction(group_sizes, cutoff, P0)
    print('number of groups:', group_selection.size)
    print('Groups above cutoff:', np.count_nonzero(group_selection))
    print('Groups below cutoff:', np.count_nonzero(~group_selection))

    peak_ids = np.\
    array([  154,   609,  1185,  1820,  1823,  2811,  3500,  3542,  5165,
            5800,  6209,  7244,  7274,  7958,  8044,  8288,  8413,  8854,
            9043,  9169,  9535,  9843, 11004, 11197, 11201, 11679, 12199,
        12743, 12958, 13194, 13311, 13413, 13643])

    fluence_ids = np.\
    array([  154,   168,   169,   170,   291,   408,   442,   609,   656,
            659,   660,   661,   662,   663,   693,   694,   715,   780,
            850,   857,   859,   869,   871,   872,  1012,  1021,  1055,
            1152,  1184,  1185,  1215,  1220,  1262,  1263,  1384,  1528,
            1529,  1530,  1531,  1532,  1533,  1534,  1535,  1768,  1836,
            1857,  1881,  1882,  1883,  1895,  1897,  1899,  2251,  2253,
            2375,  2378,  2379,  2800,  2811,  2828,  2832,  2904,  3056,
            3078,  3083,  3100,  3134,  3292,  3337,  3498,  3499,  3500,
            3542,  3546,  3630,  3652,  3653,  3655,  3656,  3719,  3746,
            3842,  3847,  3848,  3849,  3850,  3852,  3855,  3871,  3885,
            3886,  3887,  3888,  3924,  3933,  3938,  4099,  4248,  4249,
            4470,  4471,  4473,  4570,  4613,  4614,  4621,  4789,  4814,
            4920,  4947,  5158,  5163,  5164,  5165,  5166,  5167,  5178,
            5358,  5407,  5420,  5436,  5437,  5800,  5909,  5912,  5917,
            5918,  6024,  6044,  6056,  6057,  6070,  6085,  6128,  6146,
            6177,  6178,  6179,  6264,  6266,  6295,  6322,  6366,  6400,
            6401,  6434,  6435,  6506,  6508,  6718,  6769,  6771,  6772,
            6813,  6814,  6815,  6824,  6953,  6955,  6956,  6957,  7056,
            7078,  7079,  7193,  7222,  7274,  7276,  7277,  7278,  7332,
            7333,  7334,  7399,  7425,  7530,  7610,  7611,  7612,  7684,
            7689,  7691,  7692,  7737,  7738,  7741,  7742,  7842,  7884,
            7886,  7888,  7889,  7957,  7958,  7959,  8011,  8044,  8050,
            8126,  8128,  8129,  8181,  8182,  8184,  8185,  8186,  8187,
            8197,  8288,  8334,  8335,  8338,  8339,  8341,  8413,  8419,
            8421,  8463,  8538,  8540,  8541,  8542,  8543,  8617,  8618,
            8647,  8681,  8790,  8852,  8853,  8854,  8864,  8876,  8979,
            9011,  9124,  9169,  9217,  9218,  9219,  9242,  9450,  9475,
            9476,  9485,  9486,  9487,  9488,  9489,  9490,  9491,  9501,
            9502,  9535,  9681,  9701,  9702,  9703,  9704,  9743,  9745,
            9795,  9843,  9844,  9880,  9945,  9946,  9959, 10101, 10459,
        10532, 10534, 10737, 10808, 10809, 11004, 11005, 11010, 11084,
        11133, 11141, 11176, 11197, 11201, 11383, 11386, 11566, 11620,
        11631, 11677, 11679, 11686, 11773, 11886, 11889, 11935, 11942,
        12091, 12092, 12093, 12106, 12199, 12201, 12364, 12460, 12461,
        12462, 12743, 12788, 12958, 12987, 12988, 12990, 13001, 13002,
        13003, 13032, 13065, 13066, 13076, 13078, 13079, 13096, 13194,
        13244, 13253, 13344, 13413, 13548, 13643])

    print('\nCandidates only found by peak flux S/N:')
    for id in peak_ids:
        if id not in fluence_ids:
            this_group = data['group_idx'].data == id
            print('Group ID:', id)
            print('feh', end='')
            for idx in np.nonzero(this_group)[0]:
                print(f" {data['obs_id'].data[idx].decode()}_candidate_{data['cand_id'].data[idx]}.png", end='')
            print('')

    print('\nPulsar candidates:')
    for i in np.nonzero(is_pulsar)[0]:
        id = group_ids[i]
        this_group = data['group_idx'].data == id
        print('Group ID:', id, '    Group size:', group_sizes[i])
        print('feh', end='')
        for idx in np.nonzero(this_group)[0]:
            print(f" {data['obs_id'].data[idx].decode()}_candidate_{data['cand_id'].data[idx]}.png", end='')
        print('')



    plt.show()
    # exit()

    data = data[group_selection]

    if False:
        bruh = np.ones(len(data), dtype=bool)
        stem = '/media/septagonic/CORSAIR'
        src = stem+'/candidates'
        # dest = stem+'/groups_sort_named'
        dest = stem+'/groups_selected_final'
        i = 0
        for row in data:
            for ext in ['png', 'gif']:
                fname_src = f"{row['obs_id']}_candidate_{row['cand_id']}.{ext}"
                fname_dest = f"{row['group_len']:03}_{row['group_idx']:05}_{row['obs_id']}_tcg_spike_rms_{row['cand_id']:03}.{ext}"
                os.system(f"cp {src}/{fname_src} {dest}/{fname_dest}")
                if not os.path.isfile(f'{src}/{fname_src}'):
                    bruh[i] = False
            i += 1
            # if row['cand_id'] == 52:
            #     print(row['group_len'], fname)

        data = data[bruh]

    name_changes = {'peak_flux'    : 'can_peak_flux'   ,
            'beam'         : 'can_beam'        ,
            'det_stat'     : 'can_det_stat'    ,
            'nks_flux_rat' : 'can_nks_flux_rat'}
    for key in name_changes:
        data.rename_column(key, name_changes[key])

    data.write(out_fname, format='fits', overwrite=True)

if __name__=='__main__':
    Fixup('/media/septagonic/CORSAIR/candidates/try_1_islands_selected.fits', '/media/septagonic/CORSAIR/group_islands_selected_final.fits', plot=True)