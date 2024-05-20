import pandas as pd
import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.patches as patches
import mplcursors

############################## Print Confusion Matrix ##############################

def PrintConfusionMatrix(confusion, classes):
    pd.options.display.width = 100000000
    confusion_cols = list(classes) + ['Total']
    for axis in (0, 1):
        confusion = np.concatenate([confusion, confusion.sum(axis=axis, keepdims=True)], axis=axis)
    print(pd.DataFrame(confusion, confusion_cols, confusion_cols))

####################### Print LaTeX Formatted Confusion Matrix ######################

def ConfusionLatex(confusion, classes):
    print('\\hline')
    print('Index & Category &', ' & '.join(map(str, range(len(classes)))), '& Total \\\\')
    print('\\hline')
    for i in range(len(classes)):
        print(str(i), '&', classes[i], end='')
        if i > 0:
            print('&', ' & '.join(map(str, confusion[i, 0:i])), end='')
        print(f'& \\textbf{{{confusion[i, i]}}} ', end='')
        if i < len(classes)-1:
            print('&', ' & '.join(map(str, confusion[i, i+1:])), end='')
        print('&', np.sum(confusion[i]), '\\\\')
    print('\\hline')
    print('& Total &', ' & '.join(map(str, np.sum(confusion, axis=0))), '&', np.sum(confusion), '\\\\')
    print('\\hline')

def ConfusionColours(confusion, classes, axs, cmap=plt.cm.Blues, diagonal=False):
    confusion_norm = confusion.astype(np.float64)
    confusion_norm -= np.min(confusion)
    confusion_norm /= np.max(confusion_norm)
    confusion_norm **= 0.3
    axs.matshow(confusion_norm, cmap=cmap, aspect='auto')
    axs.set_yticks(np.arange(len(classes)), labels=classes, fontsize=16)
    axs.set_xticks(np.arange(len(classes))-0.5, labels=classes, fontsize=16, rotation=45, horizontalalignment='left')
    axs.tick_params(axis=u'both', which=u'both', length=0)
    axs.set_xlim([-.5, confusion.shape[1]-.5])
    axs.set_ylim([confusion.shape[0]-.5, -.5])
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            color = 'black' if confusion_norm[i,j] < 0.75 else 'white'
            axs.text(j, i, str(confusion[i,j]), va='center', ha='center', fontsize=11, color=color)
    if diagonal:
        for i in range(confusion.shape[0]):
            rect = patches.Rectangle((i-0.5, i-0.5), 0.99, 0.99, linewidth=1.5, edgecolor='black', facecolor='none')
            axs.add_patch(rect)

def CalcConfusion(data, classes):
    selection_bools = data[classes]
    confusion = np.zeros([len(classes)]*2, dtype=np.int64)
    for row in selection_bools:
        row = np.array([row[name] for name in classes], dtype=bool)
        inds, = np.nonzero(row)
        for idx in inds:
            confusion[idx, inds] += 1
    return confusion

def CalcExclusive(data, *classes_nested):
    matrix_list = []
    for classes in classes_nested:
        selection_bools = np.array([data[x].data for x in classes], dtype=bool).transpose()
        valid_rows = selection_bools[np.sum(selection_bools, axis=1) == 1]
        exclusive = np.zeros((len(classes), 1), dtype=np.int64)
        for i in range(len(classes)):
            exclusive[i] = np.count_nonzero(valid_rows[:,i])
        matrix_list.append(exclusive)
    return np.concatenate(matrix_list, axis=0)

def CalcPerCategory(data, classes, catcol):
    categories = np.unique(data[catcol])
    matrix = np.zeros((len(classes), len(categories)), dtype=np.int64)
    for i in range(len(data)):
        catidx = np.where(categories == data[catcol][i])
        for j in range(len(classes)):
            if classes[j] is None:
                matrix[j, catidx] += 1
            else:
                matrix[j, catidx] += int(data[classes[j]][i])
    return matrix, categories


def ReadTables(obslist, path):
    table_list = []
    for _, row in obslist.iterrows():
        table_list.append(Table.read(path.format(int(row['obsid'])), format='fits'))
        if 'obs_cent_freq' not in table_list[-1].colnames:
            table_list[-1].add_column(np.full(len(table_list[-1]), row['freq']), name='obs_cent_freq')
    return vstack(table_list), table_list

if __name__=='__main__':
    # obslist = pd.read_csv('/media/septagonic/CORSAIR/gxarchive/obs_data.csv')
    obslist = pd.read_csv('100_obs.csv')
    # run_name = 'mod_found'
    prefix = 'mod_'
    suffix = '_true'
    # prefix = 'real_'
    # suffix = ''
    cand_name = 'recovered modelled transients (of 100)'
    # cand_name = 'candidates'

    # ALL ISLANDS
    data, table_list = ReadTables(obslist, '/media/septagonic/CORSAIR/gxarchive/{0}/{0}_'+prefix+'islands'+suffix+'.fits')
    filter_bool_names = ['valid_tcg', 'valid_spike', 'valid_rms']
    invalid_bool_names = ['invalid_beam', 'invalid_majmin', 'scintil_dist', 'scintil_corr', 'close_to_ateam', 'close_to_bright', 'is_moon']
    data = data[np.logical_or.reduce([data[name].value for name in filter_bool_names])]
    data['invalid_majmin'] &= data['area_pix'] > 3
    selection_classes = filter_bool_names + invalid_bool_names
    fig = plt.figure(figsize=(11, 6.5))
    # Confusion
    axs = fig.add_axes(rect=(0.2, 0.11, 0.425, 0.675))
    confusion = CalcConfusion(data, selection_classes)
    ConfusionColours(confusion, selection_classes, axs, diagonal=True)
    axs.axvline(2.5, c='black')
    axs.axhline(2.5, c='black')
    # Frequencies
    axs = fig.add_axes(rect=(0.64, 0.11, 0.2, 0.675))
    percategory, categories = CalcPerCategory(data, selection_classes, 'obs_cent_freq')
    categories = [f'{int(x/1e6)} MHz' for x in categories]
    ConfusionColours(percategory, categories, axs, plt.cm.Greens)
    axs.set_yticklabels([])
    axs.axhline(2.5, c='black')
    # Overall
    axs = fig.add_axes(rect=(0.64, 0.02, 0.2, 0.0675))
    percategory, categories = CalcPerCategory(data, [None], 'obs_cent_freq')
    categories = [f'{int(x/1e6)} MHz' for x in categories]
    ConfusionColours(percategory, categories, axs, plt.cm.Greens)
    axs.set_yticklabels(['Overall'] + ['']*4)
    axs.set_xticklabels([])
    # Exclusive
    axs = fig.add_axes(rect=(0.855, 0.11, 0.04, 0.675))
    exclusive = CalcExclusive(data, filter_bool_names, invalid_bool_names)
    ConfusionColours(exclusive, ['exclusive'], axs, plt.cm.Reds)
    axs.set_yticklabels([])
    axs.axhline(2.5, c='black')
    plt.tight_layout()
    plt.savefig('paper_plots/'+prefix+'per_type_mat.pdf')

    # SELECTED ISLANDS
    # data, table_list = ReadTables(obslist, '/media/septagonic/CORSAIR/gxarchive/{0}/{0}_'+prefix+'islands_selected'+suffix+'.fits')
    data = data[~np.logical_or.reduce([data[name].value for name in invalid_bool_names])]
    selection_classes = filter_bool_names
    fig = plt.figure(figsize=(7, 3.5))
    # Confusion
    axs = fig.add_axes(rect=(0.2, 0.25, 0.2, 0.39))
    confusion = CalcConfusion(data, selection_classes)
    ConfusionColours(confusion, selection_classes, axs, diagonal=False)
    # Frequencies
    axs = fig.add_axes(rect=(0.425, 0.25, 0.4, 0.39))
    percategory, categories = CalcPerCategory(data, selection_classes, 'obs_cent_freq')
    categories = [f'{int(x/1e6)} MHz' for x in categories]
    ConfusionColours(percategory, categories, axs, plt.cm.Greens)
    axs.set_yticklabels([])
    # Overall
    axs = fig.add_axes(rect=(0.425, 0.08, 0.4, 0.13))
    percategory, categories = CalcPerCategory(data, [None], 'obs_cent_freq')
    categories = [f'{int(x/1e6)} MHz' for x in categories]
    ConfusionColours(percategory, categories, axs, plt.cm.Greens)
    axs.set_yticklabels(['Overall'] + ['']*4)
    axs.set_xticklabels([])
    # Exclusive
    axs = fig.add_axes(rect=(0.85, 0.25, 0.062, 0.39))
    exclusive = CalcExclusive(data, filter_bool_names)
    ConfusionColours(exclusive, ['exclusive'], axs, plt.cm.Reds)
    axs.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('paper_plots/'+prefix+'sel_per_type_mat.pdf')
    
    plt.figure()
    freq_list = np.unique(obslist['freq'])
    ncands = np.array([len(x) for x in table_list])
    bins = np.arange(0, np.max(ncands)+2, 10)
    hist, _ = np.histogram(ncands, bins=bins, density=True)
    plt.stairs(hist, edges=bins, label=f'Overall', alpha=0.5, fill=True, lw=2)
    norm = col.Normalize()
    colors = plt.cm.jet(norm(freq_list))
    for i in range(len(freq_list)):
        hist, _ = np.histogram(ncands[obslist['freq'] == freq_list[i]], bins=bins, density=True)
        plt.stairs(hist, edges=bins, label=f'{int(freq_list[i]/1e6)} MHz', lw=1, color=colors[i])
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Number of '+cand_name+' per observation')
    plt.ylabel('Fraction of observations at frequency')
    plt.yticks(ticks=[0.01, 0.1, 1], labels=['1%', '10%', '100%'])
    plt.tight_layout()
    plt.savefig('paper_plots/'+prefix+'sel_per_obs_hist.pdf')

    plt.figure(figsize=(12, 4))
    obslist = obslist.iloc[np.argsort(obslist['freq'])]
    freq_sort = np.argsort(obslist['freq'].values.astype(np.int64))
    freq_unique = np.unique(obslist['freq'].values.astype(np.int64))
    freqs = obslist['freq'].values[freq_sort].astype(np.int64)
    obsids = obslist['obsid'].values[freq_sort]
    counts = np.array([np.count_nonzero(data['obs_id'].astype(np.int64) == obsid) for obsid in obsids])
    print(counts)
    x = np.arange(len(counts))
    for freq in freq_unique:
        subset = freqs == freq
        plt.bar(x[subset], counts[subset], label=f'{int(freq/1e6)} MHz')
        plt.hlines(np.mean(counts[subset]), np.min(x[subset]), np.max(x[subset]), colors='k', linestyles='--')
        # if freq < np.max(freq_unique):
        #     plt.axvline(np.max(x[subset])+0.5, linestyle=':', color='k')
    plt.xticks(np.arange(len(counts)), obsids, rotation=90, fontsize='small')
    plt.xlabel('obsid')
    plt.xlim([-.75, x[-1]+.75])
    plt.ylabel('Numbr of '+cand_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig('paper_plots/'+prefix+'sel_per_obs_bar.pdf')

    if True: # For modelled data
        mod_data, mod_table_list = ReadTables(obslist, '/media/septagonic/CORSAIR/gxarchive/{0}/{0}_modtab.fits')
        # Peak flux histogram
        plt.figure(figsize=(5.5, 3.5))
        plt.subplot(1, 2, 1)
        freq_list = np.unique(obslist['freq'])
        hist, bins = np.histogram(data['mod_flux'], bins=20)
        hist_all, _ = np.histogram(mod_data['flux'], bins=bins)
        plt.stairs(hist/hist_all, edges=bins, label=f'Overall', alpha=0.5, fill=True, lw=2)
        # norm = col.Normalize()
        # colors = plt.cm.jet(norm(freq_list))
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i in range(len(freq_list)):
            hist, _ = np.histogram(data['mod_flux'][data['obs_cent_freq'] == freq_list[i]], bins=bins)
            hist_all, _ = np.histogram(mod_data['flux'][mod_data['obs_cent_freq'] == freq_list[i]], bins=bins)
            plt.stairs(hist/hist_all, edges=bins, label=f'{int(freq_list[i]/1e6)} MHz', lw=1, color=colors[i])
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Peak flux (Jy)')
        plt.ylabel('Fraction of modelled transients recovered')
        plt.yticks(ticks=[0.01, 0.1, 1], labels=['1%', '10%', '100%'])
        # Pulse width histoigram (split by peak flux)
        '''
        plt.subplot(1, 2, 2)
        flux_bins = np.array([0, 1, 2, 3])
        hist, bins = np.histogram(data['mod_dur'], density=True, bins=20)
        plt.stairs(hist, edges=bins, label=f'Overall', alpha=0.5, fill=True, lw=2)
        norm = col.Normalize()
        colors = plt.cm.jet(norm(flux_bins))
        for i in range(len(flux_bins)-1):
            hist, _ = np.histogram(data['mod_dur'][(flux_bins[i] < data['mod_flux']) & (data['mod_flux'] <= flux_bins[i+1])], bins=bins, density=True)
            plt.stairs(hist, edges=bins, label=f'{flux_bins[i]} < peak flux < {flux_bins[i+1]}', lw=1, color=colors[i])
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Modelled transient pulse width')
        plt.ylabel('Fraction of modelled transients recovered')
        plt.yticks(ticks=[0.01, 0.1, 1], labels=['1%', '10%', '100%'])
        plt.tight_layout()
        '''
        # Peak flux histogram (split by filter)
        plt.subplot(1, 2, 2)
        # data = data[data['mod_flux'] < 1]
        # mod_data = mod_data[mod_data['flux'] < 1]
        filters = np.array(['valid_tcg', 'valid_spike', 'valid_rms'])
        hist, bins = np.histogram(data['mod_dur'], bins=20)
        hist_all, _ = np.histogram(mod_data['dur'], bins=bins)
        plt.stairs(hist/hist_all, edges=bins, label=f'Overall', alpha=0.5, fill=True, lw=2)
        for i in range(len(filters)):
            hist, _ = np.histogram(data['mod_dur'][data[filters[i]]], bins=bins)
            plt.stairs(hist/hist_all, edges=bins, label=filters[i], lw=1)
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Pulse width (frames)')
        # plt.ylabel('Fraction of modelled transients recovered')
        plt.yticks([])
        plt.tight_layout()
        plt.savefig('paper_plots/'+prefix+'sel_det_hist.pdf')

    plt.show()

def InvalidArea():
    # INVESTIGATING INVALIUD AREA
    valid_area = data[~data['invalid_area']]
    invalid_area = data[data['invalid_area']]
    is_spike = data[data['valid_spike']]
    not_spike = data[~data['valid_spike']]

    for yaxis in ['peak_flux', 'spike']:
        plt.figure()
        plt.hist(valid_area['peak_flux'], bins=20, alpha=0.5, density=True, label='valid_area')
        plt.hist(invalid_area['peak_flux'], bins=20, alpha=0.5, density=True, label='invalid_area')
        plt.xlabel('peak_flux')
        plt.ylabel('Density of candidates')
        plt.legend()

        plt.figure()
        plt.hist(data['spike'], bins=20, alpha=0.5, density=True)
        plt.xlabel('spike')
        plt.ylabel('Density of candidates')

        plt.figure()
        plt.scatter(not_spike['area_pix'], not_spike[yaxis], c=not_spike['beam'], cmap=plt.cm.Greens, vmin=0, vmax=1, label='other')
        sc = plt.scatter(is_spike['area_pix'], is_spike[yaxis], c=is_spike['beam'], cmap=plt.cm.Blues, vmin=0, vmax=1, label='spike')
        plt.colorbar(label='primary beam')
        plt.xlabel('area_pix')
        plt.ylabel(yaxis)
        plt.xscale('log')
        plt.axvline(2.5, color='red', label='cutoff')
        plt.legend()

        cursor1 = mplcursors.cursor(sc, hover=False)

        @cursor1.connect("add")
        def on_add(sel):
            i = sel.index
            sel.annotation.set_text(is_spike[['obs_id', 'cand_id']][sel.index])

    plt.figure()
    freq_sort = np.argsort(obslist['freq'].values)
    freq_unique = np.unique(obslist['freq'].values)
    freqs = obslist['freq'].values[freq_sort]
    obsids = obslist['obsid'].values[freq_sort]
    counts = np.array([np.count_nonzero(data['obs_id'] == obsid) for obsid in obsids])
    x = np.arange(len(counts))
    for freq in freq_unique:
        subset = freqs == freq
        plt.bar(x[subset], counts[subset], label=f'{int(freq/1e6)} MHz')
        if freq < np.max(freq_unique):
            plt.axvline(np.max(x[subset])+0.5, linestyle=':', color='k')
    plt.xticks(np.arange(len(counts)), obsids, rotation=90)
    plt.xlabel('obsid')
    plt.ylabel('Numbr of candidates')
    plt.legend()

    plt.show()
