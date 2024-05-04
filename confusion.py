import pandas as pd
import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import matplotlib.colors as col

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

def ConfusionColours(confusion, classes, axs):
    axs.matshow(confusion**0.3, cmap=plt.cm.Blues)
    axs.set_yticks(np.arange(len(classes)), labels=classes, fontsize=16)
    axs.set_xticks(np.arange(len(classes))-0.5, labels=classes, fontsize=16, rotation=45, horizontalalignment='left')
    axs.tick_params(axis=u'both', which=u'both', length=0)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            axs.text(j, i, str(confusion[i,j]), va='center', ha='center', fontsize=11)

def CalcConfusion(data, classes):
    selection_bools = data[selection_classes]
    confusion = np.zeros([len(selection_classes)]*2, dtype=np.int64)
    for row in selection_bools:
        row = np.array([row[name] for name in classes], dtype=bool)
        inds, = np.nonzero(row)
        for idx in inds:
            confusion[idx, inds] += 1
    return confusion

def ReadTables(obslist, path):
    table_list = []
    for _, row in obslist.iterrows():
        table_list.append(Table.read(path.format(int(row['obsid'])), format='fits'))
        table_list[-1]['valid_spike'] = table_list[-1]['spike'] > 8
        # table_list[-1]['valid_tcg'] = table_list[-1]['tcg'] > row.rms * 8
        # table_list[-1]['valid_rms'] = table_list[-1]['rms'] > row.rms * 2.5
    return vstack(table_list), table_list

if __name__=='__main__':
    # obslist = pd.read_csv('/media/septagonic/CORSAIR/gxarchive/obs_data.csv')
    obslist = pd.read_csv('100_obs.csv')

    data, table_list = ReadTables(obslist, '/media/septagonic/CORSAIR/gxarchive/{0}/{0}_4_islands.fits')
    filter_bool_names = ['valid_tcg', 'valid_spike', 'valid_rms']
    # filter_bool_names = ['valid_tcg0.1', 'valid_tcg1', 'valid_tcg5']
    data = data[np.logical_or.reduce([data[name].value for name in filter_bool_names])]
    selection_classes = filter_bool_names + ['invalid_area', 'invalid_beam', 'invalid_majmin', 'scintil_dist', 'scintil_corr', 'close_to_ateam', 'close_to_bright']
    fig = plt.figure(figsize=(9, 8))
    axs = fig.add_axes(rect=(0.2, 0.05, 0.75, 0.75))
    confusion = CalcConfusion(data, selection_classes)
    ConfusionColours(confusion, selection_classes, axs)
    axs.axvline(2.5, c='black')
    axs.axhline(2.5, c='black')

    data, table_list = ReadTables(obslist, '/media/septagonic/CORSAIR/gxarchive/{0}/{0}_4_islands_selected.fits')
    data = data[np.logical_or.reduce([data[name] for name in filter_bool_names])]
    selection_classes = filter_bool_names
    fig = plt.figure(figsize=(4, 4))
    axs = fig.add_axes(rect=(0.35, 0.005, 0.55, 0.8))
    confusion = CalcConfusion(data, selection_classes)
    ConfusionColours(confusion, selection_classes, axs)
    axs.axvline(2.5, c='black')
    axs.axhline(2.5, c='black')

    plt.figure()
    freq_list = np.unique(obslist['freq'])
    ncands = np.array([len(x) for x in table_list])
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
