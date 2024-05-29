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

fname_list = [f'{sys.argv[1]}/{fname}/{fname}_{sys.argv[2]}' for fname in os.listdir(sys.argv[1])]

invalid_bool_names = ['invalid_beam', 'invalid_majmin', 'scintil_dist', 'scintil_corr', 'close_to_ateam', 'close_to_bright', 'is_moon', 'is_jupiter']
filter_bool_names = ['valid_spike', 'valid_tcg', 'valid_rms']

table_list = []
count = 0
for fname in fname_list:
    count += 1
    try:
        data = Table.read(fname, format='fits')
        data = Table.read(fname, format='fits')
        # data['valid_tcg'] = data['tcg_norm'] > 0.8
        # data['valid_rms'] = data['rms_norm'] > 0.8
        # data['invalid_beam'] = data['beam_norm'] < 0.25
        # data = data[np.logical_or.reduce([data[name].value for name in filter_bool_names])]
        # data = data[~np.logical_or.reduce([data[name].value for name in invalid_bool_names])]
        print(f'{fname} read', count, '/', len(fname_list))
        table_list.append(data)
    except:
        print(f'{fname} not found')

data = vstack(table_list)
data.write(sys.argv[3], format='fits')