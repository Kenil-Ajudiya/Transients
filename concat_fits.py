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

table_list = []
for fname in fname_list:
    try:
        data = Table.read(fname, format='fits')
        print(f'{fname} read')
        table_list.append(data)
    except:
        print(f'{fname} not found')

data = vstack(table_list)
data.write(sys.argv[3], format='fits')