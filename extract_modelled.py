from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
import pandas as pd
import numpy as np

obslist = pd.read_csv('100_obs.csv')
path = '/media/septagonic/CORSAIR/gxarchive/{0}/{0}_{1}'
obs_idx = 0
name = 'mod_scint'
mod_idx = np.array([0], dtype=np.int64)

for obsid in obslist.obsid:
    print('--------------', obsid, '------', obs_idx)
    obs_idx += 1

    try:
        for version in ['', '_selected']:
            isl_table = Table.read(path.format(obsid, name+'_islands'+version+'.fits'), format='fits')
            mod_table = Table.read(path.format(obsid, 'modtab.fits'), format='fits')

            isl_table = isl_table[~np.isnan(isl_table['ra_deg'])]
            mod_table = mod_table[~np.isnan(mod_table['ra_deg'])]
            mod_idx = np.arange(len(mod_table), dtype=np.int64) + mod_idx[-1] + 1

            isl_coord = SkyCoord(ra=isl_table['ra_deg'], dec=isl_table['dec_deg'], unit=u.deg, frame='fk5')
            mod_coord = SkyCoord(ra=mod_table['ra_deg'], dec=mod_table['dec_deg'], unit=u.deg, frame='fk5')

            idx, sep, _ = isl_coord.match_to_catalog_sky(mod_coord)
            isl_is_mod = sep < 4*u.arcmin

            found_table = isl_table[isl_is_mod]
            for col in ['flux', 'dur']:
                found_table.add_column(mod_table[col][idx[isl_is_mod]], name='mod_'+col)
            found_table.add_column(mod_idx[idx[isl_is_mod]], name='mod_idx')

            mod_found = np.ones(len(mod_table), dtype=bool)
            mod_found[idx[isl_is_mod]] = True

            not_found_table = mod_table[~mod_found]

            found_table.write(path.format(obsid, name+'_islands'+version+'_true.fits'), format='fits', overwrite=True)
            not_found_table.write(path.format(obsid, name+'_islands'+version+'_false.fits'), format='fits', overwrite=True)
    except Exception as e:
        print(e)
    