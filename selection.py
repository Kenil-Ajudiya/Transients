import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.table import Column
import os
from astropy.time import Time
from astropy.coordinates import get_body, EarthLocation

def SelectSources(obs, table, labels, filters):
    loose = False

    min_beam = 0.5              # Minimum beam value normalised by the central beam values
    min_radius = 4*u.arcmin     # Minimum radius to classify as scintillation
    radius_coef = 2             # max(table['maj_rad_deg'] * radius_coef, min_radius) is used to classify scintillation
    min_area = 3                # Minimum area of an island in pixels (spike sometimes finds single pixel spikes)
    min_corr = 0.8              # Minimum correlation coefficient to classify light curves as identical
    min_rad_ateam = 5.0*u.deg   # Radius around A-team sources to exclude candidates
    min_rad_bright = 1.0*u.deg  # Radius around bright (flux > min_flux_bright) sources to exclude candidates
    min_flux_bright = 10*u.Jy   # Minmum flux to classify source as bright
    max_majmin = 2              # maximum allowed ratio between the major and minor radii
    flux_ratio = 1.1            # minimum ratio between candidate peak flux and nearest known source flux if within min_radius
    if obs.freq < 175e6:
        flux_ratio = 1.25
    if obs.freq < 150e6:
        flux_ratio = 1.5
    if obs.freq < 100e6:
        flux_ratio = 2.0
    max_count = 100              # Maximum number of candidates allowed
    cut_scales = [1.0, 1.25, 1.5]

    if loose:
        min_beam = 0.25
        max_count = 100
        cut_scales = [0.8, 0.8, 0.8]

    cat_cands = SkyCoord(table['ra_deg'], table['dec_deg'], unit=(u.deg, u.deg), frame="fk5")

    # Is my normalised beam value too small?
    invalid_beam = table['beam_norm'] < min_beam

    # Detecting scintillation artefacts
    min_rad_close = np.where(table['maj_rad_deg'] < min_radius, min_radius, table['maj_rad_deg'])
    min_rad_far = min_rad_close * radius_coef
    scintil_dist = np.zeros(len(table), dtype=bool) # Am I too close to a nearby known source with greater flux?
    scintil_corr = np.zeros(len(table), dtype=bool) # Am I too correlated with a nearby known source with greater flux?
    for name in ['nks', 'nks2']:
        smaller_flux = table['peak_flux'] < table[name+'_flux'] * flux_ratio * table['beam']
        scintil_dist |= (table[name+'_sep_deg'] < min_rad_close) & smaller_flux
        scintil_corr |= ((table[name+'_sep_deg'] < min_rad_far) & (np.abs(table[name+'_corr']) > min_corr)) & smaller_flux

    # Reference catalogue of known sources
    ref_cat = fits.open(os.getenv('GGSM', "/home/septagonic/GLEAM-X-pipeline/models/GGSM.fits"))[1].data

    # Am I too close to super bright A-team source?
    cat_bright = SkyCoord(["23h23m24.000s", "19h59m28.35663s", "05h34m31.94s", "12h30m49.42338s", "05h19m49.7229s", "16h51m11.4s", "09h18m05.651s", "13h25m27.600s"], ["+58d48m54.00s", "+40d44m02.0970s", "+22d00m52.2s", "+12d23m28.0439s", "-45d46m43.853s", "+04d59m20s", "-12d05m43.99s", "-43d01m09s"])
    idx, sep, _ = match_coordinates_sky(cat_cands, cat_bright)
    close_to_ateam = sep < min_rad_ateam

    # Am I too close to bright source?
    cat_bright = SkyCoord(ref_cat['RAJ2000'][ref_cat['S_200']*u.Jy>min_flux_bright], ref_cat['DEJ2000'][ref_cat['S_200']*u.Jy>min_flux_bright], unit=(u.deg, u.deg), frame="fk5")
    idx, sep, _ = match_coordinates_sky(cat_cands, cat_bright)
    close_to_bright = sep < min_rad_bright

    # Detecting extended stripe islands
    invalid_majmin = (table['maj_rad_pix'] > (table['min_rad_pix'] * max_majmin)) & (table['area_pix'] > min_area)

    # Detecting the moon and jupiter
    time = Time(int(obs.obsid), format='gps')
    MWA = EarthLocation(lat=-26.70331940, lon=116.67081524)
    moon = get_body('moon', time, MWA)
    jupiter = get_body('jupiter', time, MWA)
    is_moon = moon.separation(cat_cands) < 0.5*u.deg
    is_jupiter= jupiter.separation(cat_cands) < min_radius

    # Adding removal justifications to candidate table
    table.add_columns([
    #     Column(data=invalid_area   , name='invalid_area'),
        Column(data=invalid_beam   , name='invalid_beam'),
        Column(data=invalid_majmin , name='invalid_majmin'),
        Column(data=scintil_dist   , name='scintil_dist'),
        Column(data=scintil_corr   , name='scintil_corr'),
        Column(data=close_to_ateam , name='close_to_ateam'),
        Column(data=close_to_bright, name='close_to_bright'),
        Column(data=is_moon        , name='is_moon'),
        Column(data=is_jupiter     , name='is_jupiter')
    ])

    # Determine whether the candidate's filter value is valid for each filter
    valid_any_filter = np.zeros(len(table), dtype=bool)
    for flr in filters:
        cutoff = flr.cut_high
        if flr.name != 'spike':
            if obs.freq < 150e6:
                if obs.freq < 100e6:
                    cutoff *= cut_scales[2]
                else:
                    cutoff *= cut_scales[1]
            else:
                cutoff *= cut_scales[0]
        valid_filter = table[flr.name] > cutoff
        valid_any_filter |= valid_filter
        table.add_column(Column(data=valid_filter, name='valid_'+flr.name))

    table.add_column(Column(data=valid_any_filter, name='valid_any'))

    # Oring everything together
    # invalid = invalid_area | invalid_beam | scintil_dist | scintil_corr | close_to_ateam | close_to_bright | ~valid_any_filter | invalid_majmin | is_moon | is_jupiter
    invalid = invalid_beam | scintil_dist | scintil_corr | close_to_ateam | close_to_bright | ~valid_any_filter | invalid_majmin | is_moon | is_jupiter

    too_many = np.zeros(len(table), dtype=bool)
    if np.count_nonzero(~invalid) > max_count:
        too_many[:] = True
    table.add_column(Column(data=too_many, name='too_many'))

    invalid |= too_many

    new_table = table[~invalid]

    # combined = np.zeros(len(new_table), dtype=bool)
    # for i in np.flip(np.argsort(new_table['peak_flux'])):
    #     for j in np.nonzero(~combined)[0]:
    #         if j > i:
    #             corr = np.sum(new_table['curve'][j] * new_table['curve'][i]) /\
    #                 np.sqrt(np.sum(new_table['curve'][j]**2) * np.sum(new_table['curve'][i]**2))
    #             if corr > min_corr:
    #                 combined[j] = True
    #                 labels[labels == new_table['cand_id'][j]] = labels[new_table['y_pix'][j], new_table['x_pix'][i]]

    # return new_table[~combined]

    return new_table
    
