from astropy.table import Table
from astropy.table import Column
import numpy as np
from scipy import ndimage
from astropy import units as u
from astropy.wcs.utils import pixel_to_skycoord
from gleam_x.bin.beam_value_at_radec import beam_value, parse_metafits
from astropy import units as u
import lowner_john_ellipse as ellipse
import matplotlib.pyplot as plt

def FindIslands(obs, filters, get_beam=True, run_name=None):
    # detecting islands
    valid = np.zeros(obs.shape[1:], dtype=bool)
    for flr in filters:
        valid |= flr.data > flr.cut_low
    labels, n = ndimage.label(valid, structure=np.ones((3,3)))
    slices = ndimage.find_objects(labels)

    # Building table
    table = Table([
        Column(data=np.zeros( n         , dtype=np.dtype('<S20')), name='obs_id'         , unit=None    ),
        Column(data=np.zeros( n         , dtype=np.dtype('<S20')), name='filter_id'      , unit=None    ),
        Column(data=np.zeros( n               , dtype=np.int64  ), name='cand_id'        , unit=None    ),
        Column(data=np.zeros( n               , dtype=np.int64  ), name='x_pix'          , unit=u.pix   ),
        Column(data=np.zeros( n               , dtype=np.int64  ), name='y_pix'          , unit=u.pix   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='cent_x_pix'     , unit=u.pix   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='cent_y_pix'     , unit=u.pix   ),
        Column(data=np.zeros( n               , dtype=np.int64  ), name='box_w_pix'      , unit=u.pix   ),
        Column(data=np.zeros( n               , dtype=np.int64  ), name='box_h_pix'      , unit=u.pix   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='min_rad_pix'    , unit=u.pix   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='maj_rad_pix'    , unit=u.pix   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='min_rad_deg'    , unit=u.deg   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='maj_rad_deg'    , unit=u.deg   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='rot_deg'        , unit=u.deg   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='ra_deg'         , unit=u.deg   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='dec_deg'        , unit=u.deg   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='cent_ra_deg'    , unit=u.deg   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='cent_dec_deg'   , unit=u.deg   ),
        Column(data=np.zeros( n               , dtype=np.float64), name='cent_sep_deg'   , unit=u.deg   ),
        Column(data=np.zeros( n               , dtype=np.int64  ), name='area_pix'       , unit=u.pix**2),
        Column(data=np.zeros( n               , dtype=np.float64), name='area_deg'       , unit=u.deg**2),
        Column(data=np.zeros( n               , dtype=np.float64), name='peak_flux'      , unit=u.Jy    ),
        Column(data=np.zeros( n               , dtype=np.int64  ), name='peak_frame'     , unit=None    ),
        Column(data=np.ones(  n               , dtype=np.float64), name='beam'           , unit=None    ),
        Column(data=np.ones(  n               , dtype=np.float64), name='beam_norm'      , unit=None    ),
        Column(data=np.ones(  n               , dtype=np.float64), name='obs_cent_freq'  , unit=u.Hz    ),
        Column(data=np.ones(  n               , dtype=np.float64), name='det_stat'       , unit=u.Hz    ),
        Column(data=np.zeros((n, obs.shape[0]), dtype=np.float64), name='curve'          , unit=u.Jy    )] +
       [Column(data=np.zeros( n               , dtype=np.float64), name=flr.name         , unit=None    ) for flr in filters] +
       [Column(data=np.zeros( n               , dtype=np.float64), name=flr.name+'_norm' , unit=None    ) for flr in filters])

    table['obs_id'][:] = str(obs.obsid)
    table['obs_cent_freq'][:] = obs.freq
    if run_name is None:
        table['filter_id'][:] = '_'.join([flr.name for flr in filters])
    else:
        table['filter_id'][:] = run_name

    # getting island attributes
    for i in range(n):
        table['cand_id'][i] = i + 1
        # Grabbing island region box
        max_island = np.max(obs.data[:, slices[i][0], slices[i][1]], axis=0)
        mask = labels[slices[i][0], slices[i][1]] == table['cand_id'][i]
        max_island[~mask] = np.nan
        # Island area
        table['area_pix'][i] = np.count_nonzero(mask)
        # x and y pixel coordinates of max filter value relative to island bounding box
        y_sub, x_sub = np.unravel_index(np.nanargmax(max_island), max_island.shape)
        table['x_pix'][i] = slices[i][1].start + x_sub
        table['y_pix'][i] = slices[i][0].start + y_sub
        table['box_w_pix'][i] = slices[i][1].stop - slices[i][1].start
        table['box_h_pix'][i] = slices[i][0].stop - slices[i][0].start
        table['peak_flux'][i] = max_island[y_sub, x_sub]
        table['peak_frame'][i] = np.argmax(obs.data[:, table['y_pix'][i], table['x_pix'][i]])
        # Fitting an ellipse to the island (warning: jank)
        new_mask = np.zeros((mask.shape[0]+1, mask.shape[1]+1), dtype=np.int64)
        int_mask = mask.astype(np.int64)
        new_mask[ :-1, :-1] += int_mask
        new_mask[1:  , :-1] += int_mask
        new_mask[ :-1,1:  ] += int_mask
        new_mask[1:  ,1:  ] += int_mask
        new_mask = (new_mask > 0) & (new_mask <= 2) # This is true only at boundary pixels
        x, y = np.meshgrid(np.arange(slices[i][1].start, slices[i][1].stop+1),
                            np.arange(slices[i][0].start, slices[i][0].stop+1))
        x = x[new_mask].astype(np.float64) - 0.5 # These are the boundary coordinates
        y = y[new_mask].astype(np.float64) - 0.5
        x += (np.random.rand(*x.shape)*2-1)*0.01 # A bit of wobble prevents numerical problems
        y += (np.random.rand(*x.shape)*2-1)*0.01
        try: # Fit the ellipse
            center, major, minor, angle = ellipse.welzl(np.stack([x, y], axis=1))
        except: # If fitting is too hard, do a very rough circle fit instead
            center = (np.mean(x), np.mean(y))
            major = minor = np.max(np.sqrt((x-center[0])**2 + (y-center[1])**2))
            angle = 0.0
        # plt.imshow(np.flip(mask, axis=0), cmap='gray', extent=(slices[i][1].start-0.5, slices[i][1].stop-0.5, slices[i][0].start-0.5, slices[i][0].stop-0.5))
        # plt.scatter(x, y)
        # ellipse.plot_ellipse((center, major, minor, angle))
        # print(center, major, minor, angle, major / minor)
        # plt.show()
        table['cent_x_pix'][i] = center[0]
        table['cent_y_pix'][i] = center[1]
        table['min_rad_pix'][i] = minor
        table['maj_rad_pix'][i] = major
        table['rot_deg'][i] = np.degrees(angle)
        # Peak coordinates in degrees
        skycoord = pixel_to_skycoord(table['x_pix'][i], table['y_pix'][i], obs.wcs)
        table['ra_deg'][i] = skycoord.ra.degree
        table['dec_deg'][i] = skycoord.dec.degree
        if np.isnan(table['ra_deg'][i]) or np.isnan(table['dec_deg'][i]):
            continue
        table['cent_sep_deg'][i] = obs.cent.separation(skycoord).degree
        # Converting island dimentions to degrees
        testX = pixel_to_skycoord(table['cent_x_pix'][i] + 1, table['cent_y_pix'][i], obs.wcs)
        testY = pixel_to_skycoord(table['cent_x_pix'][i], table['cent_y_pix'][i] + 1, obs.wcs)
        pix2deg = (skycoord.separation(testX).degree + skycoord.separation(testY).degree) / 2
        table['min_rad_deg'][i] = table['min_rad_pix'][i] * pix2deg
        table['maj_rad_deg'][i] = table['maj_rad_pix'][i] * pix2deg
        table['area_deg'][i] = table['area_pix'][i] * pix2deg**2
        # Center coordinates in degrees
        skycoord = pixel_to_skycoord(table['cent_x_pix'][i], table['cent_y_pix'][i], obs.wcs)
        table['cent_ra_deg'][i] = skycoord.ra.degree
        table['cent_dec_deg'][i] = skycoord.dec.degree
        if np.isnan(table['ra_deg'][i]) or np.isnan(table['dec_deg'][i]):
            continue
        # Light curve at max pixel
        table['curve'][i] = obs.data[:, table['y_pix'][i], table['x_pix'][i]]
        # Filter values at pixel coordinates
        for flr in filters:
            table[flr.name][i] = np.max(flr.data[slices[i][0], slices[i][1]][mask])
            table[flr.name+'_norm'][i] = table[flr.name][i] / flr.cut_high
            table['det_stat'][i] += table[flr.name+'_norm'][i]**2
        table['det_stat'][i] = np.sqrt(table['det_stat'][i])

    table = table[~(np.isnan(table['ra_deg']) | np.isnan(table['dec_deg']))]

    # Getting the beam values
    if get_beam:
        cent_beam = GetBeamAtCoords(obs.obsid, obs.cent.ra.deg, obs.cent.dec.deg)
        table['beam'] = GetBeamAtCoords(obs.obsid, table['ra_deg'].data, table['dec_deg'].data)
        table['beam_norm'] = table['beam'] / cent_beam

    return table, labels, slices

def GetBeamAtCoords(obsid, ra_deg, dec_deg):
    try:
        url = "http://ws.mwatelescope.org/metadata/fits?obs_id=" + str(obsid)
        t, delays, freq, gridnum = parse_metafits(url)
        beam_x, beam_y = beam_value(ra_deg, dec_deg, t, delays, freq, gridnum)
        vals = (beam_x + beam_y) / 2
        return vals
    except Exception as e:
        print(e)
        print('Failed to get beam values! Setting them to 1.')
        return np.ones(ra_deg.shape, dtype=np.float64)
