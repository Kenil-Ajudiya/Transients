import FileIO as io
import island as isl
import known as knw
import crossmatch as cm
import selection as sel
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import diagnostic
import os
from filters import RemoveLines

class Observation:
    def __init__(self, path, obsid, obs_name):
        self.obsid = obsid
        self.data, self.header = io.ReadImage(path.format(self.obsid, obs_name))
        self.time  = Time(int(self.obsid), format='gps').utc.iso
        self.shape = self.data.shape
        self.wcs   = io.ObsWCS(self.header)
        self.freq  = self.header["FREQ"]
        self.tstep = self.header['TIMESTEP']
        self.cent  = self.wcs.pixel_to_world(self.header["NAXIS1"]/2, self.header["NAXIS2"]/2)
        self.cent  = SkyCoord(self.cent.ra.value, self.cent.dec.value, unit = (u.deg, u.deg), frame = "fk5")
        self.fov   = np.sqrt(2) * self.header["CDELT2"] * max(self.header["NAXIS1"], self.header["NAXIS2"]) / 2
        self.rms   = np.std(self.data)
        self.mean  = np.mean(self.data)
        self.ncands = None

class Filter:
    def __init__(self, name, cut_low, cut_high, scale_rms, func, *args):
        self.name = name
        self.data = None
        self.cut_low_unscaled = cut_low
        self.cut_high_unscaled = cut_high
        self.cut_low = None
        self.cut_high = None
        self.func = func
        self.args = args
        self.scale_rms = scale_rms

    def apply(self, cube, rms=None):
        if self.scale_rms:
            if rms is None:
                rms = np.std(cube)
            self.cut_low = self.cut_low_unscaled * rms
            self.cut_high = self.cut_high_unscaled * rms
        else:
            self.cut_low = self.cut_low_unscaled
            self.cut_high = self.cut_high_unscaled
        self.data = self.func(cube, *self.args)

# path: a string specifying the path of files. {0} will be replaced with the obsid, {1} will be replaced with the filename. example: '~/home/whatever/observations/{0}/{0}_{1}' 
# obsid: obsid to run
# filters: list of Filter objects
# run_name: prefix to add to filenames specifying what is being done. This is what's stored in the data table in column filter_id
# make_plots: boolean whether to make plots
# save_filtered: whether to save fits images of the filter results - normally False
# obs_name: suffix name of the transient cube
# max_plots: max number of plots to create
# true_mask: something or other
# table_name: suffix name to give the final output table

def TransientSearch(path, obsid, filters, run_name, make_plots, save_filtered, obs_name='transient.hdf5', max_plots=100, true_mask=None, table_name='islands_selected_meta'):
    obs = Observation(path, obsid, obs_name)

    # Ignoring high RMS frames

    frame_rms = np.std(obs.data, axis=(1,2))
    ignore_frames = frame_rms > obs.rms * 1.5
    cube_purged = obs.data[~ignore_frames, :, :]
    if np.count_nonzero(ignore_frames) > 0:
        print('skipped frames:', np.nonzero(ignore_frames)[0], flush=True)
    cube_purged = RemoveLines(cube_purged)

    # Applying filters

    for flr in filters:
        flr.apply(cube_purged)
    
    if save_filtered:
        for flr in filters:
            io.WriteImage(path.format(obs.obsid, run_name+'_'+flr.name), flr.data, obs.header, flr.name)

    # Detect islands
    isl_table, isl_labels, _ = isl.FindIslands(obs, filters, True, run_name=run_name)

    if true_mask is not None:
        cands_coord = SkyCoord(isl_table['ra_deg'], isl_table['dec_deg'], unit=(u.deg, u.deg), frame="fk5")
        _, d2d, _ = cands_coord.match_to_catalog_sky(true_mask)
        isl_table = isl_table[d2d < 1*u.arcmin]

    # Cross-match with catalogue of known sources in observation
    knw_table = knw.FindKnownSources(obs, os.getenv('GGSM', '~/GLEAM-X-pipeline/models/GGSM.fits'), filters)
    cm.CrossMatch(isl_table, knw_table, obs)

    isl_table_selected = sel.SelectSources(obs, isl_table, isl_labels, filters, max_count=max_plots)

    print('obsid:', obsid, '    freq:', obs.freq, '    rms:', obs.rms, '    pointing:', obs.cent.ra.to_string(u.hour), obs.cent.dec.to_string(u.degree))
    print('candidates', len(isl_table), '->', len(isl_table_selected), flush=True)
    for flr in filters:
        print(flr.name, np.count_nonzero(isl_table['valid_'+flr.name]), '->', np.count_nonzero(isl_table_selected['valid_'+flr.name]), flush=True)

    # knw_table.write(path.format(obsid, 'known.fits'), format='fits', overwrite=True)
    isl_table.write(path.format(obsid, run_name+'_islands.fits'), format='fits', overwrite=True)
    isl_table_selected.write(path.format(obsid, run_name+'_islands_selected.fits'), format='fits', overwrite=True)
    obs.ncands = f'{len(isl_table_selected)} / {len(isl_table)}'
    
    if make_plots:
        if len(isl_table_selected) <= max_plots:
            for candidate in isl_table_selected:
                diagnostic.DiagnosticPlot(path, obs, filters, candidate, isl_labels, run_name)

    # Prepare for upload
    isl_table_selected.add_column(isl_table_selected['maj_rad_pix'], name='rad_pix')
    isl_table_selected.add_column(isl_table_selected['maj_rad_deg'], name='rad_deg')

    fits_fields = ['obs_id', 'filter_id', 'cand_id', 'x_pix', 'y_pix', 'ra_deg', 'dec_deg', 'area_pix', 'rad_pix', 'rad_deg', 'cent_sep_deg', 'peak_flux', 'beam', 'obs_cent_freq', 'det_stat', 'nks_sep_deg', 'nks_x_pix', 'nks_y_pix', 'nks_ra_deg', 'nks_dec_deg', 'nks_flux', 'nks_name', 'nks_flux_rat']
    meta_fields_int = ['min_rad_pix', 'maj_rad_pix', 'peak_frame']
    meta_fields_float = ['rot_deg', 'beam_norm', 'tcg', 'spike', 'rms', 'tcg_norm', 'spike_norm', 'rms_norm', 'nks_corr', 'nks2_sep_deg', 'nks2_flux', 'nks2_corr', 'nks2_flux_rat']
    meta_fields_str = ['nks2_name']
    name_changes = {'peak_flux'    : 'can_peak_flux'   ,
                    'beam'         : 'can_beam'        ,
                    'det_stat'     : 'can_det_stat'    ,
                    'nks_flux_rat' : 'can_nks_flux_rat'}

    new_table = isl_table_selected[fits_fields]
    for key in name_changes:
        new_table.rename_column(key, name_changes[key])
    new_table.add_column(np.zeros(len(new_table), dtype=np.dtype('<S500')), name='meta')

    for i in range(len(new_table)):
        new_table['meta'][i] = '{' + ','.join(
            ['"%s":%d'   % (field, isl_table_selected[field][i]) for field in meta_fields_int] +
            ['"%s":%.4f' % (field, isl_table_selected[field][i]) for field in meta_fields_float] +
            ['"%s":"%s"' % (field, isl_table_selected[field][i]) for field in meta_fields_str]) + '}'

    new_table.write(path.format(obsid, run_name+'_'+table_name+'.fits'), format='fits', overwrite=True)

    return isl_table_selected
