#!/usr/bin/env python

from astropy.io import fits
import numpy as np
from astropy.wcs.utils import pixel_to_skycoord
from astropy import units as u
import sys
import FileIO as io
import island as isl
import gaussian as g
from astropy.table import Table, Column
import pandas as pd

def InsertModelled(obs_data, num, flux_min, flux_max, flux_base=None, scintil=0.1, dur_min=0.5, dur_max=10, obs_id=None, obs_wcs=None, nobeam=False):
    x_num, y_num = int(np.sqrt(num)), int(np.sqrt(num))
    x_min, x_max = 200, obs_data.shape[2] - 200
    y_min, y_max = 200, obs_data.shape[1] - 200
    x_pix, y_pix = np.meshgrid(np.linspace(x_min, x_max, x_num), np.linspace(y_min, y_max, y_num))
    x_pix, y_pix = x_pix.astype(int).flat, y_pix.astype(int).flat
    sky = pixel_to_skycoord(x_pix, y_pix, obs_wcs)
    ra_deg = sky.ra.deg
    dec_deg = sky.dec.deg
    num = len(x_pix)
    if flux_base is None:
        peak_flux = np.linspace(flux_min, flux_max, num)
    else:
        peak_flux_log = np.linspace(np.log(flux_min), np.log(flux_max), num) / np.log(flux_base)
        peak_flux = flux_base ** peak_flux_log
    np.random.shuffle(peak_flux)
    rad_pix = np.ones(num)
    dur = dur_min + np.random.rand(num) * (dur_max - dur_min)
    
    beam = np.empty(num)
    
    for i in range(num):
        if obs_id is None:
            beam[i] = 1
        else:
            beam[i] = isl.GetBeamAtCoords(obs_id, ra_deg[i], dec_deg[i])
        mod_data = MakeTrans((dur[i], rad_pix[i], rad_pix[i]), (obs_data.shape[0], 10, 10), shift=(5, scintil, scintil)) * peak_flux[i]
        if not nobeam:
            mod_data *= beam[i]
        obs_data[:, y_pix[i]-5:y_pix[i]+5, x_pix[i]-5:x_pix[i]+5] += mod_data
        
    return x_pix, y_pix, ra_deg, dec_deg, peak_flux, dur, beam

def MakeTrans(std, shape, shift=(1, 0, 0), thermal=(0, 1, 1), profile=None):
    y_noise = g.SmoothNoise((shift[0], 1, 1), (shape[0], 1, 1)) * shift[1]
    x_noise = g.SmoothNoise((shift[0], 1, 1), (shape[0], 1, 1)) * shift[2]
    
    if profile is None:
        profile = g.Gaussian((std[0], 1, 1), (shape[0], 1, 1))
    else:
        profile = np.reshape(profile, (shape[0], 1, 1))
    
    x, y = np.meshgrid(range(shape[1]), range(shape[2]))
    rx, ry = (shape[2] - 1) / 2, (shape[1] - 1) / 2
    trans = profile * np.exp(-((x - rx + x_noise)**2 / std[1]**2 + (y - ry + y_noise)**2 / std[2]**2) / 2)

    noise = np.zeros(shape)
    
    if thermal[0] > 0:
        for i in range(shape[0]):
            noise[i] = g.SmoothNoise((thermal[1], thermal[2]), (shape[1], shape[2]))

    return trans - np.mean(trans, axis=0) + thermal[0] * noise
        
if __name__=='__main__':
    obslist = pd.read_csv('100_obs.csv')
    path = '/media/septagonic/CORSAIR/gxarchive/{0}/{0}_{1}'
    obs_idx = 0

    for obsid in obslist.obsid:
        print('--------------', obsid, '------', obs_idx)
        obs_idx += 1

        obs_data, obs_header = io.ReadImage(path.format(obsid, 'transient.hdf5'))
        obs_wcs = io.ObsWCS(obs_header)
        # freq = int(obs_header['FREQ'] / 1e6)
        # args.fmax *= (freq / 154)**-1.2
        x_pix, y_pix, ra_deg, dec_deg, flux, dur, beam = InsertModelled(
            obs_data  = obs_data,
            num       = 100,
            flux_min  = 0.1,
            flux_max  = 3,
            scintil   = 2,
            dur_min   = 0.1,
            dur_max   = 5,
            flat      = True,
            obs_id    = obsid,
            obs_wcs   = obs_wcs,
            nobeam    = True)
        
        table = Table([
            Column(data=x_pix  , name='x_pix'  , unit=u.pix),
            Column(data=y_pix  , name='y_pix'  , unit=u.pix),
            Column(data=ra_deg , name='ra_deg' , unit=u.deg),
            Column(data=dec_deg, name='dec_deg', unit=u.deg),
            Column(data=flux   , name='flux'   , unit=u.Jy ),
            Column(data=dur    , name='dur'    , unit=u.s  ),
            Column(data=beam   , name='beam'   , unit=None)])
        
        io.WriteImage(path.format(obsid, 'modcube_scint.fits'), obs_data, obs_header, sys.argv)
        table.write(path.format(obsid, 'modtab_scint.fits'), format='fits', overwrite=True)
