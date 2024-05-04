#!/usr/bin/env python

from astropy.io import fits
import numpy as np
from h5py import File
from astropy.wcs import WCS
import os
import re
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import SkyCoord
from astropy import units as u

# FITS format code         Description                     8-bit bytes

# L                        logical (Boolean)               1
# X                        bit                             *
# B                        Unsigned byte                   1
# I                        16-bit integer                  2
# J                        32-bit integer                  4
# K                        64-bit integer                  8
# A                        character                       1
# E                        single precision float (32-bit) 4
# D                        double precision float (64-bit) 8
# C                        single precision complex        8
# M                        double precision complex        16
# P                        array descriptor                8
# Q                        array descriptor                16

def ReadHeader(fname):
    obs_name, obs_ext = os.path.splitext(os.path.basename(fname))
    hist_data = []
    hist_header = []
    
    if obs_ext == '.fits': # Open fits file
        hdul = fits.open(fname)
        obs_header = hdul[0].header
        obs_header['TIMESTEP'] = obs_header['CDELT3']
        hdul.close()
    elif obs_ext == '.hdf5': # Open hdf5 file
        with File(fname) as df:
            obs_header = dict(df['header'].attrs)
            obs_header['FREQ'] = obs_header['CRVAL3']
            obs_header['TIMESTEP'] = df.attrs['TIME_INTERVAL']
    else:
        print('Only support fits and hdf5 tables!')
        exit()
    
    return obs_header

def ReadImage(fname):
    obs_name, obs_ext = os.path.splitext(os.path.basename(fname))
    hist_data = []
    hist_header = []
    
    if obs_ext == '.fits': # Open fits file
        hdul = fits.open(fname)
        obs_data = np.array(hdul[0].data, dtype=float)
        obs_header = hdul[0].header
        obs_header['TIMESTEP'] = obs_header['CDELT3']
        hdul.close()
    elif obs_ext == '.hdf5': # Open hdf5 file
        with File(fname) as df:
            obs_data = np.squeeze(np.array(df['image'])).astype('float32')
            if len(obs_data.shape) == 3:
                obs_data = obs_data.swapaxes(0, 2).swapaxes(1, 2)
            obs_header = dict(df['header'].attrs)
            obs_header['FREQ'] = obs_header['CRVAL3']
            obs_header['TIMESTEP'] = df.attrs['TIME_INTERVAL']
    else:
        print('Only support fits and hdf5 tables!')
        exit()
    
    return obs_data, obs_header

def ReadTable(fname):
    obs_name, obs_ext = os.path.splitext(os.path.basename(fname))
    if obs_ext == '.fits': # Open fits file
        hdul = fits.open(fname)
        if len(hdul) < 2:
            print('Could not find table hdu!')
            exit()
        tab_data = hdul[1].data
        tab_header = hdul[1].header
        img_header = hdul[0].header
        hdul.close()
    else:
        print('Only support fits tables!')
        exit()
        
    return tab_data, tab_header, img_header

def GetDescriptions(header):
    descriptions = []
    for i in range(1, header['TFIELDS'] + 1):
        descriptions += [header['TCOMM' + str(i)]]
    return descriptions

def GetUnits(header):
    units = []
    for i in range(1, header['TFIELDS'] + 1):
        units += [header['TUNIT' + str(i)]]
    return units

def GetNames(header):
    names = []
    for i in range(1, header['TFIELDS'] + 1):
        names += [header['TTYPE' + str(i)]]
    return names

def GetFormats(header):
    formats = []
    for i in range(1, header['TFIELDS'] + 1):
        formats += [header['TFORM' + str(i)]]
    return formats

def GetColumns(data, header):
    columns = []
    for i in range(1, header['TFIELDS'] + 1):
        columns += [data[header['TTYPE' + str(i)]]]
    return columns

def CombineHeaders(header, header_add):
    if header_add is not None:
        for key, value in header_add.items():
            if key[:5] == 'NAXIS':
                header['NAXPR' + key[5:]] = header_add[key]
            elif key not in header:
                header[key] = value

def HistHeader(header, hist_add):
    if hist_add is not None:
        n_hist = 1
        if 'NHIST' in header:
            n_hist = header['NHIST'] + 1
        header['NHIST'] = n_hist
        if type(hist_add) == list:
            hist_add = ' '.join(hist_add)
        header['HIST' + str(n_hist)] = hist_add

def WriteTable(fname, col_vals, col_forms, col_names, col_units, col_descr, header=None, new_hist=None):
    cols = []
    
    for i in range(len(col_vals)):
        cols += [fits.Column(name=col_names[i], format=col_forms[i], array=col_vals[i])]
    
    hdu_table = fits.BinTableHDU.from_columns(cols)
    
    for i in range(len(col_vals)):
        hdu_table.header['TCOMM' + str(i+1)] = str(col_descr[i])
        hdu_table.header['TUNIT' + str(i+1)] = str(col_units[i])
    
    hdu_image = fits.PrimaryHDU([[0, 0], [0, 0]])
    CombineHeaders(hdu_image.header, header)
    HistHeader(hdu_image.header, new_hist)
    
    hdul = fits.HDUList([hdu_image, hdu_table])
    hdul.writeto(fname, overwrite=True)

def WriteImage(fname, img_vals, header, new_hist):
    hdu_image = fits.PrimaryHDU(img_vals)
    CombineHeaders(hdu_image.header, header)
    HistHeader(hdu_image.header, new_hist)
    hdul = fits.HDUList([hdu_image])
    hdul.writeto(fname, overwrite=True)

def ObsWCS(obs_header):
    return WCS(obs_header, naxis=2)

def ObsCentPoint(obs_header):
    obs_wcs = ObsWCS(obs_header)
    #if 'NAXIS2' in obs_header and 'NAXIS1' in obs_header:
        #obs_skycoord = obs_wcs.pixel_to_world(obs_header['NAXIS1']/2, obs_header['NAXIS2']/2)
    #elif 'NAXPR2' in obs_header and 'NAXPR1' in obs_header:
        #obs_skycoord = obs_wcs.pixel_to_world(obs_header['NAXPR1']/2, obs_header['NAXPR2']/2)
    obs_skycoord = obs_wcs.pixel_to_world(1200, 1200)
    return SkyCoord(obs_skycoord.ra.degree, obs_skycoord.dec.degree, unit=(u.deg, u.deg), frame="fk5")

def ObsShape(obs_header):
    n = 1
    shape = []
    axtype = 'NAXIS'
    
    if 'NAXPR1' in obs_header:
        axtype='NAXPR'
    
    while axtype + str(n) in obs_header:
        shape += [obs_header[axtype + str(n)]]
        n += 1
        
    return shape
        
    #return (2400, 2400)

def ObsID(fname):
    obs_name, obs_ext = os.path.splitext(os.path.basename(fname))
    return re.findall(r'\d+', obs_name)[0]
