#!/usr/bin/env python

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.visualization import PercentileInterval
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
from astropy.io import fits
import os
from glob import glob
from astropy.wcs import WCS
import os
from astropy.wcs.utils import pixel_to_skycoord

def ShowCutout(fig, axsize, cutout, wcs, isl_labels, candidate, pulsars, ftitle, ctitle, xax=True, yax=True, highlight=False):
    image_cut = cutout.astype(np.float32)
    image_min, image_max = np.min(image_cut), np.max(image_cut)
    ax = fig.add_axes(axsize, projection=wcs)
    im = ax.imshow(image_cut, vmin=image_min, vmax=image_max, origin='lower', cmap='gray')
    ax.set_title(ftitle)
    cbar = plt.colorbar(im, ax=ax, label=ctitle)
    if xax:
        ax.set_xlabel('RA')
    else:
        ax.coords['ra'].set_ticklabel_visible(False)
    if yax:
        ax.set_ylabel('Dec')
    else:
        ax.coords['dec'].set_ticklabel_visible(False)
    ax.contour(isl_labels, levels=0, linewidths=0.5, colors=['blue'], extent=im.get_extent())
    for p in pulsars:
        ax.scatter(p["RAJ2000"], p["DEJ2000"], marker='o', facecolors='none', edgecolors='green', transform=ax.get_transform('world'))
        ax.text(p["RAJ2000"]+0.03, p["DEJ2000"]+0.03, "PSR{0}".format(p["PSRJ"]), color="green", transform=ax.get_transform('world'))
    ax.plot(candidate['nks1_ra_deg'], candidate['nks1_dec_deg'], 'xr', transform=ax.get_transform('world'))
    ax.plot(candidate['nks2_ra_deg'], candidate['nks2_dec_deg'], 'xg', transform=ax.get_transform('world'))
    if highlight:
        cbar.ax.yaxis.label.set_color('red')
    return ax

def ShowCurve(fig, axsize, obs, candidate):
    time = np.arange(0, obs.shape[0]) * obs.tstep
    ax = fig.add_axes(axsize)
    markers, caps, bars = ax.errorbar(time, candidate['curve'], fmt='b-', label='Candidate', yerr=obs.rms)
    (bar.set_alpha(0.5) for bar in bars)
    ax.plot(time, candidate['nks1_curve'], 'r-', alpha=0.5, label='Known 1')
    ax.plot(time, candidate['nks2_curve'], 'g-', alpha=0.5, label='Known 2')
    ax.axhline(obs.mean, color='black', linestyle=':')
    ax.axhline(obs.mean - obs.rms, color='black', linestyle=':')
    ax.axhline(obs.mean + obs.rms, color='black', linestyle=':')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flux (Jy)')
    ax.legend()

def ShowHist(fig, axsize, obs_cutout, candidate):
    ax = fig.add_axes(axsize)
    cut_counts, cut_bins = np.histogram(obs_cutout.reshape(-1), bins=20, density=True)
    ax.stairs(cut_counts, cut_bins, fill=True, color='blue', alpha=0.5, label='Transient cutout pixels')
    ax.axvline(candidate['peak_flux'], color='red', label='Candidate peak')
    ax.set_yscale('log')
    ax.set_xlabel('Flux (Jy)')
    ax.set_ylabel('Number density')
    ax.legend()

def DiagnosticPlot(path, obs, filters, candidate, isl_labels):
    boxsize = 1*u.deg
    skycoord = SkyCoord(ra=candidate['ra_deg'], dec=candidate['dec_deg'], unit=(u.deg, u.deg), frame='fk5')

    # Getting image cutouts
    peak_frame = Cutout2D(obs.data[candidate['peak_frame']], skycoord, boxsize, wcs=obs.wcs)
    island_cutout = Cutout2D(isl_labels, skycoord, boxsize, wcs=obs.wcs)
    island = island_cutout.data == candidate['cand_id']
    flr_cut = [Cutout2D(flr.data, skycoord, boxsize, wcs=obs.wcs).data for flr in filters]

    try:
        deep_fname = path.format(obs.obsid, 'deep-MFS-image-pb.fits')
        if not os.path.isfile(deep_fname):
            os.system('scp -i id_rsa ubuntu@146.118.68.233:/mnt/gxarchive/Archived_Obsids/{0}/{0}_deep-MFS-image-pb.fits {1}'.format(obs.obsid, deep_fname))
        deep = fits.open(deep_fname)
        deep_data = np.squeeze(deep[0].data)
        deep_wcs = WCS(deep[0].header, naxis=['longitude', 'latitude'])
    except FileNotFoundError:
        deep_data = np.zeros(obs.shape[1:])
        deep_wcs = obs.wcs
    deep = Cutout2D(deep_data, skycoord, boxsize, wcs=deep_wcs)

    obs_cutout = []
    for i in range(obs.shape[0]):
        obs_cutout.append(Cutout2D(obs.data[i], skycoord, boxsize, wcs=obs.wcs).data)
    obs_cutout = np.stack(obs_cutout, axis=0)

    try:
        # 'GLEAM 72-103 MHz', 'GLEAM 103-134 MHz', 'GLEAM 139-170 MHz', 'GLEAM 170-231 MHz'
        survey='GLEAM 170-231 MHz'
        gleam_hdu = SkyView.get_images(position=skycoord, survey=survey, radius=boxsize)[0][0]
        # gleam, _ = reproject_interp(gleam_hdu, peak_frame.wcs, peak_frame.data.shape)
        gleam_data = gleam_hdu.data
        gleam_wcs = WCS(gleam_hdu.header, naxis=2)
    except Exception as e:
        gleam_hdu = fits.open(os.getenv('GLEAM_GP', "~/Documents/MWA-GPM-data/GLEAM_GP.fits"))[0]
        gleam_data, _ = reproject_interp(gleam_hdu, peak_frame.wcs, peak_frame.data.shape)
        gleam_wcs = peak_frame.wcs
        print(e, flush=True)

    # Getting pulsar catalogue
    psrs = fits.open(os.getenv('ATNF_PULSAR_CAT', "~/Documents/MWA-GPM-data/atnf_pulsar_cat.fits"))[1].data
    psr_coords = SkyCoord(psrs["RAJ2000"], psrs["DEJ2000"], unit=(u.deg, u.deg))
    idx_psrs = psr_coords.separation(skycoord) < boxsize / 2
    psrs = psrs[idx_psrs]

    # Placing plots on figure
    fig = plt.figure(figsize=(28, 14))
    #                left   bottom width  height
    ShowCutout(fig, [0.100, 0.350, 0.200, 0.300], deep.data      , deep.wcs      , island, candidate, psrs, 'Deep'      , 'Jy')
    ShowCutout(fig, [0.100, 0.750, 0.200, 0.300], gleam_data     , gleam_wcs     , island, candidate, psrs, 'GLEAM'     , 'Jy')
    ShowCutout(fig, [0.325, 0.350, 0.200, 0.300], peak_frame.data, peak_frame.wcs, island, candidate, psrs, 'Peak Frame', 'Jy')
    ShowCurve( fig, [0.333, 0.750, 0.375, 0.300], obs, candidate)
    ShowHist(  fig, [0.555, 0.350, 0.150, 0.300], obs_cutout, candidate)
    
    ShowCutout(fig, [0.725, 0.350, 0.150, 0.200], flr_cut[0], peak_frame.wcs, island, candidate, psrs, None, filters[0].name, highlight=candidate['valid_'+filters[0].name])
    ShowCutout(fig, [0.725, 0.600, 0.150, 0.200], flr_cut[1], peak_frame.wcs, island, candidate, psrs, None, filters[1].name, highlight=candidate['valid_'+filters[1].name])
    ShowCutout(fig, [0.725, 0.850, 0.150, 0.200], flr_cut[2], peak_frame.wcs, island, candidate, psrs, None, filters[2].name, highlight=candidate['valid_'+filters[2].name])

    fig.savefig(path.format(obs.obsid, f'candidate_{candidate["cand_id"]}.png'), bbox_inches="tight")
    plt.close(fig)

    # Cutout GIF
    dpi = 10
    vmin, vmax = np.nanmin(obs_cutout), np.nanmax(obs_cutout)
    # Plotting each frame
    for i in np.arange(0, obs.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(obs_cutout[i], vmin=vmin, vmax=vmax, origin="lower")
        plt.axis("off")
        plt.margins(x=0)
        plt.margins(y=0)
        fname = path.format(obs.obsid, f"{i:02d}.png")
        fig.savefig(fname, bbox_inches="tight", dpi=dpi, pad_inches = 0)
        plt.close(fig)
    # Combining pngs to gif
    os.system("convert {0} {1}".format(
        path.format(obs.obsid, '??.png'),
        path.format(obs.obsid, f'candidate_{candidate["cand_id"]}.gif')))
    # Removing png frames
    madefiles = glob(path.format(obs.obsid, '??.png'))
    for f in madefiles:
        os.remove(f)
