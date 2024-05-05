import FileIO as io
import filters as fil
import island as isl
import known as knw
import crossmatch as cm
import selection as sel
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import diagnostic
from astropy.io import fits
import os
from glob import glob
import matplotlib.pyplot as plt

class Observation:
    obsid = None
    data = None
    header = None
    time = None
    shape = None
    wcs = None
    freq = None
    tstep = None
    cent = None
    fov = None
    rms = None
    mean = None

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

    def apply(self, cube):
        if self.scale_rms:
            rms = np.std(cube)
            self.cut_low = self.cut_low_unscaled * rms
            self.cut_high = self.cut_high_unscaled * rms
        else:
            self.cut_low = self.cut_low_unscaled
            self.cut_high = self.cut_high_unscaled
        self.data = self.func(cube, *self.args)


def TransientSearch(path, obsid, filters, run_name, make_plots, save_filtered):
    obs = Observation()
    obs.obsid = obsid
    obs.data, obs.header = io.ReadImage(path.format(obs.obsid, 'transient.hdf5'))
    obs.time  = Time(int(obs.obsid), format='gps').utc.iso
    obs.shape = obs.data.shape
    obs.wcs   = io.ObsWCS(obs.header)
    obs.freq  = obs.header["FREQ"]
    obs.tstep = obs.header['TIMESTEP']
    obs.cent  = obs.wcs.pixel_to_world(obs.header["NAXIS1"]/2, obs.header["NAXIS2"]/2)
    obs.cent  = SkyCoord(obs.cent.ra.value, obs.cent.dec.value, unit = (u.deg, u.deg), frame = "fk5")
    obs.fov   = np.sqrt(2) * obs.header["CDELT2"] * max(obs.header["NAXIS1"], obs.header["NAXIS2"]) / 2
    obs.rms   = np.std(obs.data)
    obs.mean  = np.mean(obs.data)

    # Ignoring high RMS frames

    frame_rms = np.std(obs.data, axis=(1,2))
    ignore_frames = frame_rms > obs.rms * 1.5
    cube_purged = obs.data[~ignore_frames, :, :]
    if np.count_nonzero(ignore_frames) > 0:
        print('skipped frames:', np.nonzero(ignore_frames)[0])

    # Applying filters

    for flr in filters:
        flr.apply(cube_purged)
    
    if save_filtered:
        for flr in filters:
            io.WriteImage(path.format(obs.obsid, flr.data, obs.header, flr.name))

    # Detect islands
    isl_table, isl_labels, isl_slices = isl.FindIslands(obs, filters, True)

    # Cross-match with catalogue of known sources in observation
    knw_table = knw.FindKnownSources(obs, os.getenv('GGSM', '~/GLEAM-X-pipeline/models/GGSM.fits'), filters)
    cm.CrossMatch(isl_table, knw_table, obs)

    isl_table_selected = sel.SelectSources(isl_table, isl_labels, filters)

    print(len(isl_table), '->', len(isl_table_selected))
    for flr in filters:
        print(flr.name, np.count_nonzero(isl_table['valid_'+flr.name]), '->', np.count_nonzero(isl_table_selected['valid_'+flr.name]))

    # knw_table.write(path.format(obsid, 'known.fits'), format='fits', overwrite=True)
    isl_table.write(path.format(obsid, run_name+'_islands.fits'), format='fits', overwrite=True)
    isl_table_selected.write(path.format(obsid, run_name+'_islands_selected.fits'), format='fits', overwrite=True)

    if make_plots:
        # Removing old images
        oldfiles = glob(path.format(obs.obsid, 'candidate_*'))
        for f in oldfiles:
            os.remove(f)

        if len(isl_table_selected) <= 10:
            for candidate in isl_table_selected:
                diagnostic.DiagnosticPlot(path, obs, filters, candidate, isl_labels)

    return isl_table_selected

'''

plt.imshow(np.log(np.max(data, axis=0)), cmap='gray')
plt.scatter(isl_table['x_pix'], isl_table['y_pix'])
plt.contour(isl_labels > 0)
# plt.show()

# exit()


curves = isl_table['curve'].data / np.max(isl_table['curve'].data, axis=1, keepdims=True)
group = np.zeros(curves.shape[0], dtype=np.int64)

curves -= np.mean(curves, axis=1, keepdims=True)

# curves = np.random.normal(0, 1, curves.shape)

print(curves.shape)
maxind = np.argmax(isl_table['area_pix'])

def Pearson(a, b):
    return np.sum(a[np.newaxis, :] * b, axis=1) / np.sqrt(np.sum(a[np.newaxis, :]**2, axis=1) * np.sum(b**2, axis=1))

inds = np.arange(curves.shape[0])
fluxsort = np.flip(np.argsort(isl_table['peak_flux']))
inds = inds[fluxsort]
curves = curves[fluxsort]
# corr = np.zeros(curves.shape[0])
corr = np.abs(Pearson(curves[0], curves))
for i in range(1, curves.shape[0]):
    k = 0.75
    pearson = Pearson(curves[i-1], curves[i:])
    corr[i:] = k*corr[i:] + (1-k)*np.abs(pearson)
    r = 1
    # a = (1/r * np.sqrt((isl_table['x_pix'][i:]-isl_table['x_pix'][i-1])**2 + (isl_table['y_pix'][i:]-isl_table['y_pix'][i-1])**2))
    # corr[i:] = corr[i:] ** a
    curves[i:][pearson < 0] *= -1
    corrsort = np.flip(np.argsort(corr[i:]))
    curves[i:] = curves[corrsort+i]
    inds[i:] = inds[corrsort+i]

plt.figure()
plt.imshow(curves, aspect='auto', cmap='jet')
plt.gca().invert_yaxis()

plt.figure()
plt.plot(corr)

# plt.figure()
# for i in range(575, 600):
#     plt.plot(isl_table['curve'][inds[i]])
#     # plt.scatter(isl_table['x_pix'][inds[i]], isl_table['y_pix'][inds[i]])

for i in range(len(isl_table_selected)):
    plt.figure()
    plt.imshow(np.log(np.max(data, axis=0)), cmap='gray')
    plt.scatter(isl_table_selected['x_pix'][i], isl_table_selected['y_pix'][i])
    plt.contour(isl_labels == isl_table_selected['can_idx'][i])

plt.show()
'''