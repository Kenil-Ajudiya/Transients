import numpy as np
import matplotlib.pyplot as plt
from transient_search import Observation
from astropy.wcs.utils import pixel_to_skycoord
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.ticker import LogLocator

# READING OBSERVATION DATA
path = '/media/septagonic/CORSAIR/gxarchive/{0}/{0}_{1}'
obsid = '1201786160'
obs_name = 'transient.hdf5'
obs = Observation(path, obsid, obs_name)
rms = np.std(obs.data)
print('FREQ:', obs.freq)

# CREATING FIGURE
fig, ax = plt.subplots(figsize=(4, 4))
plt.subplots_adjust(bottom=0.15, left=0.18, right=0.95, top=0.88)

# HISTOGRAM OF PIXEL VALUES
hist, bin_edges = np.histogram(obs.data.reshape(-1), bins=100, density=False)
ax.stairs(hist, bin_edges, fill=True, label='All pixels', alpha=0.7)

# HISTOGRAM WITHOUT SOURCES
y_grid, x_grid = np.mgrid[0:obs.data.shape[1], 0:obs.data.shape[2]]
x_grid = x_grid.reshape(-1)
y_grid = y_grid.reshape(-1)
skycoord_grid = pixel_to_skycoord(x_grid, y_grid, obs.wcs)

tab = fits.open('~/GLEAM-X-pipeline/models/GGSM.fits')
cat = tab[1].data
ra, dec = obs.cent.ra.value, obs.cent.dec.value
ind = np.where(tab[1].data["DEJ2000"] > dec - obs.fov)
ind = np.intersect1d(ind, np.where(tab[1].data["DEJ2000"] < dec + obs.fov))
cat = tab[1].data[ind]
cat = cat[cat["S_200"] > 0.00]
skycoord_cat = SkyCoord(ra=cat["RAJ2000"], dec=cat["DEJ2000"], unit=(u.deg, u.deg), frame="fk5")

idx, d2d, d3d = skycoord_grid.match_to_catalog_sky(skycoord_cat)
sample = (d2d > 4*u.arcmin).reshape(obs.data.shape[1:])

hist, bin_edges = np.histogram(obs.data[:, sample].reshape(-1), bins=bin_edges, density=False)
ax.stairs(hist, bin_edges, fill=True, label='Non-masked pixels', alpha=0.7)

# fig2, axs2 = plt.subplots(1, 1)
# axs2.pcolor(np.max(np.abs(obs.data), axis=0), vmin=0, vmax=0.15)
# axs2.contour(sample)

# NORMAL DISTRIBUTION
x = np.linspace(-7*rms, 7*rms, 1000)
gauss = np.max(hist)*np.exp(-x**2 / 2 / rms**2)
ax.plot(x, gauss, c='k', label='Normal distribution')

# FINALISING FIGURE
ax.set_xlabel('Pixel brightness (Jy/beam)')
ax.set_ylabel('Number of pixels')
ax.legend(loc='upper right')
ax.set_yscale('log')
# ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
# ax.tick_params(axis='y', which='minor', length=40, color='black')
secax = ax.secondary_xaxis('top', functions=(lambda x:x/rms, lambda x:x*rms))
secax.set_xlabel('$\\sigma$')

plt.show()