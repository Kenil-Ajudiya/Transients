import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy import units as u
from astropy.table import Table
from astropy.table import Column

def FindKnownSources(obs, catalogue, filters):
    # Low-priority TODO: make this more generalised to any kind of input catalogue
    tab = fits.open(catalogue)
    cat = tab[1].data
    ra, dec = obs.cent.ra.value, obs.cent.dec.value
    # Make some very basic cuts on dec before doing the big transform, to save time and memory
    # RA is harder because it wraps, so don't bother
    ind = np.where(tab[1].data["DEJ2000"] > dec - obs.fov)
    ind = np.intersect1d(ind, np.where(tab[1].data["DEJ2000"] < dec + obs.fov))
    cat = tab[1].data[ind]
    catalog = SkyCoord(ra = cat["RAJ2000"], dec = cat["DEJ2000"], unit = (u.deg, u.deg), frame = "fk5")

    separations = np.array(obs.cent.separation(catalog))
    # Select sources within crop radius of your original RA and Dec
    ind = np.where(separations < obs.fov)
    catalog = catalog[ind]
    
    # Only bother for sources bright enough to appear in a single snapshot
    bright = cat["S_200"] > 0.1
    #bright = np.ones(len(cat["S_200"]), dtype=bool)
    
    coords = SkyCoord(cat[bright]["RAJ2000"], cat[bright]["DEJ2000"], unit = (u.deg, u.deg), frame = "fk5")
    x, y = skycoord_to_pixel(coords, obs.wcs)
    inside = (x >= 0) & (x < obs.shape[2]) & (y >= 0) & (y < obs.shape[1])
    x = RoundInt(x[inside], 0, obs.shape[2]-1)
    y = RoundInt(y[inside], 0, obs.shape[1]-1)

    brightness = np.array(cat["S_200"][bright][inside] * ((obs.freq / 2e8) ** cat["alpha"][bright][inside]))
    ra = coords.ra.degree
    dec = coords.dec.degree

    # x, y, ra, and dec are already "bright" so the only subset that matters to them is "inside"
    # brightness only calculated for "bright" "inside" sources
    # source names are from original catalogue so need both subsets

    table = Table([
        x, y, ra[inside], dec[inside], brightness, (cat["Name"][bright])[inside]],
        names=['x_pix', 'y_pix', 'ra_deg', 'dec_deg', 'flux', 'name'],
        units=[u.pix, u.pix, u.deg, u.deg, u.Jy, None]) 

    # Grabbing the filter value at the nks location
    for flr in filters:
        table.add_column(Column(data=flr.data[y, x], name=flr.name))
        table.add_column(Column(data=flr.data[y, x] / table['flux'], name=flr.name+'_norm'))

    return table

def RoundInt(vals, minval, maxval):
    result = np.rint(vals).astype(np.int64)
    result = np.where(result < minval, minval, result)
    result = np.where(result > maxval, maxval, result)
    return result
