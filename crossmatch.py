from astropy.table import Column
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import units as u

def CrossMatch(cands, known, obs, radius=4*u.arcmin):
    cands_coord = SkyCoord(cands['ra_deg'], cands['dec_deg'], unit=(u.deg, u.deg), frame="fk5")
    known_coord = SkyCoord(known['ra_deg'], known['dec_deg'], unit=(u.deg, u.deg), frame="fk5")

    cands.add_columns([
        Column(data=np.zeros(len(cands), dtype=np.int64)  , name='nks1_idx'    , unit=None),
        Column(data=np.zeros(len(cands), dtype=np.int64)  , name='nks2_idx'    , unit=None),
        Column(data=np.zeros(len(cands), dtype=np.float64), name='nks1_sep_deg', unit=u.deg),
        Column(data=np.zeros(len(cands), dtype=np.float64), name='nks2_sep_deg', unit=u.deg)])

    for i in range(len(cands)):
        search_rad = max(radius, 2*cands['maj_rad_deg'][i]*u.deg)
        sep = cands_coord[i].separation(known_coord)
        # Find two nearby sources. bright_idx is the brightest within radius if exists, or the closest.
        # close_idx is the closest that isn't bright_idx
        sepsort = np.argsort(sep)
        match_idx_arr = np.nonzero(sep <= search_rad)[0]
        if len(match_idx_arr) >= 1:
            bright_idx = match_idx_arr[np.argmax(known['flux'][match_idx_arr])]
            close_idx = sepsort[0]
            if close_idx == bright_idx:
                close_idx = sepsort[1]
        else:
            bright_idx = sepsort[0]
            close_idx = sepsort[1]
        # Store the match index and separation in the candidate table
        cands['nks1_idx'][i] = bright_idx
        cands['nks1_sep_deg'][i] = sep[bright_idx].deg
        cands['nks2_idx'][i] = close_idx
        cands['nks2_sep_deg'][i] = sep[close_idx].deg
    
    # Copying data of known sources to the candidate table
    for nks in ['nks1', 'nks2']:
        # Copying fields from the known table to the cands table, with prefix added to column names
        for cname in known.colnames:
            cands[f'{nks}_{cname}'] = known[cname][cands[f'{nks}_idx']]
        
        # Pixel coordinates of known sources
        x = RoundInt(cands[f'{nks}_x_pix'], 0, obs.shape[2]-1)
        y = RoundInt(cands[f'{nks}_y_pix'], 0, obs.shape[1]-1)

        # Light curves of known sources
        curves = obs.data[:, y, x].transpose()
        
        # Calculate correlation coefficient between light curve of known source and candidate
        corr = np.zeros(len(cands))
        for i in range(len(cands)):
            curve1 = curves[i] - curves[i].mean()
            curve2 = cands['curve'][i] - cands['curve'][i].mean()
            corr[i] = np.sum(curve1 * curve2) / np.sqrt(np.sum(curve1**2) * np.sum(curve2**2))

        # Add results to the candidate table
        cands.add_columns([
            Column(data=curves             , name=f'{nks}_curve', unit=u.Jy),
            Column(data=corr               , name=f'{nks}_corr' , unit=None),
            Column(data=curves.max(axis=1) , name=f'{nks}_max'  , unit=u.Jy),
            Column(data=curves.mean(axis=1), name=f'{nks}_mean' , unit=u.Jy)])

def RoundInt(vals, minval, maxval):
    result = np.rint(vals).astype(np.int64)
    result = np.where(result < minval, minval, result)
    result = np.where(result > maxval, maxval, result)
    return result
