from transient_search import *
import filters as fil
from astropy.io import fits
import os
import sys
import gc

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])

clean_up = False
do_scp = False
scp_source = 'ubuntu@146.118.68.233:/mnt/gxarchive/Archived_Obsids'

# obsids_fname = 'obsids_shif.txt'
# obsids_fname = 'obsids_groups.txt'
# obsids_fname = 'Epoch0084_obsids.txt'
obsids_fname = 'Epoch0789_obsids.txt'

with open(obsids_fname, 'r') as file:
    lines = file.readlines()
    obsids = [line.strip() for line in lines]

# main_dir = 'candidates'
# main_dir = 'groups'
# main_dir = 'Epoch0084_candidates'
main_dir  = 'Epoch0789_candidates'

# true_mask_table = fits.open(os.path.expanduser('~/group_islands.fits'))[1].data
# true_mask = SkyCoord(true_mask_table['ra_deg'], true_mask_table['dec_deg'], unit=(u.deg, u.deg), frame="fk5")
true_mask = None

path = os.path.expanduser('~/'+main_dir+'/{0}/{0}_{1}')
workdir = os.path.expanduser('~/'+main_dir+'/{0}')
obs_idx = start_idx

# filters = [
#     Filter('tcg'  , 5.5, 7, True , fil.Correlator, (1,1,1), (25,1,1)),
#     Filter('spike', 5.5, 8.5, False, fil.Spike, 4),
#     Filter('rms'  , 2.0, 2.5, True , fil.RMS)]

filters = [
    Filter('tcg'  , 5.5, 7.0, True , fil.Correlator, (1,1,1), (125,1,1)),
    Filter('spike', 5.5, 6.0, False, fil.Spike, 4),
    Filter('rms'  , 2.0, 2.25, True , fil.RMS)]

for obsid in obsids[start_idx:end_idx]:
    print('--------------', obsid, '------', obs_idx, flush=True)
    obs_idx += 1
    try:
        cube_fname = path.format(obsid, 'transient.hdf5')
        if not do_scp or not os.path.exists(workdir.format(obsid)):
            if do_scp:
                os.system('mkdir ~/{0}/{1}'.format(main_dir, obsid))
                os.system('scp -i id_rsa {0}/{1}/{1}_transient.hdf5 {2}'.format(scp_source, obsid, cube_fname))
            if os.path.isfile(cube_fname):
                cands = TransientSearch(path, obsid, filters, 'try_1', True, False, max_plots=150, true_mask=true_mask)
                # if len(cands) > 10:
                #     os.system(f'echo {obsid}, {len(cands)} >> bad_obsids.txt')
                if clean_up:
                    os.system(f'rm {cube_fname}')
                    if os.path.isfile(path.format(obsid, "deep-MFS-image-pb.fits")):
                        os.system(f'rm {path.format(obsid, "deep-MFS-image-pb.fits")}')
            elif clean_up:
                os.system(f'rm -r ~/{main_dir}/{obsid}')
    except Exception as e:
        print(e)
    gc.collect()
