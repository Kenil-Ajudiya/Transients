from transient_search import *
import os
import sys

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])

with open('obsids_shuf.txt', 'r') as file:
    lines = file.readlines()
    obsids = [line.strip() for line in lines]

path = os.path.expanduser('~/candidates/{0}/{0}_{1}')
workdir = os.path.expanduser('~/candidates/{0}')
obs_idx = start_idx

filters = [
    Filter('tcg'  , 5, 8, True , fil.Correlator, (1,1,1), (25,1,1)),
    Filter('spike', 5, 8, False, fil.Spike, 4),
    Filter('rms'  , 2, 3, True , fil.RMS)]

for obsid in obsids[start_idx:end_idx]:
    print('--------------', obsid, '------', obs_idx, flush=True)
    obs_idx += 1
    cube_fname = path.format(obsid, 'transient.hdf5')
    if not os.path.exists(workdir.format(obsid)):
        os.system('mkdir ~/candidates/{0}'.format(obsid))
        os.system('scp -i id_rsa ubuntu@146.118.68.233:/mnt/gxarchive/Archived_Obsids/{0}/{0}_transient.hdf5 {1}'.format(obsid, cube_fname))
        if os.path.isfile(cube_fname):
            cands = TransientSearch(path, obsid, filters, 'try_1', True, False, max_plots=15)
            if len(cands) > 15:
                os.system(f'echo {obsid}, {len(cands)} >> bad_obsids.txt')
            os.system(f'rm {cube_fname}')
            if os.path.isfile(path.format(obsid, "deep-MFS-image-pb.fits")):
                os.system(f'rm {path.format(obsid, "deep-MFS-image-pb.fits")}')
        else:
            os.system(f'rm -r ~/candidates/{obsid}')