#!/usr/bin/python

import numpy as np

from astropy import units as u
from astropy import coordinates
from astropy.coordinates import SkyCoord

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from matplotlib import rc
from astropy.table import Table

def GetSkyplotAxes(gal_col='black'):
    rc('text', usetex=True)
    rc('font',**{'family':'serif','serif':['serif']})

    plt.rcParams["axes.axisbelow"] = False

    sbplt_pad_left  = 0.125  # the left side of the subplots of the figure
    sbplt_pad_right = 0.9    # the right side of the subplots of the figure
    sbplt_pad_bottom = 0.1   # the bottom of the subplots of the figure
    sbplt_pad_top = 0.9      # the top of the subplots of the figure
    sbplt_pad_wspace = 0.1   # the amount of width reserved for blank space between subplots
    sbplt_pad_hspace = 0.5   # the amount of height reserved for white space between subplots

    subplot_cols = 1
    subplot_rows = 1

    def redo_axis_labels(ticklist):
        new_label_list = []
        for loc in ticklist:
    ##        # Get numeric equivalent, change to hours:
            tick_num = np.degrees(loc)
            if tick_num <= 0.:
                tick_num = tick_num + 360.
            tick_num = 360. - tick_num
            tick_num /= 15.
        # Note that this rounds to integer, might not be what you want:
            tick_string = "{:2.0f}".format(tick_num)
        # Add a raised "h" to denote hours:
            new_label = "$" + tick_string + "^h$"
    #        new_label = "$" + str(tick_num) + "^h$"
            new_label_list.append(new_label)
        return new_label_list

    def hr2rad(x):
        ''' convert hours to radians for the molleweide projection '''
        if x > 12.:
            x -= 24.
        return -np.radians(x*15.)

    def unwrap(x):
        if x > np.radians(180.):
            x -= 2*np.pi
        return -x
    vunwrap = np.vectorize(unwrap)

    plotnum = 1

    fig = plt.figure(figsize=(10,5))

    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)

    # Define already-observed region
    tmp = [0., 0., 12, 12]
    x_obs_1 = np.array([hr2rad(x) for x in tmp])
    y_obs_1 = np.radians([-90, 30., 30, -90.])

    tmp = [12.000001, 12.000001, 23.9999, 23.9999]
    x_obs_2 = np.array([hr2rad(x) for x in tmp])
    y_obs_2 = np.radians([-90., 30., 30., -90.])

    # Define DR2
    tmp = [21., 21., 4., 4.]
    x_obs_3 = np.array([hr2rad(x) for x in tmp])
    y_obs_3 = np.radians([-90, 30., 30, -90.])

    # Define DR1
    tmp = np.array([12., 12., 4., 4.])
    x_es = np.array([hr2rad(x) for x in tmp])
    y_es = np.radians([-32.7, -20.7, -20.7, -32.7])

    #width = np.radians(30.)
    #height = np.radians(30.)
    #ax.add_patch(Ellipse((hr2rad(18.),np.radians(-30.)),
    #                         width=width, height=height, angle = 0.0,
    #                        ))
    # First one is just to get it plotted so I can get the ticklabels
    # fig.savefig("dummy.png")

    # Observed region
    # ax.plot(x_obs_1, y_obs_1, color="blue", alpha=0.1, zorder = -10)
    # ax.fill(x_obs_1, y_obs_1, color="blue", alpha=0.1, zorder = -10)
    # ax.plot(x_obs_2, y_obs_2, color="blue", alpha=0.1, zorder = -10)
    # ax.fill(x_obs_2, y_obs_2, color="blue", alpha=0.1, zorder = -10)
    # DR1
    # ax.plot(x_es, y_es, color="orange", alpha=0.5)
    # ax.fill(x_es, y_es, color="orange", alpha=0.5, label="DR1", zorder=-10)
    # DR2
    # ax.plot(x_obs_3, y_obs_3, color="yellow", alpha=0.3, zorder = -5)
    # ax.fill(x_obs_3, y_obs_3, color="yellow", alpha=0.3, zorder = -5, label="DR2")

    # Plot the Galactic plane
    l = np.arange(0, 360, 3)
    b1 = -10.*np.ones(len(l))
    b2 = 10.*np.ones(len(l))

    gal_1 = SkyCoord(l, b1, unit=(u.deg, u.deg), frame="galactic")
    gal_2 = SkyCoord(l, b2, unit=(u.deg, u.deg), frame="galactic")

    ax.scatter(vunwrap(np.radians(gal_1.fk5.ra.value)), np.radians(gal_1.fk5.dec.value), color=gal_col, zorder=100, marker=".", s=0.5)
    ax.scatter(vunwrap(np.radians(gal_2.fk5.ra.value)), np.radians(gal_2.fk5.dec.value), color=gal_col, zorder=100, marker=".", s=0.5)

    gc = SkyCoord(0.0, 0.0, unit=(u.deg, u.deg), frame="galactic")

    hyda = coordinates.get_icrs_coordinates("Hydra A")
    vira = coordinates.get_icrs_coordinates("Virgo A")
    crab = coordinates.get_icrs_coordinates("Crab")
    cena = coordinates.get_icrs_coordinates("Centaurus A")
    pica = coordinates.get_icrs_coordinates("Pictor A")

    ax.scatter(unwrap(np.radians(gc.fk5.ra.value)), np.radians(gc.fk5.dec.value), marker="*", s=200, label='Galactic center')
    label = 'Bright sources'
    for a in hyda, vira, crab, cena, pica:
        ax.scatter(unwrap(np.radians(a.fk5.ra.value)), np.radians(a.fk5.dec.value), marker="*", color="orange", label=label)
        label=None

    # ticklist = ax.get_xmajorticklabels()
    ticklist = ax.xaxis.get_majorticklocs()
    new_label_list = redo_axis_labels(ticklist)
    new = ax.set_xticklabels(new_label_list)

    ax.legend()
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    # fig.savefig("gleamx_plans_2022.png", dpi=200, pad_inches=0.0, bbox_inches='tight')

    return fig, ax

def SkyplotCands(all_cands, gal_col):
    fig, axs = GetSkyplotAxes(gal_col)
    if all_cands:
        data = Table.read('/media/septagonic/CORSAIR/candidates/try_1_islands_selected.fits', format='fits')
        data['ra'] = -np.radians(data['ra_deg'])
        data['dec'] = np.radians(data['dec_deg'])
        data['ra'] = np.where(data['ra']<-np.pi, data['ra']+2*np.pi, data['ra'])
        axs.scatter(data['ra'], data['dec'], s=0.1, c='blue', label='All candidates')
    # data = np.loadtxt('gxarchive_ra_dec.csv', delimiter=',', dtype=[('obsid', np.int64), ('ra', np.float64), ('dec', np.float64)])
    # data['ra'] = -np.radians(data['ra'])
    # data['dec'] = np.radians(data['dec'])
    # data['ra'] = np.where(data['ra']<-np.pi, data['ra']+2*np.pi, data['ra'])
    # axs.scatter(data['ra'], data['dec'], s=1, c='red', label='Observation centers')

    data = Table.read('/media/septagonic/CORSAIR/group_islands_selected.fits', format='fits')
    data['ra'] = -np.radians(data['ra_deg'])
    data['dec'] = np.radians(data['dec_deg'])
    data['ra'] = np.where(data['ra']<-np.pi, data['ra']+2*np.pi, data['ra'])
    axs.scatter(data['ra'], data['dec'], s=5, c='red', label='Final candidates')

    known_lpt = np.array([
        ['GLEAM-X J162759.5-523504.3', '16h27m59.50s', '-52°35′04.30″'],
        ['GPM J1839-10'              , '18h39m02.00s', '-10°31′49.37″'],
        ['GCRT J1745-3009'           , '17h45m05.00s', '-30°09′52.00″'],
        ['ASKAP J193505.1+214841.0'  , '19h35m05.13s', ' 21°48′41.05″'],
        ['ASKAP J175534.9-252749.1'  , '17h55m34.87s', '-25°27′49.10″'],
        ['CHIME J0630+25'            , '06h30m43.00s', ' 25°23′24.00″'],
        ['ILT J110160.52+552119.62'  , '11h01m50.50s', ' 55°21′19.60″'],
        ['GLEAM-X J0704-37'          , '07h04m07.51s', '-37°05′40.56″'],
        ['ASKAP J1839-0756'          , '18h39m50.40s', '-07°56′35.38″'],
        ['ASKAP J1832-0911'          , '18h32m48.46s', '-09°11′15.30″'],
        # ['ASKAP J175154.89-255135.3' , '17h51m54.89s', '-25°51′35.30″'],
        # ['ASKAP J172755-343119'      , '17h27m55.26s', '-34°31′19.41″'],
        ['ASKAP J144834-685644'      , '14h48m34.29s', '-68°56′44.10″'],
        ['CHIME J1634+44 / ILT J163430+445010', '16h34m29.96s', '44°50′13.50″']])
    lpt_coords = SkyCoord(ra=known_lpt[:,1], dec=known_lpt[:,2], frame='fk5')
    ra = -np.radians(lpt_coords.ra.deg)
    dec = np.radians(lpt_coords.dec.deg)
    ra = np.where(ra<-np.pi, ra+2*np.pi, ra)
    axs.scatter(ra, dec, c='pink', marker='*', label='Known LPT')
    # for i in range(len(known_lpt)):
    #     axs.text(ra[i]+0.1, dec[i], known_lpt[i,0])

    pulsar_group_ids = np.array([13487, 13257, 7911, 12802, 5688, 11050])
    lpt_group_ids = np.array([10311])
    is_pulsar  = np.isin(data['group_idx'].data, pulsar_group_ids)
    is_lpt = np.isin(data['group_idx'].data, lpt_group_ids)

    axs.scatter(data['ra'][is_pulsar | is_lpt], data['dec'][is_pulsar | is_lpt], marker='o', s=100, label='Real transients', facecolors='none', edgecolors='r')

    plt.legend()
    # plt.show()
    return fig, axs

if __name__=='__main__':
    SkyplotCands(True, 'black')
    plt.tight_layout()
    # plt.show()
    plt.savefig('sky_map_cands.png', dpi=600, bbox_inches='tight')