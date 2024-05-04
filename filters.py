import numpy as np
import scipy.signal as sig
import gaussian as g

def Spike(cube, sigma=5):
    std = np.std(cube, axis=0, keepdims=True)
    # Basic sigma-clip
    cube_copy = cube - np.mean(cube, axis=0, keepdims=True)
    cube_copy[cube_copy / std > sigma] = np.nan
    std = np.nanstd(cube_copy, axis=0, keepdims=True)
    return np.nanmax((cube - np.nanmean(cube_copy, axis=0, keepdims=True)) / std, axis=0)

def Correlator(cube, std, shape):
    window = g.Gaussian(std, shape)
    window -= np.mean(window)
    window /= np.sum(window**2)
    filtered = sig.oaconvolve(cube-np.mean(cube, axis=0), window, 'same')
    
    return np.max(filtered, axis=0)

def RMS(cube):
    return np.std(cube, axis=0)

def MultiScaleCorrFilter(cube, r=None):
    flr_data = np.zeros(cube.shape)
    
    #for win_size in 2**np.arange(int(np.log2(cube.shape[0]))):
    for win_size in 2**np.arange(int(np.log2(10))):
    #for win_size in range(1, cube.shape[0]-2):
        if r is None:
            win_std = (win_size / 6, win_size / 6, win_size / 6)
            win_size = (win_size, win_size, win_size)
        else:
            win_std = (win_size / 6, r / 6, r / 6)
            win_size = (win_size, r, r)
        
        flr_temp = Correlator(cube, win_std, win_size)
        flr_greater = flr_temp > flr_data
        flr_data[flr_greater] = flr_temp[flr_greater]
        
    return flr_data
    

# gaussian smoothing filter
# Produces smooth noise

def SmoothFilter(cube, std, shape):
    # generating window
    window = g.Gaussian(std, shape)
    window /= np.sqrt(np.sum(window**2))

    # filtering
    return sig.convolve(cube, window, 'same')

# wacky spectrogram filter i came up with
# This calculates the FFT of the time-series of each pixel. The average time-series FFT
# of the cube is compared to that of each pixel by the root sum square difference. Pixels
# with the largest difference are the most tlikely transient candidates. This is done to
# time-slices of a few frames rather than an entire 24 frame cube because the transient
# only appears in some frames. This code should be modified to also return the frame of
# the max difference too to make it easier to find the time of the transient

def FFTFilter(cube, width=None, step=1):
    if width is None:
        width = cube.shape[0] - 1
    
    window = g.Gaussian((width/3, 1, 1), (width, 1, 1))
    inds = range(0, cube.shape[0] - width, step)
    filtered = np.zeros((len(inds), cube.shape[1], cube.shape[2]))
    
    for i in range(len(inds)):
        part = cube[inds[i]:inds[i]+width] * window
        fft = np.abs(np.fft.rfft(part, axis=0))
        fft_mean = np.expand_dims(np.expand_dims(np.mean(fft, axis=(1, 2)), 1), 2)
        filtered[i] = np.sum((fft - fft_mean)**2, axis=0)
    
    #collapsed = np.max(filtered, axis=0)
        
    return filtered

def GradFilter(cube):
    filtered = cube[1:] - cube[:-1]
    #collapsed = np.sqrt(np.sum(np.power(filtered, 2.0), axis=0))
    return filtered