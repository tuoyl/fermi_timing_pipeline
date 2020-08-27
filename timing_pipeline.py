#!/usr/bin/env python
import yaml
import numpy as np
import numba 
from astropy.io import fits
import matplotlib.pyplot as plt
import wget
import os

# declare global variables
MJDREFF = 0.00074287037037037
MJDREFI = 51910

def download_data(yamlfile):
    """
    download weekly Photon files and Spacecraft files (if not exists)

    """
    par = get_parlist(yamlfile)
    evfile = par['data']['evfile']
    scfile = par['data']['scfile']
    evfile_filename = os.path.basename(evfile)
    scfile_filename = os.path.basename(scfile)
    evfile_dir = os.path.dirname(evfile)
    scfile_dir = os.path.dirname(scfile)

    if os.path.exists(evfile) & os.path.exists(scfile):
        return
    if not os.path.exists(evfile):
        print("Checking file --> %s not exists"%(evfile))
        print("Retrieving file --> %s"%(evfile))
        wget.download('https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/photon/%s'%(evfile_filename),
                out=evfile)
    else:
        print("Checking file --> %s exists"%(evfile))
    if not os.path.exists(scfile):
        wget.download('https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/%s'%(scfile_filename),
                out=scfile)
    else:
        print("Checking file --> %s exists"%(scfile))

def transfer_hms2degree(ra_time, dec_time):

    ra0 = np.floor(ra_time/3600)
    ra1 = np.floor((ra_time-ra0*3600)/60)
    ra2 = ra_time-ra0*3600-ra1*60
    dec0 = np.floor(dec_time/3600)
    dec1 = np.floor((dec_time-dec0*3600)/60)
    dec2 = dec_time-dec0*3600-dec1*60

    ra = ra0*15.0 + (ra1*15.0)/60.0 + (ra2*15)/3600.0
    if dec0 <= 0:
        dec = dec0 - dec1/60.0 - dec2/3600.0
    else:
        dec = dec0 + dec1/60.0 + dec2/3600.0
    
    return ra, dec

def fermi_gtanalyse(yamlfile):
    par = get_parlist(yamlfile)
    evfile   = par['data']['evfile']
    scfile   = par['data']['scfile']
    gtselect = par['data']['gtselect']
    gtbary   = par['data']['gtbary']

    emin     = par['selection']['emin']
    emax     = par['selection']['emax']
    zmax     = par['selection']['zmax']
    rad      = par['selection']['rad']
    evclass  = par['selection']['evclass']
    if not par['selection']['evtype']:
        evtype = "INDEF"
    else:
        evtype = par['selection']['evtype']

    # get tmin and tmax
    hdulist = fits.open(evfile)
    time = hdulist[1].data.field("Time")
    if not par['selection']['tmin']:
        tmin = np.min(time) + 100 # Trim off 100 seconds before and after from the data.
    else:
        tmin = par['selection']['tmin']

    if not par['selection']['tmax']:
        tmax = np.max(time) - 100 # Trim off 100 seconds before and after from the data.
    else:
        tmax = par['selection']['tmin']

    ra = par['parameters']['RAJ']
    dec= par['parameters']['DECJ']
    ra, dec = transfer_hms2degree(ra, dec)

    command_gtselect = "gtselect infile={} outfile={} ra={} dec={} rad={} tmin={} tmax={} emin={} emax={} zmax={}".format(
            evfile, gtselect, str(ra), str(dec), str(rad), str(tmin), str(tmax), str(emin), str(emax), str(zmax))
    print("Executing --> {}".format(command_gtselect))
    os.system(command_gtselect)

    command_gtbary = "gtbary evfile={} scfile={} outfile={} ra={} dec={}".format(
            gtselect, scfile, gtbary, ra, dec)
    print("Executing --> {}".format(command_gtbary))
    os.system(command_gtbary)

def get_parlist(yamlfile):
    with open(yamlfile)as fin:
        parlist = yaml.load(fin, Loader=yaml.FullLoader)
    return parlist


@numba.njit
def cal_chisquare(data, f, pepoch, bin_profile, F1, F2, F3, F4):
    chi_square = np.zeros(len(f), dtype=np.float64)
    #chi_square = np.zeros(len(f))
    #chi_square = numba.float64(chi_square)

    for i in range(len(f)):
        phi = (data-pepoch)*f[i] + (1.0/2.0)*((data-pepoch)**2)*F1 + (1.0/6.0)*((data-pepoch)**3)*F2 +\
                (1.0/24.0)*((data-pepoch)**4)*F3 + (1.0/120.0)*((data-pepoch)**5)*F4
        phi = phi - np.floor(phi)
        counts  = numba_histogram(phi, bin_profile)[0]
        chi_square[i] = np.sum( (counts - np.mean(counts))**2 / np.mean(counts) )
    return chi_square

@numba.jit(nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@numba.jit(nopython=True)
def numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges

def cal_toa(fbest, profile, data):
    delta_phi = np.argmax(profile)
    toa = (1/fbest)*delta_phi + np.min(data)
    toa = (toa / 86400.0) + MJDREFI + MJDREFF
    #TODO:ToA error
    return toa



def fsearch(yamlfile, **kwargs):
    """
    search the best frequency

    Parameters :
    ---------------
    yamlfile: string
        The name of parameter YAML file 

    Returns :
    ---------------

    chi_square : array-like
        The Chi Square distribution of Epoch folding 

    fbest : float
        The most significant frequency
    """
    
    par = get_parlist(yamlfile)
    filename = par['data']['gtbary']
    print("Fsearch --> %s"%( filename ))

    hdulist = fits.open(filename)
    time = hdulist[1].data.field("TIME")
    
    MJDREFF = 0.00074287037037037
    MJDREFI = 51910

    #read parfile and parameters
    PEPOCH = par['parameters']['PEPOCH']
    F0     = par['parameters']['F0']
    F1     = par['parameters']['F1']
    if 'F2' in par['parameters']:
        F2 = par['parameters']['F2']
    else:
        F2 = 0
    if 'F3' in par['parameters']:
        F3 = par['parameters']['F3']
    else:
        F3 = 0
    if 'F4' in par['parameters']:
        F4 = par['parameters']['F4']
    else:
        F4 = 0
    frange = par['parameters']['FRANGE']
    fstep  = par['parameters']['FSTEP']

    pepoch = (PEPOCH - MJDREFF - MJDREFI)*86400


    data = time
    if len(data)==0:
        raise IOError("Error: Data is empty")
    t0 = np.min(data)
    bin_profile = 20

    f = np.arange(F0-frange,F0+frange,fstep)
    data = numba.float64(data)
    ## calculate chisquare   
    chi_square = cal_chisquare(data, f, pepoch, bin_profile, F1, F2, F3, F4)
    fbest = f[np.argmax(chi_square)]

    phi = (data-pepoch)*fbest + (1.0/2.0)*((data-pepoch)**2)*F1 + (1.0/6.0)*((data-pepoch)**3)*F2 +\
            (1.0/24.0)*((data-pepoch)**4)*F3 + (1.0/120.0)*((data-pepoch)**5)*F4
    phi = phi - np.floor(phi)
    counts, phase  = numba_histogram(phi, bin_profile)
    phase = phase[:-1]

    toa = cal_toa(fbest, counts, data)
    print("ToA --> {}".format(toa))

    if ("figure" in kwargs):
        if kwargs["figure"]:
            plt.figure("chisquare")
            plt.plot(f, chi_square)
            plt.figure("profile")
            plt.plot(phase, counts)
            plt.show()
        
    


if __name__ == "__main__" :
    yamlfile = "config_timing.yaml"
    download_data(yamlfile)
    fermi_gtanalyse(yamlfile)
    fsearch(yamlfile, figure=True)
