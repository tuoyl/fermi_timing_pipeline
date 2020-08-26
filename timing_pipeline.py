import yaml
import numpy as np
import numba 
from astropy.io import fits
import matplotlib.pyplot as plt
import wget
import os

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
        The 




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
#    fsearch(yamlfile, figure=True)

#    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
#            description='Example: python he_pipeline.py -i hxmt_filename -p outprofile.dat -c outchisquare.dat')
#    parser.add_argument("-i","--input",help="input filename of HXMT screen file")
#    parser.add_argument("-p","--profile",help="out profile name")
#    parser.add_argument("-c","--chisquare",help="chisquare distribution file",type=str)
#    parser.add_argument("-f0","--freqency", help="f0 for searching frequency", type=float)
#    parser.add_argument("-f1","--freqderive", help="f1 for searching frequency", type=float)
#    parser.add_argument("-f1step",help="frequency derivative intervals for frequency search", type=float)
#    parser.add_argument("-f1range", help="frequency derivative range for searching", type=float)
#    parser.add_argument("-f2","--freqsecderive", help="f2 for searching frequency", type=float)
#    parser.add_argument("-f3","--freqthirdderive", help="f3 for searching frequency", type=float)
#    parser.add_argument("-fstep",help="frequency intervals for frequency search", type=float)
#    parser.add_argument("-frange", help="frequency range for searching", type=float)
#    parser.add_argument("-epoch", help="epoch time for fsearch (MJD)", type=float)
#    parser.add_argument("-bins", help="profile bin numbers", type=int)
#    args = parser.parse_args()
#    filename = args.input
#    outprofile = args.profile
#    outchisq   = args.chisquare
#    f0 = args.freqency
#    fstep = args.fstep
#    frange = args.frange
#    epoch  = args.epoch
#    binprofile = args.bins
#    if args.freqderive:
#        freqderive = args.freqderive
#    else:freqderive = 0
#    if args.freqsecderive:
#        freqsecderive = args.freqsecderive
#    else:freqsecderive = 0
#    if args.freqthirdderive:
#        freqthirdderive = args.freqthirdderive
#    else:freqthirdderive = 0
#
#    if args.f1range:
#        print("DO THIS")
#        f1range = args.f1range
#        f1step  = args.f1step
#        fsearch(filename, outprofile, outchisq, f0, fstep, frange, epoch, binprofile, f1=freqderive, f2=freqsecderive, f3=freqthirdderive, f1step=f1step, f1range=f1range)
#    else:
#        fsearch(filename, outprofile, outchisq, f0, fstep, frange, epoch, binprofile, f1=freqderive, f2=freqsecderive, f3=freqthirdderive)

