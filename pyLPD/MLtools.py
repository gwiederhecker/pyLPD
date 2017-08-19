# coding: utf-8

# # Load matlab structure files (with substrctures)
# The purpose is to be able to access matlab structures and substructures in a easy fashion.
# This code comes from the post: https://stackoverflow.com/a/29126361/7938052

'''
MLtools is a package from pyLPD intented to recreate some useful Matlab functions in Python.
'''



import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
from scipy.interpolate import interp1d
from math import ceil
import numpy as np
from math import factorial    
import scipy.io as spio


def loadmat(filename):
    '''
    The purpose is to be able to access matlab structures and substructures in a easy fashion. 
    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    Parameters
    ---------
    filename : string
        string containing the full path of the .MAT file.

    Notes
    -----
    This code comes from the post: https://stackoverflow.com/a/29126361/7938052  
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem
    return dict

def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays 
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []            
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list


def peakdet(v, delta, x = None):
    """
    Detects maxima and minima based on variation of neighboring points.

    Parameters
    -------
    v : array-like, shape (N,)
        containing y-values
    delta : double
        A point is considered a maximum peak if it has the maximal
        value, and was preceded (to the left) by a value lower by
        delta.         
    x : array-like, shape (N,)  
        containing x-values. When provided, x-values are returned instead of indexes.

    Returns
    -------
    maxtab: 2D, Nx2
        indices ([:,1]) and values ([:,2]) of maxima 
    mintab: 2D, Nx2
        indices ([:,1]) and values ([:,2]) of minima
    
    Notes
    -----
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    This function is released to the public domain; Any use is allowed.
    
    Examples
    ------
    >>> import matplotlib.pyplot as plt 
    >>> series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
    >>> maxtab, mintab = peakdet(series,.3)
    >>> plt.plot(series)
    >>> plt.scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
    >>> plt.scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
    >>> plt.show()      
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
                
    ind_max = array(maxtab)[:,0].astype(int);
    ind_min = array(mintab)[:,0].astype(int);

    return ind_max, array(maxtab), ind_min, array(mintab)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = savitzky_golay(y, window_size=31, order=4)
    >>> plt.plot(t, y, label='Noisy signal')
    >>> plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> plt.legend()
    >>> plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except(ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def envPeak(x, delta=0.2, window_size = 0.05, sg_order = 0, interp_kind='slinear'):
    '''
    Find envelope of dataset based on interpolation of maxima and minima.

    Parameters
    ----------
    x: array-like, shape (N,)
        signal whose envelope is to be determined
    delta : double from(0,1)
        A point is considered a maximum peak if it has the maximal
        value, and was preceded (to the left) by a value lower by
        delta. From peakdet function.
    window_size : double from (0,1)
        the length of the window (1 correponds to full vector length).
        From Savitzy_Golay function
    sg_order:  int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1. From Savitzy_Golay function
    interp_kind: string ('slinear','quadratic','cubic')
        interpolation kind, from interp1 (scipy)

    Returns
    -------
    ylower: array-lie, shape (N,).
        Lower envelope.
    yupper: array-like, shape (N,).
        Upper envelope.

    Notes
    -----
    This function was adapted from Matlab's Signal Processing toolbox. 

    Examples
    --------
    >>> import pyLPD.MLtools as mlt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> %matplotlib inline
    >>> #generate sinusoid
    >>> x = np.linspace(-10,10,1e4)
    >>> y = np.sin(2*np.pi*x)*np.exp(-x**2/10)
    >>> ylower,yupper = mlt.envPeak(y,delta=0.05, sg_order=0)
    >>> plt.plot(x,y,'k')
    >>> plt.plot(x,yupper,'.r')
    >>> plt.plot(x,ylower,'.b')
    '''

    #----------------------------------------------------   
    # pre-allocate space for results
    nx = np.size(x)
    yupper = np.zeros(nx,dtype=x.dtype)
    ylower = np.zeros(nx,dtype=x.dtype)
    #----------------------------------------------------   
    #-----
    #round to next event nunmber (required fro Savitzky_Golay)
    # handle default case where not enough input is given
    if nx < 2:
        yupper = x
        ylower = x
        return
    # find local maxima separated by at least N samples
    iPk_max, maxtab, iPk_min, mintab = peakdet(x,delta) 
    #upper
    iLocs_max = np.append([0],iPk_max)
    iLocs_max = np.append(iLocs_max,nx-1)
    #lower
    iLocs_min = np.append([0],iPk_min)
    iLocs_min = np.append(iLocs_min,nx-1)

    # smoothly connect the maxima via a spline.
    #------------------
    nsmooth = ceil(len(x[iLocs_max])*window_size/ 2.) * 2+1
    xs = savitzky_golay(x[iLocs_max],nsmooth,order=sg_order)
    yupper_ifunc = interp1d(iLocs_max,xs,kind=interp_kind)
    yupper = yupper_ifunc(range(0,nx))   
    #yupper = savitzky_golay(yupper,nsmooth,order=sg_order)
    #------------------
    nsmooth = ceil(len(x[iLocs_min])*windows_size/ 2.) * 2+1
    xs = savitzky_golay(x[iLocs_min],nsmooth,order=sg_order)
    ylower_ifunc = interp1d(iLocs_min,x[iLocs_min],kind=interp_kind)
    ylower = ylower_ifunc(range(0,nx)) 
    ylower = savitzky_golay(ylower,nsmooth,order=sg_order)
    #-----------------
    return ylower,yupper





