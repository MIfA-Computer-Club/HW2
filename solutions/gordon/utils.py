#! /usr/bin/env python
'''
Utility functions for centroiding/profiling
'''
import numpy as np
from astropy.modeling import models, fitting
import pyfits

fitter = fitting.LevMarLSQFitter()

def centroid_gaussian(data, coords, rad = 30, returnFit = False):
    '''Determine centroid given initial guess by fitting a 2D Gaussian'''
    if isinstance(data,str):
        data = pyfits.getdata(data)

    # Transpose x and y b/c reasons
    center_y,center_x = coords
    dslice = data[center_x-rad:center_x+rad,center_y-rad:center_y+rad]

    # Construct a grid of coordinates
    x,y = np.mgrid[0:dslice.shape[0],0:dslice.shape[1]]
    x -= dslice.shape[0]/2.
    y -= dslice.shape[1]/2.
                
    p_init = models.Gaussian2D(np.max(dslice),0,0,rad,rad)
    p = fitter(p_init,x,y,dslice)
    
    # Rescale coordinates to match data
    p.x_mean = center_y - p.x_mean
    p.y_mean = center_x - p.y_mean

    if returnFit:
        return p.x_mean.value, p.y_mean.value, p
    
    else:
        return p.x_mean.value, p.y_mean.value


def centroid_airy(data, coords, rad = 30, returnFit = False):
    '''Determine centroid given initial guess by fitting a 2D AiryDisk'''
    if isinstance(data,str):
        data = pyfits.getdata(data)

    # Transpose x and y b/c reasons
    center_y,center_x = coords
    dslice = data[center_x-rad:center_x+rad,center_y-rad:center_y+rad]

    # Construct a grid of coordinates
    x,y = np.mgrid[0:dslice.shape[0],0:dslice.shape[1]]
    x -= dslice.shape[0]/2.
    y -= dslice.shape[1]/2.
                
    p_init = models.AiryDisk2D(np.max(dslice),0,0,rad)
    p = fitter(p_init,x,y,dslice)

    # Rescale coordinates to match data
    px = center_y - p.x_0
    py = center_x - p.y_0

    if returnFit:
        return px, py, p
    
    else:
        return px, py


def radialprofile(data, center=None, binsize=1, interpnan = True):
    '''Construct azimuthal average radial profile.
    [based on Ginsburg method]
    '''
    
    if isinstance(data, str):
        data = pyfits.getdata(data)

    # get indices of data
    y, x = np.indices(data.shape)

    # if center not specified, assume image is centroided
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    # construct matrix of distances from center pixels    
    r = np.hypot(x - center[0], y - center[1])

    # generate bins spanning the entire array
    nbins = int(np.round(r.max() / binsize) + 1)
    maxbins = nbins * binsize
    bins = np.linspace(0,maxbins,nbins+1)
    # calculate bin centers for later
    bin_centers = (bins[1:] + bins[:-1])/2.0

    # use histogram to count the number of pixels a distance r from the center,
    #  weighing by flux counts.
    radial_prof = np.histogram(r,bins, weights = data)[0] / np.histogram(r,bins, weights=np.ones(data.shape))[0]

    if interpnan:
        # interpolate by using only 'real' bins
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof])    

    return (bin_centers, radial_prof)


def sector_mask(shape,center,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = center
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def sectorprofile(data, center=None, rad=15, binsize=30, interpnan = True):
    '''
    Construct a 'sector profile' by summing up angle bins.
    '''
    if isinstance(data, str):
        data = pyfits.getdata(data)

    # generate indices
    y, x = np.indices(data.shape)
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    
    sector_prof = []
    bin_centers = []
    for start,stop in zip(np.arange(0,361,binsize),np.arange(binsize,361+binsize,binsize)):
        # mask out each theta bin
        mask = sector_mask(data.shape,center,radius=rad,angle_range= (start,stop))
        sector = np.ma.MaskedArray(data,~mask)
        # average counts in bin
        sector_prof.append(sector.sum()/sector.count())
        bin_centers.append(np.mean([start,stop]))

    
    return (bin_centers, sector_prof)
