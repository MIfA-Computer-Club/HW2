#! /usr/bin/env python
'''
Utility functions for centroiding/profiling
'''
import numpy as np
from astropy.modeling import models, fitting
import pyfits
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.optimize import curve_fit

fitter = fitting.LevMarLSQFitter()

def nanrms(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))

def find_nearest_element(array,value,index=False):
    """Return closest value in array, or its index"""
    idx = np.abs(array-value).argmin()
    return (idx,array.flat[idx]) if index else array.flat[idx]

def exp_Re(x,y):
    '''Fit exponential profile and return Re'''
    p,pcov = curve_fit(exp_prof,x,y,p0=[np.mean(x),np.max(y),np.min(y)])
    return p[0]*1.67835  #exact solution for exponential profile

def H(x):
    '''Heaviside step function'''
    return 0.5 * (np.sign(x) + 1)

exp_prof = lambda r,rd,Id,c: Id*np.exp(-r/rd)+c
sersic = lambda x,n,Ie,b,re,c: Ie*np.exp(-b*(x/re)**(1/n)-1) + c

def sersic_core(x,n,Ib,b,re,c,rb,gam):
    '''Trujillo et al (2004)'''
    return Ib*((rb/x)**gam*H(rb-x)+np.exp(b*(rb/re)**(1/n))*np.exp(-b*(x/re)**(1/n))*H(x-rb))
    

def sersic_fit(x,y,rebin=None):
    p0 = [2,np.max(y),1,np.mean(x),np.min(y),np.mean(x),1]
    try:
        p,pcov = curve_fit(sersic_core,x,y,p0=p0)
        if rebin:
            x = np.linspace(np.min(x),np.max(x),num=rebin)
        return (x,sersic_core(x,*p),p)
    
    except: # try one component
        p,pcov = curve_fit(sersic,x,y,p0=p0[0:5])
        if rebin:
            x = np.linspace(np.min(x),np.max(x),num=rebin)
        return (x,sersic(x,*p),p)
        


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

    plt.imshow(data)
        
    # get indices of data
    y, x = np.indices(data.shape)

    # if center not specified, assume image is centroided
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-ymin())/2.0])

    # construct matrix of distances from center pixels    
    r = np.hypot(x - center[0], y - center[1])

    # generate bins spanning the entire array
    nbins = int(np.round(r.max() / binsize) + 1)
    maxbins = nbins * binsize
    bins = np.linspace(0,maxbins,nbins+1)
    # calculate bin centers for later
    bin_centers = (bins[1:] + bins[:-1])/2.0

    # use histogram to count the number of pixels a distance r from the center,
    #  weighting by flux counts.
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


class CircularAperture(object):

    def __init__(self, xycenter, data, radius, skyradius):
        self.xc, self.yc = xycenter
        self.data = data
        self.radius = radius
        self.skyradius = skyradius

        self.aper = self.make_aperture(self.radius)
        self.skyaper = self.make_aperture(self.radius,self.skyradius)


    def make_aperture(self,radius,radius2=None):
        x,y = np.ogrid[:self.data.shape[0],:self.data.shape[1]]
        cx,cy = (self.xc,self.yc)

        # circular mask
        r = np.hypot(x-cx,y-cy)

        if not radius2:
            circmask = r <= radius - 0.5

        # pixels entirely within aperture have weighting of 1
        #weights = np.ones(self.data.shape) * circmask.astype('float')

        else:
            circmask = (r <= radius2 - 0.5) & (r >= radius + 0.5)
            
        return circmask.astype('float')
    
    @staticmethod
    def num_pix(weights):
        return np.sum(weights)    

    def run(self,longTable=False):
        '''Return:
        (skysub_aper_counts, aper_area, total_aper_counts,
         sky_area, total_sky_counts, med_sky_counts, std_sky_counts)
        '''
        aper = self.aper * self.data
        sky = self.skyaper * self.data
        aper_area = self.radius*self.radius*np.pi
        sky_area = self.skyradius**2*np.pi - aper_area

        total_aper_counts = np.sum(aper)
        total_sky_counts = np.sum(sky)
        med_sky_counts = np.median(sky)
        skysub_aper_counts = total_aper_counts - med_sky_counts*aper_area
        std_sky_counts = np.std(sky)/np.sqrt(self.num_pix(self.skyaper))

        counts_per_pix2 = skysub_aper_counts / aper_area
        
        if longTable:
            return {'counts/pix^2':counts_per_pix2,
                    'skysub_aper_counts':skysub_aper_counts,
                    'aper_area':aper_area,
                    'total_aper_counts':total_aper_counts,
                    'sky_area':sky_area,
                    'total_sky_counts':total_sky_counts,
                    'med_sky_counts':med_sky_counts,
                    'std_sky_counts':std_sky_counts}

        else:
            return {'counts/pix^2':counts_per_pix2,
                    'err_cp2':std_sky_counts,
                    '#pix':aper_area}
        

        
        
def contour_profile(data,num_levels=20,get_levels=False):
    # make contours
    plt.figure()
    c = plt.contour(data)
    # get default levels
    levels = c.levels

    # rebin levels
    levels = np.linspace(np.min(levels),np.max(levels),num=num_levels)

    plt.clf()
    # Remake contours with binned levels
    c = plt.contour(data,levels=levels)

    masks = []
    for p in c.collections:
        path = p.get_paths()
        path = Path(path[0].vertices,closed=True) # close path
        # get all points in path
        mask = np.array([False if path.contains_point([x,y]) else True for x,y in np.ndindex(data.shape)])
        mask = mask.reshape(data.shape).T
        masks.append(mask)
    plt.close()

    # get indices for radial binning
    cx,cy = (data.shape[0]/2,data.shape[1]/2)
    x,y = np.ogrid[:data.shape[0],:data.shape[1]]
    rad = np.hypot(x-cx,y-cy)
        
    r = []
    med = []
    err = []
    for idx in range(0,len(masks)-1):
        maskA = masks[idx]
        maskB = masks[idx+1]
        # get area between     not (outsideA and inside B)
        brokeassbooleanlogic = ~(~maskA & maskB)
        annulus = np.ma.MaskedArray(data,brokeassbooleanlogic)

        # choose mean value for annulus
        med.append(np.float(np.ma.mean(annulus)))
        err.append(np.float(np.ma.std(annulus)))

        # choose mean radius of annulus
        ###  do geometric mean radius based on ellipticity?
        maskrad = np.ma.MaskedArray(rad,brokeassbooleanlogic)
        r.append(np.float(np.ma.mean(maskrad)))

    # get smallest annulus
    annulus = np.ma.MaskedArray(data,masks[-1])
    maskrad = np.ma.MaskedArray(rad,masks[-1])
    med.append(np.float(np.ma.mean(annulus)))
    err.append(np.float(np.ma.std(annulus)))
    r.append(np.float(np.ma.mean(maskrad)))

    # reverse them all to go from smallest to largest radius
    r.reverse()
    med.reverse()
    err.reverse()

    if get_levels:
        return (r,med,err,levels)
    else:
        return (r,med,err)
