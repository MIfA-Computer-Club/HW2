#! /usr/bin/env python
import pyregion
import argparse
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import utils
from astropy.table import Table, Column
from matplotlib.backends.backend_pdf import PdfPages

class Galaxy(object):
    
    def __init__(self, name, initial_coords, data, rad = 20):
        self.name = name
        self.initial_x, self.initial_y = initial_coords
        self.rad = rad

        self.x, self.y = utils.centroid_gaussian(data,initial_coords,rad)
        
        self.swatch = self.get_swatch(data)
        self.xc,self.yc = (self.swatch.shape[0]/2,self.swatch.shape[1]/2)
        
    def get_swatch(self,data):
        # Crop slice bigger than specified radius
        x,y = self.initial_y, self.initial_x
        size = int(self.rad*2)
        dslice = data[x-size:x+size,y-size:y+size]
               
        xs = self.y - x
        ys = self.x - y

        # Shift to center of object
        nslice = shift(dslice,[ys,xs]) # reverse x/y cuz scipy
        
        # Resize to rad x rad
        nslice = nslice[size-self.rad:size+self.rad,
                        size-self.rad:size+self.rad]

        return nslice
    
    def show(self):
        fig = plt.figure()
        ax = plt.imshow(self.swatch,origin='lower',cmap=plt.cm.gist_heat_r)
        plt.title(self.name)
        return (fig,ax)
        

def main():
    parser = argparse.ArgumentParser(description='Perform centroiding and profiling of input galaxies.')

    parser.add_argument('data',type=str,help='Image file')
    parser.add_argument('region',type=str,help='Region file')

    # Parse dat ass
    args = parser.parse_args()

    # Parse coordinates
    regions = pyregion.open(args.region)
    names = [r.comment for r in regions]
    coords = [(int(r.coord_list[0]),int(r.coord_list[1])) for r in regions]

    # Grab those juicy data meats
    data,header = fits.getdata(args.data,header=True)

    # For each coord, initialize and centroid galaxy
    galaxies = [Galaxy(name,co,data) for co,name in zip(coords,names)]

    # Show each swatch
    pp = PdfPages('galaxy_im.pdf')
    for gal in galaxies:
        pp.savefig(gal.show()[0])
    #plt.show()
    pp.close()

    # Perform aperture photometry
    rows = []
    for gal in galaxies:
        ap = utils.CircularAperture((gal.xc,gal.yc), gal.swatch, 15,19)
        rows.append(ap.run())

    t = Table(rows=rows)
    t.add_column(Column([gal.name for gal in galaxies],name='name'),index=0)

    # Convert to counts/arcsec^2
    pltscale = header['pltscale']    # "/mm
    pltscale = pltscale /1000        # "/um
    pixscaleX = header['xpixelsz']   # um/pix
    pixscaleX = pixscaleX * pltscale # "/pix
    pixscaleY = header['ypixelsz']   # um/pix
    pixscaleY = pixscaleY * pltscale # "/pix
    pixscale = pixscaleX * pixscaleY # arcsec^2/pix^2
    cpa2 = t['counts/pix^2'] / pixscale
    epa2 = t['err_cp2'] * pixscale
    
    cpa2 = Column(cpa2,name='counts/arcsec^2')
    epa2 = Column(epa2,name='err_ca2')
    t.add_columns([cpa2,epa2])

    t.pprint()

    '''
name       #pix      counts/pix^2    err_cp2    counts/arcsec^2    err_ca2   
------- ------------- ------------- ------------- --------------- -------------
NGC4875 706.858347058 7721.59941623 117.896411641   7584.60432695 120.025887186
NGC4869 706.858347058 11151.8722709 173.611128029   10954.0179592 176.746937224
GMP4277 706.858347058 9756.15839964 139.752361268   9583.06656737 142.276604641
GMP4350 706.858347058 8855.62011973 130.378485589   8698.50545946  132.73341559
NGC4860 706.858347058 11748.7754577 157.304664726   11540.3310078 160.145942354
NGC4881 706.858347058 11499.1497714 161.710512127   11295.1341311 164.631369312
NGC4921 706.858347058  12822.257865 215.253917975   12594.7679026 219.141890035
    '''

    # prep PDF
    pp = PdfPages('galaxy_prof.pdf')
    pp2= PdfPages('galaxy_contour.pdf')

    contours = []
    for gal in galaxies:
        # perform contour isophotal fitting for profile
        r,med,err,levels = utils.contour_profile(gal.swatch,get_levels=True)
        contours.append(levels)

        fig = plt.figure()

        # plot profile with errorbars
        plt.errorbar(np.array(r)*pixscaleX,med,yerr=err,fmt='o',ms=8)

        # fit sersic function
        try:
            # hard-coded bullshit!!
            if gal.name == 'NGC4860':
                xi,yi = r[:-2],med[:-2]
            elif gal.name == 'GMP4350':
                xi,yi = r[1:],med[1:]
            elif gal.name == 'NGC4869':
                xi,yi = r[:-4],med[:-4]
            else:
                xi,yi = r,med 
            xf,yf,p = utils.sersic_fit(xi,yi,rebin=100)
        except:
            plt.legend([gal.name])
            continue

        # plot fit
        plt.plot(xf*pixscaleX,yf,'r-')
        plt.ylabel('Counts')
        plt.xlabel('Radius [arcsec]')
        #plt.xlim([0,12])
        #plt.legend([gal.name,'n = %.2f' % p[0]])

        # plot Re as just half max
        Ie = yf.max()/2
        ReIdx,Ie = utils.find_nearest_element(yf,Ie,index=True)
        Re = xf[ReIdx]*pixscaleX
        # calculate Re from exp prof?
        #Re = utils.exp_Re(xf*pixscaleX,yf)
        plt.legend([gal.name,'n = %.2f\nRe = %.2f' % (p[0],Re)])
        plt.plot([np.min(xf)*pixscaleX,Re],[Ie,Ie],'r-.')

        pp.savefig(fig)


    # plot image and contours
    for gal in galaxies:
        fig = plt.figure()
        plt.imshow(gal.swatch,origin='lower',cmap=plt.cm.gist_heat_r)
        plt.contour(gal.swatch,levels=levels,alpha=0.7)
        plt.title(gal.name)


        pp2.savefig(fig)        
        
        

    #plt.show()
    pp.close()
    pp2.close()

    exit()
        

    '''    
        plt.figure()
        c = plt.contour(gal.swatch)
        levels = c.levels
        levels = np.linspace(np.min(levels),np.max(levels),num=20)
        #print levels

        plt.clf()
        c = plt.contour(gal.swatch,levels=levels)
        p = c.collections[0].get_paths()
        print Path(p[0].vertices,closed=True)
        
        
        #plt.show()
        exit()
        
        bins, prof = utils.radialprofile(gal.swatch,binsize=0.5)
        #radbins = np.sqrt(pixscale)*bins
        #rof_arcsec = prof / pixscale
        plt.figure()
        plt.plot(bins,prof,'bo')
        plt.title(gal.name)
        plt.show()
        exit()
    plt.show()
    '''








    
    
'''
    # Make a sector profile to determine if another source is present
    binsize = 15
    for gal in galaxies:
        theta,flux = utils.sectorprofile(gal.swatch,binsize=binsize,rad=np.max(gal.swatch.shape))
        # normalize around 1
        flux = flux - np.median(flux)

        # If sector deviates by more than 3 sigma, there is another source
        #  present. So, mask out that region
        dsflux = flux/np.std(flux)
        if np.max(dsflux) > 3.0:
            badsector = theta[np.argmax(dsflux)]
            badrange = [badsector-binsize,badsector+binsize]
            # get mask of badsector
            mask = utils.sector_mask(shape=gal.swatch.shape,
                                     center=[gal.xc,gal.yc],
                                     radius=25,
                                     angle_range=badrange)
            # apply mask
            badmap = np.ma.MaskedArray(gal.swatch,~mask)
            #plt.imshow(badmap,origin='lower')

            # make radial profile of sector to find other peak
            secx,secy = utils.radialprofile(badmap.filled())
            # initial center of other gal
            rad = secx[np.argmax(secy)]
            print rad
            print badsector
            xc = rad * np.cos(np.deg2rad(badsector)) + gal.swatch.shape[0]/2
            yc = rad * np.sin(np.deg2rad(badsector)) + gal.swatch.shape[1]/2

            #plt.imshow(gal.swatch,origin='lower')
            print xc,yc
            #plt.show()
            #exit()
            #centroid on these coords
            bx,by,bp = utils.centroid_gaussian(gal.swatch,[xc,yc],rad=5,
                                               returnFit=True)
            print bx,by,bp
'''




if __name__ == '__main__':
    main()
