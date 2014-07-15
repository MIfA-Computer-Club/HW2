#! /usr/bin/env python
import argparse
import numpy as np
import pyfits
import pyregion
try: 
    from astropy.modeling import models,fitting
except:
    print 'astropy.modeling module not installed '
import matplotlib.pyplot as plt

def gauss_centroid(im, x_init, y_init):
    """Find centroid of source by fitting a 2D Gaussian"""
    fitter = fitting.LevMarLSQFitter()
    size = 250
    
    # get initial guesses for the parameters
    amplitude = np.max( im[y_init-size:y_init+size, x_init-size:x_init+size] )
    sig_x = 10.
    sig_y = 10.

    g_init = models.Gaussian2D(amplitude, x_init, y_init, sig_x, sig_y)

    # get indices of subsection of image
    x,y = np.mgrid[x_init-size:x_init+size, y_init-size:y_init+size]
    
    p = fitter(g_init, x, y, im[x_init-size:x_init+size, 
        y_init-size:y_init+size] )

    cenx = p.x_mean.value
    ceny = p.y_mean.value

    return cenx,ceny


def main():
    parser = argparse.ArgumentParser(
        description='Perform aperture photometry on a list of coordinates')
    parser.add_argument('filename', type=str, nargs=1,
        help='Input FITS image. If ')
    parser.add_argument('-c', '--coordfile', nargs=1, 
        help='If specified, use coords from coordfile. Otherwise find ' +\
             'coordinates by centroiding objects identified in ds9 region ' +\
             'file. coordfile lists x,y with 1 source per line')
    args = parser.parse_args()

    # read in image
    image,hdr = pyfits.getdata(args.filename[0], header=True)
    
    # size of output boxes around galaxies
    size = 50

    if args.coordfile:
        data = np.genfromtxt(args.coordfile[0], 
            dtype=[('gal', 'S10'), ('x', 'f8'), ('y', 'f8')])
        galaxy = data['gal']
        cenx = data['x']
        ceny = data['y']
            
    else:
        # read in region file
        region = args.filename[0].replace('.fits', '.reg')
        r = pyregion.open(region)

        # centroid sources
        cenx = np.zeros( (len(r),4), dtype=float)
        ceny = np.zeros( (len(r),4), dtype=float)
        for i,gal in enumerate(r):
            print 'Centroid for ' + gal.comment
            x_init = gal.coord_list[0]
            y_init = gal.coord_list[1]

            # make small cutouts around each galaxy
            im = image[y_init-size:y_init+size, x_init-size:x_init+size]

            # get indices of subsection of image
            x,y = np.mgrid[x_init-size:x_init+size, y_init-size:y_init+size]

            # using 2D Gaussian from astropy.modeling
            try:
                cenx[i,0],ceny[i,0] = gauss_centroid(im, x_init, y_init)
            except:
                print '   astropy.modeling not installed, ' +\
                    'skipping Gaussian centroid'

            # using the pixel with the max counts
            #print x[im == np.max(im)]
            #print y[im == np.max(im)]
            #print x_init, y_init
            
            # display swatch of galaxies 
            # calculate stddev and mean of image swatch for z sort of z-scaling
            sig = np.std(im[im != 0])
            mean = np.mean(im[im != 0])
            vmin = mean - 2.5 * sig
            vmax = mean + 2.5 * sig
            plt.figure()
            plt.imshow(image, cmap='gray_r', vmin=vmin, vmax=vmax, 
                aspect='auto', origin='lower')
            plt.xlim([x_init-size, x_init+size])
            plt.ylim([y_init-size, y_init+size])
            plt.autoscale(False)

            # plot the center positions from the various centroid methods
            plt.scatter(x[im == np.max(im)], y[im == np.max(im)], marker='x', 
                s=100, c='b')

            # using the center-of-mass definition
            xarr = np.sum(x * im, axis=0)
            yarr = np.sum(y * im, axis=1)
            xc = x[np.argmax(xarr)]
            yc = y[np.argmax(yarr)]

            # using where the derivatives go to zero
        plt.show()


if __name__ == '__main__':
    main()
