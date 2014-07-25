from astropy.io import fits
import matplotlib.pyplot as p
import matplotlib.cm as cm
import numpy as n
import pyregion
from scipy import optimize

def ap_flux(image, pixel_scale, radius):
	"""
	  PURPOSE: Calculate the flux in a circular area
	           centered on the maximum of an image 

	  ARGUMENTS:
	    image: 2d array of image
	    pixel_scale: [arcsec/pixel]
	    radius: radius of circle in arcsec

	  RETURNS: integrated flux per arcsec^2, flux per pixel, stddev of flux
  	"""
  	#find the location of the brightest pixel in the image
	x_cen = n.argmax(n.max(image,axis=1))
	y_cen = n.argmax(n.max(image,axis=0))
	mask = n.copy(image)*0.
	#convert arcsec to pixels
	r_pix = radius/pixel_scale
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if n.sqrt((i-x_cen)**2+(j-y_cen)**2) <= r_pix:
				mask[i,j] = 1.
	sq_arcsec = (n.sum(mask)*pixel_scale)**2
	flux_pa = n.sum(image*mask)/sq_arcsec
	flux_pp = n.sum(image*mask)/n.sum(mask)
	return flux_pa, flux_pp, n.std(image*mask)

def sersic_fit(r,I):
  """
  PURPOSE: Fit a sersic profile to data

  ARGUMENTS:
    x, y: numpy ndarrays with (hopefully) matching sizes

  RETURNS: best fit parameters
  """

  sersicfunc = lambda r,Ie,ns,Re: Ie*n.exp(-(1.9992*ns - 0.3271)*((r/Re)**(1/ns)-1.0))
  #*NOTE: according to Capaccioli 1991 if 0.5 < ns < 10 bns = 1.9992*ns - 0.3271
  #p0 = [Ie,ns,Re]
  popt,pcov = optimize.curve_fit(sersicfunc,r,I,p0=[n.max(I),1.,.5*n.max(r)])
  
  #overplot the fit with the data
  
  rfit = n.arange(n.min(r),n.max(r),0.1)
  Ifit = n.copy(rfit)*0.0
  Ifit = popt[0]*n.exp(-(1.9992*popt[1] - 0.3271)*((rfit/popt[2])**(1/popt[1])-1.0))
  #p.plot(r,I)
  p.plot(rfit,Ifit,color='red')
  #p.show()
  p.title("Opt_n = %.3f" % popt[1])
  return popt

def main():
	#read in the image and header
	img_file = 'POSIIF_Coma.fits'
	hdulist = fits.open(img_file)
	img = hdulist[0].data
	plate_scale = hdulist[0].header['pltscale'] #arcsec/mm
	x_pix_scale = hdulist[0].header['xpixelsz'] #um/pix
	y_pix_scale = hdulist[0].header['ypixelsz'] #um/pix
	x_pix = hdulist[0].header['naxis1']
	y_pix = hdulist[0].header['naxis2']
	plate_size = hdulist[0].header['pltsizex']#mm same for x and y
	hdulist.close()
	x_pixel_scale = x_pix_scale*(plate_size/1000) #arcsec/pixel
	y_pixel_scale = y_pix_scale*(plate_size/1000) #arcsec/pixel
	#load the region file
	reg_file = "POSIIF_Coma_new.reg"
	r = pyregion.open(reg_file)
	n_regions = n.size(r)
	#create a figure with all the galaxy postage stamps
	fig = p.figure()
	for i in range(n_regions):
		coord = r[i].coord_list
		diam = coord[2]
		region = img[coord[1]-diam:coord[1]+diam,coord[0]-diam:coord[0]+diam]
		#pick a sky region to the lower left
		sky = img[coord[1]-2.*diam:coord[1]-2.*diam+10,coord[0]-2.*diam:coord[0]-2.*diam+10]
		#subtract out the average sky background
		region -= n.median(sky)
		fig.add_subplot(2,7,(i+1))
		p.imshow(region, cmap=cm.spectral, origin="lower")
		#find the flux per pixel (pp) and per arcsec^2 (pa) 
		#for simplicity I'll assume pixels are square
		flux_arcsec, flux_pixel, flux_std = ap_flux(region, x_pixel_scale, diam*x_pixel_scale)
		print "Galaxy[%i]: " %i 
		print "Flux per pixel = ", flux_pixel
		print "Flux per arcsec^2 = ", flux_arcsec
		#evalute the flux as a function of radius
		radii = n.arange((1.*x_pixel_scale),(coord[2]*x_pixel_scale),(3.*x_pixel_scale))
		flux_r_pp,flux_r_pa,flux_r_std = n.copy(radii)*0.,n.copy(radii)*0.,n.copy(radii)*0.
		for j in range(radii.size):
			flux_r_pp[j],flux_r_pa[j],flux_r_std[j] = ap_flux(region, x_pixel_scale, radii[j])
		fig.add_subplot(2,7,(i+8))
		p.errorbar(radii,flux_r_pa,yerr=flux_r_std,fmt="o")
		opt_params = sersic_fit(radii, flux_r_pa)
		print "Best fit sersic =  ", opt_params[1]
		print "Half light radius ["u"\u0022""] = ", opt_params[2]
	fig.savefig('galaxy_stamps')
	p.show(fig)



if __name__ == '__main__':
    main()