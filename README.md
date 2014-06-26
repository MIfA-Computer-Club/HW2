Astro Computer Club 2014
========================

Assignment 2, Due ~Friday, July 11
----------------------------------

### Aperture Photometry, Radial Profiles

In this assignment, you will be designing your own aperture photometry packages.  I have included a Palomar Sky Survey image from the STScI archives.  The image is focused on the Coma Cluster of galaxies.  With the included ```ds9``` region file, you will be perfoming various tasks on some of the galaxies in the cluster.  There are many ways to do the following, and you are welcome to use tools outside of ```python``` (except ```IRAF```:  I explicitly forbid you from using ```IRAF```).

  If you have never touched ```ds9``` before, now is the time to play around.

------
1) Load the ```FITS``` file into ```ds9``` with the included region file to visualize which galaxies I have selected for this task.  You will notice that the region file contains image coordinates, not ```RA``` and ```DEC``` (because I do not trust the astrometry/```WCS``` from the STScI archive).  However, the coordinates in the region do not necessarily line up with the centers of the objects, so be careful.
  * The coordinates in the region file are not necessarily at the centers of the galaxies.  You should boot up ```ds9```, determine the centers of the objects, and then write a new region file.  OR...
  * Bonus: Write your own centroiding package.  This can be as simple as finding the maximum pixel in a box, as complicated as least-asymmetric Gaussian fitting, or anything in between.
  * Read the ```FITS``` file and the coordinates into ```python``` (you are welcome to use pre-existing packages for the latter, like ```pyregions```), but do not hard-code anything.
  * Use ```matplotlib.pyplot``` to display a 'swatch' of each galaxy.  Warning:  if using ```imshow```, be sure to set ```origin = 'lower'```.  ```ds9``` indexes 2D-arrays from the bottom-left, but ```numpy``` goes from the top-left.  When displaying your swatches, make sure the images are oriented the same as they are in ```ds9``` (and not transposed).

2) For each galaxy, perform basic aperture photometry.
  * Choose a circular aperture and calculate flux counts per aperture area (```counts/pixel^2```).  Subtract from this number the background counts (sky flux), which you can calculate by choosing a region of the sky next to the source.  Ordinarily, an annulus around the object is chosen for the sky region, but some of these galaxies are right next to other sources, so be careful.  Whatever way you do it, be sure to 
  * Using the plate scale, pixel scale, and pixel size keywords from the header, translate your flux counts into ```counts/arcsec^2```.  (We will not worry about flux to magnitude conversion yet, because I cannot find a reliable zeropoint for this instrument/filter).

3) Make a radial, azimuthally-averaged, profile of each galaxy.
  * Choose an appropriate size for your circular bins, and plot ```counts/arcsec^2``` vs. radial distance from the center of the source (in ```arcsec```).  For each bin, calculate (and plot on your profile) the error as the standard deviation within the bin.
  * Bonus:   Drop elliptical bins instead for a more accurate profile.
  * Bonus+:  From a contour map, choose bins based on elliptical isophotal fitting.

4) For each of your profiles, fit a Sersic function and label on the plot the best-fitting Sersic index.

5) Calculate the half-right radius (in ```arcsec```) from your radial profile.
