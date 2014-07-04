# 1a. Using ds9 region file positions as initial guesses, determine the
#     centroids of the sources.

"""
Write a quick ds9 region file parser.

Assume the aperture is fully specified: x, y, r for circular; centroiding
is an optional step for improving the accuracy of x and y.

"""


# 1b. Plot a "swatch" of each source.

"""
3x3 grid?

"""


# 2. Aperture phot in counts/pix2, accounting for background flux, and
#    convert to counts/arcsec2.

"""
Try upsampling the image in the aperture to a finer grid, ideally just
around the aperture border, to account for partial pixels.

"""


# 3. Azimuthally-averaged radial profile as counts/arcsec2 vs. arcsec,
#    including standard deviations. Plot the radial profiles with
#    uncertainties.
#    - circular bins
#    - elliptical bins
#    - isophotal bins


# 4. Fit Sersic functions to the radial profiles; plot


# 5. Half-light radius (arcsec) of the profiles.
