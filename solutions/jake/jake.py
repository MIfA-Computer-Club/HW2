import astropy.coordinates
import astropy.io.fits
import astropy.wcs
import matplotlib.gridspec
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import os


import galaxyphot


REPO_DIR = '/Users/Jake/Research/code/computer_club/HW2'  # winston
#REPO_DIR = '/Users/Jake/Research/code/computer_club/cc2.0/HW2'  # mbook

# Figure stuff
FIG_DX, FIG_DY = 6, 6  # inches
NROW, NCOL = 3, 3


def normexponent(val):
    """Return the exponent n such that 0 < val/10**n < 10."""
    n = np.log10(val)
    if n < 0:
        n = int(n) - 1
    else:
        n = int(n)
    return n


def problem_1(**kwargs):
    verbose = kwargs.get('verbose', False)
    plot = kwargs.get('plot', False)

    results = {}  # A place to store results for subsequent problems


    # Load image data
    img_file = os.path.join(REPO_DIR, 'POSIIF_Coma.fits')
    image, header = astropy.io.fits.getdata(img_file, header=True)
    results['image'], results['header'] = image, header


    # Create apertures from the DS9 region file.
    reg_file = os.path.join(REPO_DIR, 'POSIIF_Coma.reg')
    aperture_list = galaxyphot.from_region_file(reg_file)


    # Measure the centroids of the sources and reposition the apertures.
    # '2dgauss' (2d gaussing fitting) is more accurate than 'com' (center
    # of mass), but it's much slower.
    rows = []
    for aperture in aperture_list:
        xy0_old = aperture.xy0
        xy0 = galaxyphot.find_centroid(image, aperture, mode='2dgauss')
        rows.append((aperture.label, xy0_old[0], xy0_old[1], xy0[0], xy0[1]))
    results['apertures_#1'] = aperture_list

    if verbose:
        print (
            'Centroids:\n'
            '\n'
            '======= ================= =================\n'
            'galaxy  old x,y           new x, y\n'
            '======= ================= ================='
            )
        for row in rows:
            line = '{0:7s} {1:8.3f} {2:8.3f} {3:8.3f} {4:8.3f}'
            print line.format(*row)
        print '======= ================= =================\n\n'

    """
    Centroids:

    ======= ================= =================
    galaxy  old x,y           new x, y
    ======= ================= =================
    NGC4875 1619.000 1686.000 1619.288 1687.129
    NGC4869 1809.667 1705.167 1810.737 1705.404
    GMP4277 2500.600 2083.800 2503.363 2083.335
    GMP4350 2608.600 1919.400 2615.732 1917.089
    NGC4860 2056.000 2464.000 2055.827 2466.567
    NGC4881 1345.363 2897.784 1344.625 2897.277
    NGC4921  203.461 1594.507  198.296 1596.100
    ======= ================= =================

    """


    # Plot a "swatch" of each source on a 3x3 grid
    if plot:

        gridspec_kwargs = dict(
            left=0.02, bottom=0.02,
            right=0.98, top=0.98,
            wspace=0.06, hspace=0.06
            )

        fmin, fmax = 0.01, 1.0  # Scale the image to these fractions of the CDF
        a = 2  # Log stretch factor
        cmap = plt.cm.gist_heat_r

        gs = matplotlib.gridspec.GridSpec(NROW, NCOL, **gridspec_kwargs)
        ax_list = []
        for aperture, spec in zip(aperture_list, gs):
            ax = plt.subplot(spec)

            data = aperture.extract(image).copy()

            # Find the values for fmin and fmax
            values = np.sort(data.ravel())
            cdf = values.cumsum(dtype=float)
            cdf /= cdf[-1]
            values = values[(fmin <= cdf) & (cdf <= fmax)]
            vmin, vmax = values.min(), values.max()

            # Clip, normalize, and apply log stretch
            data = ((np.clip(data, vmin, vmax).astype(float) - vmin)
                    / (vmax - vmin))
            data = np.log(a * data + 1) / np.log(a)

            # Plot image and aperture
            ax.imshow(data, origin='lower', interpolation='nearest',
                      extent=aperture.extent, cmap=cmap)
            ax.plot(aperture.xy0[0], aperture.xy0[1], 'cx', mew=1)
            aperture.plot(ax=ax, ec='k', fc='none', alpha=0.3, zorder=10)

            # Text
            txtkw = dict(transform=ax.transAxes, zorder=50)
            ax.text(0.03, 0.92, aperture.label, size=10, **txtkw)
            ax.text(0.03, 0.85, 'x = {:.3f}'.format(aperture.xy0[0]),
                    size=8, **txtkw)
            ax.text(0.03, 0.78, 'y = {:.3f}'.format(aperture.xy0[1]),
                    size=8, **txtkw)

            ax.axis(aperture.extent)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            ax_list.append(ax)

        fig  = ax_list[0].figure
        fig.set_size_inches((FIG_DX, FIG_DY))
        filename = os.path.join(REPO_DIR, 'solutions', 'jake', 'galaxies.pdf')
        fig.savefig(filename, dpi=200)

    return results


def problem_2(results, **kwargs):
    verbose = kwargs.get('verbose', False)


    # Determine pixel area in arcsec2
    header = results['header']
    wcs = astropy.wcs.WCS(header)
    x, y = np.array([header['crpix1'], header['crpix2']])
    ra, dec = wcs.wcs_pix2world([x, x+1, x], [y, y, y+1], 1)
    points = astropy.coordinates.SkyCoord(ra, dec, 'icrs', unit=wcs.wcs.cunit)
    dxy = astropy.coordinates.Angle([points[0].separation(points[1]),
                                     points[0].separation(points[2])])
    pixel_scales = dxy.arcsec
    pixel_area = pixel_scales[0] * pixel_scales[1]
    results['pixel_scales'], results['pixel_area'] = pixel_scales, pixel_area


    # Photometry
    radius_mult = 1.5  # Size of inner radius wrt aperture (factor)
    dr = 50.0  # Size of outer radius wrt inner (offset in pixels)
    aperture_list = results['apertures_#1']
    image = results['image']
    annulus_list = []
    for aperture in aperture_list:
        xy0 = aperture.xy0
        r1 = radius_mult * aperture.r
        r2 = r1 + dr
        annulus = galaxyphot.CircularAnnulus(xy0, r1, r2, label=aperture.label)
        annulus_list.append(annulus)

    apphot = galaxyphot.apphot(image, aperture_list)
    anphot = galaxyphot.apphot(image, annulus_list)


    # Intensities, counts/pixel
    intensity_tot = apphot['mean']  # Same as total/area
    intensity_bg = anphot['median']
    intensity = intensity_tot - intensity_bg
    err = np.sqrt(intensity_tot * apphot['area'] +
                  intensity_bg * anphot['area']) / apphot['area']


    # Intensities, counts/arcsec2
    intensity /= pixel_area
    err /= pixel_area
    results['intensity_bgannulus'] = intensity_bg / pixel_area

    if verbose:
        print (
            'Photometry:\n'
            '\n'
            '======= ============= ========\n'
            'galaxy  intensity     err\n'
            '        (cnt/arcsec2)\n'
            '======= ============= ========'
            )
        for i, label in enumerate(apphot['label']):
            vals = (label, intensity[i], err[i])
            print '{0:7s} {1:13.2e} {2:8.2e}'.format(*vals)
        print '======= ============= ========\n\n'

    """
    Photometry:

    ======= ============= ========
    galaxy  intensity     err
            (cnt/arcsec2)
    ======= ============= ========
    NGC4875      1.73e+03 7.69e+00
    NGC4869      5.64e+03 1.08e+01
    GMP4277      2.69e+03 5.59e+00
    GMP4350      2.19e+03 5.54e+00
    NGC4860      4.84e+03 7.79e+00
    NGC4881      1.66e+03 2.37e+00
    NGC4921      7.39e+02 4.91e-01
    ======= ============= ========

    """

    return results


def problem_3(results, **kwargs):
    verbose = kwargs.get('verbose', False)
    plot = kwargs.get('plot', False)

    image = results['image']
    aperture_list = results['apertures_#1']
    intensity_bg_list = results['intensity_bgannulus']  # cnt/arcsec2
    pixel_scales = results['pixel_scales']
    pixel_area = results['pixel_area']


    # For each galaxy...
    dr = 3.0  # Width of the radial bins in pixels
    radial_bins = []
    for i, ap in enumerate(aperture_list):
        label, xy0, rmax = ap.label, ap.xy0, ap.r
        intensity_bg = intensity_bg_list[i]


        # Get elliptical parameters for the galaxy
        aperture = galaxyphot.EllipticalAperture(xy0, rmax, rmax, 0)
        amp, x0, y0, a, b, theta = galaxyphot.fit_gaussian(image, aperture)
        if a < b:
            a, b = b, a
            theta += 90.0
        axratio = b / a


        # List of r values from 0 to near rmax; make apertures
        n = (int(rmax / dr) + 1)
        radii = np.linspace(dr, dr * n, n)
        aperture_list = [galaxyphot.EllipticalAperture(xy0, r, r*axratio, theta)
                         for r in radii]


        # Measure intensities, cnt/arcsec2
        apphot = galaxyphot.apphot(image, aperture_list)
        intensity_tot = apphot['mean']
        intensity = intensity_tot - intensity_bg
        err = apphot['std']
        intensity /= pixel_area
        err /= pixel_area

        radial_bins.append((label, radii, intensity, err))


    # Plot
    if plot:

        gridspec_kwargs = dict(
            left=0.1, bottom=0.1,
            right=0.98, top=0.98,
            wspace=0.25, hspace=0.25
            )

        gs = matplotlib.gridspec.GridSpec(NROW, NCOL, **gridspec_kwargs)
        ax_list = []
        for radial_bin, spec in zip(radial_bins, gs):
            label, radii, intensity, err = radial_bin

            ax = plt.subplot(spec)

            # Using the standard deviation of pixel values in a radial bin
            # isn't really correct. Intensity can't be negative, so it
            # would really be better to assume a lognormal distrubution.
            # (Of course, that would require well-calibrated data. Image
            # data often has weird offsets, sometimes leading to negative
            # values.) But anyway...
            err1, err2 = intensity - err, intensity + err

            n = normexponent(intensity.max())
            radii = radii * np.mean(pixel_scales)
            ax.plot(radii, intensity/10**n, 'ko', ms=3, zorder=10)
            ax.vlines(radii, err1/10**n, err2/10**n, colors='0.7', linestyles='-',
                      zorder=9)
            ax.set_ylim([0, ax.get_ylim()[1]])

            # Tick adjustments
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))

            # Text
            ax.text(0.03, 0.03, label, size=10,
                    transform=ax.transAxes, zorder=50)

            ax_list.append(ax)

        # Axis labels, figure size
        ax = ax_list[0]
        fig  = ax.figure
        ax.text(0.5, 0.02, 'r (arcsec)', size=10,
                ha='center', transform=fig.transFigure)
        ylabel = (r'Intensity ($10^{{{:d}}} '
                  '\mathrm{{\,cnt \,arcsec^2}}$)'.format(n))
        ax.text(0.02, 0.5, ylabel, size=10, rotation='vertical',
                va='center', transform=fig.transFigure)
        fig.set_size_inches((FIG_DX, FIG_DY))

        filename = os.path.join(REPO_DIR, 'solutions', 'jake',
                                'radial_profiles.pdf')
        fig.savefig(filename)


        # Optional: use isophotes
        #
        # - calculate contours (matplotlib?)
        # - make a list of I values
        # - at each I:
        #
        #   - make a polygon aperture
        #   - get weights
        #   - make an r array, same shape as weights
        #   - weighted average of r, std
        #
        # - plot

        return results


def main():
    results = problem_1(verbose=True, plot=True)
    results = problem_2(results, verbose=True)
    results = problem_3(results, verbose=True, plot=True)

if __name__ == '__main__':
    main()



# 4. Fit Sersic functions to the radial profiles; plot


# 5. Half-light radius (arcsec) of the profiles.
