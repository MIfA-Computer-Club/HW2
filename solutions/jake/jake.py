import astropy.io.fits
import matplotlib.gridspec
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import os


import galaxyphot


def problem_1():
    # Load image data
    repo_dir = '/Users/Jake/Research/code/computer_club/HW2'
    img_file = os.path.join(repo_dir, 'POSIIF_Coma.fits')
    data = astropy.io.fits.getdata(img_file)

    # Create apertures from the DS9 region file, measure the centroids of
    # the sources and reposition the apertures
    reg_file = os.path.join(repo_dir, 'POSIIF_Coma.reg')
    aperture_list = galaxyphot.from_region_file(reg_file)
    for aperture in aperture_list:
        aperture.image = data
        xy1 = aperture.xy
        xy2 = aperture.centroid(adjust=True)
        line = '{0:s}    old: {1:8.3f} {2:8.3f}    new: {3:8.3f} {4:8.3f}'
        print line.format(aperture.label, xy1[0], xy1[1], xy2[0], xy2[1])
    print

    """
    NGC4875    old: 1619.000 1686.000    new: 1619.288 1687.129
    NGC4869    old: 1809.667 1705.167    new: 1810.764 1705.461
    GMP4277    old: 2500.600 2083.800    new: 2503.357 2083.336
    GMP4350    old: 2608.600 1919.400    new: 2615.730 1917.089
    NGC4860    old: 2056.000 2464.000    new: 2055.825 2466.551
    NGC4881    old: 1345.363 2897.784    new: 1344.625 2897.277
    NGC4921    old:  203.461 1594.507    new:  198.450 1596.031

    """

    # Plot a "swatch" of each source on a 3x3 grid
    fig_dx, fig_dy = 6, 6  # inches
    nrow, ncol = 3, 3
    cmap = plt.cm.gist_heat_r
    pmin, pmax = 0.01, 1.0

    gs = matplotlib.gridspec.GridSpec(
        nrow, ncol, left=0.02, bottom=0.02, right=0.98, top=0.98,
        wspace=0.02, hspace=0.02)
    ax_list = []
    for aperture, spec in zip(aperture_list, gs):
        ax = plt.subplot(spec)

        extent = (aperture.jmin+0.5, aperture.jmax+1.5,
                  aperture.imin+0.5, aperture.imax+1.5)
        data = aperture.section.copy()

        # Find vmin and vmax to scale the image from pmin to pmax
        values = np.sort(data.ravel())
        cdf = values.cumsum(dtype='float')
        cdf /= cdf[-1]
        values = values[(pmin <= cdf) & (cdf <= pmax)]
        vmin, vmax = values.min(), values.max()
        data = np.clip(data, vmin, vmax)

        # Stretch
        a = 1e5
        data = np.log10(data - vmin + a)
        vmin, vmax = data.min(), np.log10(vmax - vmin + a)

        # Plot image and aperture
        ax.imshow(
            data, origin='lower', interpolation='nearest',
            extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.plot(aperture.x, aperture.y, 'cx', mew=2)
        circle = matplotlib.patches.Circle(
            (aperture.x, aperture.y), radius=aperture.r, ec='k', fc='none')
        ax.add_patch(circle)

        # Text
        ax.text(0.03, 0.93, aperture.label, size=10, transform=ax.transAxes)
        ax.text(0.03, 0.87, 'x = {:.3f}'.format(aperture.x), size=8, transform=ax.transAxes)
        ax.text(0.03, 0.81, 'y = {:.3f}'.format(aperture.y), size=8, transform=ax.transAxes)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax_list.append(ax)

    fig  = ax_list[0].figure
    fig.set_size_inches((fig_dx, fig_dy))
    filename = os.path.join(repo_dir, 'solutions', 'jake', 'galaxies.pdf')
    fig.savefig(filename, dpi=200)

    return aperture_list


def problem_2(aperture_list):
    # Aperture phot in counts/pix2, accounting for background flux, and
    # convert to counts/arcsec2.

    for aperture in aperture_list:
        # Total counts
        weights = aperture.weights()
        counts = np.sum(aperture.section * weights)

        print '{0:s}: {1:.2e}'.format(aperture.label, counts)

        """
        NGC4875: 8.59e+06
        NGC4869: 9.52e+06
        GMP4277: 1.36e+07
        GMP4350: 1.26e+07
        NGC4860: 1.24e+07
        NGC4881: 3.54e+07
        NGC4921: 2.64e+08

        """

    return



def main():
    aperture_list = problem_1()
    problem_2(aperture_list)


if __name__ == '__main__':
    main()







# 3. Azimuthally-averaged radial profile as counts/arcsec2 vs. arcsec,
#    including standard deviations. Plot the radial profiles with
#    uncertainties.
#    - circular bins
#    - elliptical bins
#    - isophotal bins


# 4. Fit Sersic functions to the radial profiles; plot


# 5. Half-light radius (arcsec) of the profiles.



def testplot(ap, arr, grid=False, cmap= plt.cm.gist_heat):
    plt.cla()
    x = np.linspace(ap.xmin, ap.xmax, 1000)
    y1 = np.sqrt(ap.r**2 - (x-ap.x)**2) + ap.y
    y2 = -np.sqrt(ap.r**2 - (x-ap.x)**2) + ap.y
    extent = (ap.jmin+0.5, ap.jmax+1.5, ap.imin+0.5, ap.imax+1.5)
    plt.imshow(arr, origin='lower', interpolation='nearest', extent=extent,
               cmap=cmap)
    plt.plot(np.hstack((x, x[::-1])), np.hstack((y1, y2[::-1])), 'g-')
    plt.plot(ap.x, ap.y, 'gx')
    if grid:
        xl = np.arange(ap.jmin+0.5, ap.jmax+1.5)
        yl = np.arange(ap.imin+0.5, ap.imax+1.5)
        plt.vlines(xl, extent[2], extent[3])
        plt.hlines(yl, extent[0], extent[1])
    plt.axis(extent)
