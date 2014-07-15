import astropy.io.fits
import os


import galaxyphot


def problem_1():
    """Using ds9 region file positions as initial guesses, determine the
    centroids of the sources.

    """
    repo_dir = '/Users/Jake/Research/code/computer_club/cc2.0/HW2'
    img_file = os.path.join(repo_dir, 'POSIIF_Coma.fits')
    data = astropy.io.fits.getdata(img_file)

    reg_file = os.path.join(repo_dir, 'POSIIF_Coma.reg')
    aperture_list = galaxyphot.from_region_file(reg_file)
    for aperture in aperture_list:
        aperture.image = data
        xy1 = aperture.xy
        xy2 = aperture.centroid(adjust=True)
        line = '{0:s}    old: {1:8.3f} {2:8.3f}    new: {3:8.3f} {4:8.3f}'
        print line.format(aperture.label, xy1[0], xy1[1], xy2[0], xy2[1])

    """
    NGC4875    old: 1619.000 1686.000    new: 1619.065 1687.107
    NGC4869    old: 1809.667 1705.167    new: 1810.735 1705.055
    GMP4277    old: 2500.600 2083.800    new: 2503.109 2083.081
    GMP4350    old: 2608.600 1919.400    new: 2615.179 1916.992
    NGC4860    old: 2056.000 2464.000    new: 2055.988 2466.879
    NGC4881    old: 1345.363 2897.784    new: 1344.473 2896.951
    NGC4921    old:  203.461 1594.507    new:  198.211 1594.774

    """
    return


def main():
    problem_1()


if __name__ == '__main__':
    main()





"""
a = aperture_list[2]
def testplot(arr):
    plt.cla()
    X = np.linspace(a.xmin, a.xmax, 1000)
    Y1 = np.sqrt(a.r**2 - (X-a.x)**2) + a.y
    Y2 = -np.sqrt(a.r**2 - (X-a.x)**2) + a.y
    extent = (a.jmin+0.5, a.jmax+1.5, a.imin+0.5, a.imax+1.5)
    plt.imshow(arr, origin='lower', interpolation='nearest', extent=extent)
    plt.plot(np.hstack((X, X[::-1])), np.hstack((Y1, Y2[::-1])), 'g-')
    plt.plot(a.x, a.y, 'gx')
    xl = np.arange(a.jmin+0.5, a.jmax+1.5)
    yl = np.arange(a.imin+0.5, a.imax+1.5)
    plt.vlines(xl, extent[2], extent[3])
    plt.hlines(yl, extent[0], extent[1])
    plt.axis(extent)
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
