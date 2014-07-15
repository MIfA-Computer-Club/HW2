import numpy as np

# 1a. Using ds9 region file positions as initial guesses, determine the
#     centroids of the sources.

"""
Write a quick ds9 region file parser.

Assume the aperture is fully specified: x, y, r for circular; centroiding
is an optional step for improving the accuracy of x and y.

"""


class CircularAperture(object):

    """A Circular aperture.

    Parameters
    ----------
    xy : 2-tuple of floats
        Initialize the `xy` property.
    r : float
        Initialize the `r` property.
    image : 2d array, optional
        Initialize the `image` property. Default is None.
    label : string, optional
        Initialize the `label` attribute. Default is None.
    nsub : int, optional
        Initialize the `nsub` attribute. Default is 1.

    Attributes
    ----------
    parameters
    x
    y
    xy
    r
    xmin
    xmax
    ymin
    ymax
    xrange
    yrange
    xyrange
    imin
    imax
    jmin
    jmax
    irange
    jrange
    ijrange
    image
    pixels
    weights
    label : string
        Region label.
    nsub : int
        Number of subpixels (per side) in which to sample each border pixel
        when calculating weights for partial pixels in the aperture. The
        total number of subpixel samples for a border pixel is ``nsub**2``.

    Methods
    -------
    centroid

    """

    def __init__(self, xy, r, image=None, label=None, nsub=1):
        self.parameters = [xy[0], xy[1], r]
        self.image = image
        self.label = label
        self.nsub = nsub  # make a property?

    @property
    def parameters(self):
        """x, y, and r values of the circle.

        See the `x`, `y`, and `r` properties.

        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters
        self._reset()

    @property
    def x(self):
        """x coordinate of the center."""
        return self._parameters[0]

    @x.setter
    def x(self, x):
        self._parameters[0] = x
        self.parameters = self._parameters

    @property
    def y(self):
        """y coordinate of the center."""
        return self._parameters[1]

    @y.setter
    def y(self, y):
        self._parameters[1] = y
        self.parameters = self._parameters

    @property
    def xy(self):
        """x and y coordinates of the center."""
        return self._parameters[:2]

    @xy.setter
    def xy(self, xy):
        self._parameters[:2] = xy
        self.parameters = self._parameters

    @property
    def r(self):
        """Circle radius."""
        return self._parameters[2]

    @r.setter
    def r(self, r):
        self._parameters[2] = r
        self.parameters = self._parameters

    @property
    def xmin(self):
        """Minimum x value covered by the circle; read only."""
        return self.x - self.r

    @property
    def xmax(self):
        """Maximum x value covered by the circle; read only."""
        return self.x + self.r

    @property
    def ymin(self):
        """Minimum y value covered by the circle; read only."""
        return self.y - self.r

    @property
    def ymax(self):
        """Maximum y value covered by the circle; read only."""
        return self.y + self.r

    @property
    def xrange(self):
        """Minimum and maximum x values covered by the circle; read
        only.

        """
        return (self.xmin, self.xmax)

    @property
    def yrange(self):
        """Minimum and maximum y values covered by the circle; read
        only.

        """
        return (self.ymin, self.ymax)

    @property
    def xyrange(self):
        """Minimum and maximum x and y values covered by the circle; read
        only.

        """
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def imin(self):
        """Row index of `ymin`; read only."""
        return int(self.ymin - 0.5)

    @property
    def imax(self):
        """Row index of `ymax`; read only."""
        return int(self.ymax - 0.5)

    @property
    def jmin(self):
        """Column index of `xmin`; read only."""
        return int(self.xmin - 0.5)

    @property
    def jmax(self):
        """Column index of `xmax`; read only."""
        return int(self.xmax - 0.5)

    @property
    def irange(self):
        """Slice of all rows from `imin` to `imax`, inclusive; read
        only.

        """
        return slice(self.imin, self.imax+1)

    @property
    def jrange(self):
        """Slice of all columns from `jmin` to `jmax`, inclusive; read
        only.

        """
        return slice(self.jmin, self.jmax+1)

    @property
    def ijrange(self):
        """Slice of all rows and columns from `imin` to `imax` and `jmin`
        to `jmax`, inclusive; read only.

        """
        return (self.irange, self.jrange)

    @property
    def image(self):
        """Image data as a 2d array."""
        return self._image

    @image.setter
    def image(self, arr):
        self._image = arr
        self._reset()

    @property
    def pixels(self):
        """Image data for the smallest square containing the circle; read
        only.

        This is a view into `image`. Although it is a read only property,
        the contents of the array can be changed.

        """
        return self._pixels

    def _reset(self):
        """Reset various attributes so that they are consistent with the
        current `x`, `y`, `r`, and `image`.

        """
        # Reset the view into the image array
        try:
            self.image
        except AttributeError:
            # self._image does not exist yet
            pass
        else:
            if self.image is None:
                self._pixels = None
            else:
                self._pixels = self.image[self.ijrange]

    def weights(self):
        """Fraction of each pixel's area that is within the aperture.

        Areas for partial pixels along the aperture border are approximated
        by sampling each border pixel with ``nsub**2`` subpixels.

        Returns
        -------
        array
            Weights from 0 to 1, same shape as `pixels`.

        """
        if self.pixels is None:
            weights = None
        else:
            # Pixel centers in pixel coordinates; add 0.5 for pixel center,
            # add another 0.5 for array-pixel coord conversion
            xcmin, xcmax = self.jmin+1, self.jmax+1
            ycmin, ycmax = self.imin+1, self.imax+1
            xc = np.arange(xcmin, xcmax+1).reshape(1, -1)  # Cheaper than full grid
            yc = np.arange(ycmin, ycmax+1).reshape(-1, 1)

            # Distances from the circle center
            dx, dy = xc - self.x, yc - self.y
            r = np.sqrt(dx**2 + dy**2)

            # Pixels with centers within r of circle center
            weights = (r <= self.r).astype('float')

            # Partial pixels
            if self.nsub > 1:
                # Border pixels: any pixel with distance within sqrt(0.5)
                # of r; expand dimensions for later broadcasting
                i, j = np.where(np.abs(r - self.r) <= np.sqrt(0.5))
                i, j = i[:,None,None], j[:,None,None]

                # Generic subpixel grid
                gridx = (np.arange(self.nsub).reshape(1, -1) + 0.5) / self.nsub
                gridy = (np.arange(self.nsub).reshape(-1, 1) + 0.5) / self.nsub

                # Centers of subpixels
                xsub = gridx + self.jmin + j + 0.5  # (len(i), 1, nsub)
                ysub = gridy + self.imin + i + 0.5  # (len(i), nsub, 1)

                # Distances from circle center
                dxsub, dysub = xsub - self.x, ysub - self.y
                rsub = np.sqrt(dxsub**2 + dysub**2)  # (len(i), nsub, nsub)

                # New pixel weights
                i, j = i[:,0,0], j[:,0,0]
                sumkwargs = dict(axis=(1, 2), dtype='float')
                weights[i,j] = np.sum(rsub <= self.r, **sumkwargs) / self.nsub**2

        return weights

    def centroid(self, adjust=False):
        """Find the centroid of the source in the aperture.

        Parameters
        ----------
        adjust : bool, optional
            If True, set the aperture's center point to the centroid.

        Returns
        -------
        2-tuple of floats
            x and y pixel coordinates of the centroid of the source.

        """
        if self.pixels is None:
            xyc = None
        else:
            xyc = calc_centroid(self.pixels, weights=self.weights())

        if adjust and xyc is not None:
            self.xy = list(xyc)

        return xyc


def from_region_file(filename):
    """Create apertures from a DS9 region file.

    "circle", "ellipse", and "polygon" region definitions are used to
    create a list of `Circle`, `Ellipse`, and `Polygon` apertures,
    respectively. All values are assumed to be in pixel coordinates.

    Only relatively basic region files can be read. The following features
    of the `DS9 region file format
    <http://ds9.si.edu/doc/ref/region.html#RegionFileFormat>`_ are either
    ignored (i) or are not supported (ns) by the parser:

    - i: Global properties
    - i: Coordinate systems ("PHYSICAL", "FK5", etc.). All values are assumed
      to be in pixel coordinates.
    - i: Shapes other than "circle", "ellipse", and "polygon"

    - ns: Local properties. Any text appearing after "#" in a region definition
      is instead interpreted as the region's label.
    - ns: Using ";" as a line delimiter
    - ns: Multiple coordinate systems
    - ns: Mutliple WCS
    - ns: Explicit units and sexigesimal positions and sizes
    - ns: Composite and template regions

    Parameters
    ----------
    filename : string
        Absolute path to the region file.

    Returns
    -------
    list

    """
    aperture_list = []
    with open(filename, 'r') as f:
        for line in f:
            # Condition input
            line = line.split('#', 1)
            if len(line) == 1:
                line, label = line[0], None
            else:
                line, label = line
                label = label.strip()
            replace = ('(', ')', ',')
            line = ''.join(' ' if char in replace else char for char in line)
            line = line.strip()

            # Region definitions
            if line.startswith('circle'):
                x, y, r = [float(i) for i in line.split()[1:]]
                aperture = CircularAperture((x, y), r, label=label)
                aperture_list.append(aperture)
            elif line.startswith('ellipse'):
                x, y, a, b, theta = [float(i) for i in line.split()[1:]]
                aperture = EllipticalAperture((x, y), a, b, theta, label=label)
                aperture_list.append(aperture)
            elif line.startswith('polygon'):
                xy = np.array(line.split()[1:], 'float').reshape(-1, 2)
                aperture = PolygonAperture(xy, label=label)
                aperture_list.append(aperture)
            else:
                pass

    return aperture_list


def calc_centroid(img, weights=None, xc=None, yc=None):
    """Calculate the centroid of a source.

    The centroid is calculated from the flux-weighted positions of the
    marginal distributions for x and y.

    Parameters
    ----------
    img : array
        Image of the source as a 2d array.
    weights : array, optional
        Array of weights (from 0 to 1) for the pixels in `img`. Default is
        None (no weighting).
    xyc : tuple, optional
        Initial guess for x and y coordinates of the centroid. Default is
        to use the center of `img`.

    Returns
    -------
    tuple
        x and y coordinates, with respect to `img`, of the centroid.

    """
    # Marginal distributions
    if weights is None:
        weights = 1
    zx = np.sum(img * weights, axis=0)
    zy = np.sum(img * weights, axis=1)
    zx, zy = zx - zx.min(), zy - zy.min()

    # x and y coordinates of pixel centers
    x, y = np.arange(zx.size) + 1, np.arange(zy.size) + 1

    # Flux-weighted x and y
    xc, yc = 1, 1
    niter = 10
    xc = x[x.size/2] if xc is None else xc
    yc = y[y.size/2] if yc is None else yc
    dx = np.sum((x - xc) * zx, dtype='float') / np.sum(zx)
    dy = np.sum((y - yc) * zy, dtype='float') / np.sum(zy)
    xc, yc = xc + dx, yc + dy

    return (xc, yc)



import astropy.io.fits

imgfile = '/Users/Jake/Research/code/computer_club/HW2/POSIIF_Coma.fits'
data = astropy.io.fits.getdata(imgfile)


regfile = '/Users/Jake/Research/code/computer_club/HW2/POSIIF_Coma.reg'
aperture_list = from_region_file(regfile)
for aperture in aperture_list:
    aperture.image = data


a = aperture_list[2]
self = a
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
