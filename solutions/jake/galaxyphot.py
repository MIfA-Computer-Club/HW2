"""

============
`galaxyphot`
============

Utilities for galaxy photometry.


Classes
-------

==================== =======================
`CircularAperture`   A circular aperture.
`EllipticalAperture` An elliptical aperture.
`PolygonAperture`    A polygon aperture.
==================== =======================


Functions
---------

================== ===========================================
`from_region_file` Create apertures from a DS9 region file.
`center_of_mass`   Calculate the "center of mass" in an image.
`gaussian2d`       2d Gaussian function.
================== ===========================================

"""
import numpy as np
import scipy.optimize


class CircularAperture(object):

    """A circular aperture.

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
        Initialize the `nsub` attribute. Default is 50.

    Attributes
    ----------
    parameters
    i
    j
    ij
    x
    y
    xy
    r
    imin
    imax
    jmin
    jmax
    irange
    jrange
    ijrange
    ijslice
    xmin
    xmax
    ymin
    ymax
    xrange
    yrange
    xyrange
    image
    section
    label : string
        Region label.
    nsub : int
        Number of subpixels (per side) in which to sample each border pixel
        when calculating weights for partial pixels in the aperture. The
        total number of subpixel samples for a border pixel is ``nsub**2``.
        Small apertures require larger `nsub` values to maintain accuracy,
        while smaller `nsub` values will suffice for large apertures. If 1,
        then partial pixels are not computed.

    Methods
    -------
    weights
    fit_gaussian
    centroid

    """

    def __init__(self, xy, r, image=None, label=None, nsub=50):
        i, j = xy[1] - 0.5, xy[0] - 0.5  # Convert into array coordinates
        self.parameters = [i, j, r]
        self.image = image
        self.label = label
        self.nsub = nsub

    @property
    def parameters(self):
        """i, j, and r values of the aperture.

        See the `i`, `j`, and `r` properties.

        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

        # Reset the view into the image array
        try:
            self.image
        except AttributeError:
            # self._image does not exist yet
            pass
        else:
            if self.image is not None:
                self._section = self.image[self.ijslice]

    @property
    def i(self):
        """i coordinate of the center."""
        return self._parameters[0]

    @i.setter
    def i(self, i):
        self._parameters[0] = i
        self.parameters = self._parameters

    @property
    def j(self):
        """j coordinate of the center."""
        return self._parameters[1]

    @j.setter
    def j(self, j):
        self._parameters[1] = j
        self.parameters = self._parameters

    @property
    def ij(self):
        """i and j coordinates of the center."""
        return self._parameters[:2]

    @ij.setter
    def ij(self, ij):
        self.i, self.j = ij

    @property
    def x(self):
        """x coordinate of the center."""
        return self.j + 0.5

    @x.setter
    def x(self, x):
        self._parameters[1] = x - 0.5
        self.parameters = self._parameters

    @property
    def y(self):
        """y coordinate of the center."""
        return self.i + 0.5

    @y.setter
    def y(self, y):
        self._parameters[0] = y - 0.5
        self.parameters = self._parameters

    @property
    def xy(self):
        """x and y coordinates of the center."""
        return (self.x, self.y)

    @xy.setter
    def xy(self, xy):
        self.x, self.y = xy

    @property
    def r(self):
        """Circle radius."""
        return self._parameters[2]

    @r.setter
    def r(self, r):
        self._parameters[2] = r
        self.parameters = self._parameters

    @property
    def imin(self):
        """Minimum i value of the aperture; read only."""
        return self.i - self.r

    @property
    def imax(self):
        """Maximum i value of the aperture; read only."""
        return self.i + self.r

    @property
    def jmin(self):
        """Minimum j value of the aperture; read only."""
        return self.j - self.r

    @property
    def jmax(self):
        """Maximum j value of the aperture; read only."""
        return self.j + self.r

    @property
    def irange(self):
        """Minimum and maximum i values of the aperture; read only."""
        return (self.imin, self.imax)

    @property
    def jrange(self):
        """Minimum and maximum i values of the aperture; read only."""
        return (self.jmin, self.jmax)

    @property
    def ijrange(self):
        """Minimum and maximum i and j values of the aperture; read only."""
        return (self.imin, self.imax, self.jmin, self.jmax)

    @property
    def ijslice(self):
        """Slice of from `imin` to `imax` and `jmin` to `jmax`, inclusive;
        read only.

        """
        return (slice(self.imin, self.imax+1), slice(self.jmin, self.jmax+1))

    @property
    def xmin(self):
        """Minimum x value of the aperture; read only."""
        return self.x - self.r

    @property
    def xmax(self):
        """Maximum x value of the aperture; read only."""
        return self.x + self.r

    @property
    def ymin(self):
        """Minimum y value of the aperture; read only."""
        return self.y - self.r

    @property
    def ymax(self):
        """Maximum y value of the aperture; read only."""
        return self.y + self.r

    @property
    def xrange(self):
        """Minimum and maximum x values of the aperture; read only."""
        return (self.xmin, self.xmax)

    @property
    def yrange(self):
        """Minimum and maximum y values of the aperture; read only."""
        return (self.ymin, self.ymax)

    @property
    def xyrange(self):
        """Minimum and maximum x and y values of the aperture; read only."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def image(self):
        """Image data as a 2d array."""
        return self._image

    @image.setter
    def image(self, arr):
        self._image = arr

        # Reset the view into the image array
        self._section = None if arr is None else arr[self.ijslice]

    @property
    def section(self):
        """Smallest section of the image containing the aperture; read
        only.

        This is a view into `image`. Although it is a read only property,
        the contents of the array can be changed.

        """
        return self._section

    def weights(self):
        """Fraction of each pixel's area that is within the aperture.

        Areas for partial pixels along the aperture border are approximated
        by sampling each border pixel with ``nsub**2`` subpixels.

        Returns
        -------
        array
            Weight values between 0 to 1, same shape as `section`.

        """
        if self.section is None:
            weights = None
        else:
            # Pixel centers in array coordinates
            ni, nj = self.section.shape
            i = np.arange(ni).reshape(-1, 1) + 0.5 + self.imin
            j = np.arange(nj).reshape(1, -1) + 0.5 + self.jmin

            # Distances from the aperture center
            r = np.sqrt((i - self.i)**2 + (j - self.j)**2)

            # Pixels with centers within r of aperture center
            weights = (r <= self.r).astype('float')

            # Partial pixels
            if self.nsub > 1:
                # Indices (lower-left corners) of border pixels
                i, j = np.where(np.abs(r - self.r) <= np.sqrt(0.5))

                # Generic subpixel grid
                gridi = (np.arange(self.nsub).reshape(-1, 1) + 0.5) / self.nsub
                gridj = (np.arange(self.nsub).reshape(1, -1) + 0.5) / self.nsub

                # Centers of subpixels
                isub = gridi + i[:,None,None] + self.imin  # (len(i), nsub, 1)
                jsub = gridj + j[:,None,None] + self.jmin  # (len(i), 1, nsub)

                # Distances from aperture center; (len(i), nsub, nsub)
                rsub = np.sqrt((isub - self.i)**2 + (jsub - self.j)**2)

                # Refined pixel weights
                kwargs = dict(axis=(1, 2), dtype='float')
                weights[i,j] = np.sum(rsub <= self.r, **kwargs) / self.nsub**2

        return weights

    def fit_gaussian(self):
        """Fit a 2d Gaussian function to the source.

        Initial guesses for the amplitude, center, width, and rotation
        parameters are the difference between the maximum and minimum
        values in `section`, the current x and y of the aperture, the
        current radius, and 0, respectively.

        Returns
        -------
        tuple
            Best-fit 2d Gaussian parameters: amplitude, x and y pixel
            coordinates of the center, x and y widths (sigma), and counter
            clockwise rotation angle.

        """
        def func(xy, amp, x0, y0, sigmax, sigmay, theta):
            f = gaussian2d(xy[0], xy[1], amp, x0, y0, sigmax, sigmay, theta)
            return f.ravel()

        # Pixel centers in array coordinates
        ni, nj = self.section.shape
        i = np.arange(ni).reshape(-1, 1) + 0.5 + self.imin
        j = np.arange(nj).reshape(1, -1) + 0.5 + self.jmin

        # Find the best fit parameters
        dz = self.section.ravel() - self.section.min()
        p0 = (dz.max(), self.j, self.i, self.r/2.0, self.r/2.0, 0.0)
        popt, pcov = scipy.optimize.curve_fit(func, (j, i), dz, p0=p0)
        popt[1], popt[2] = popt[1] + 0.5, popt[2] + 0.5  # Pixel coords
        return tuple(popt)

    def centroid(self, adjust=False, mode='2dgauss', rtol=0.01):
        """Find the centroid of the source in the aperture.

        Parameters
        ----------
        adjust : bool, optional
            If True, set the aperture's center point to the centroid.
            Default is False.
        mode : {'2dgauss', 'com'}, optional
            The method for computing the centroid. If '2dgauss' (default),
            a 2d Gaussian function is fit to the source using the
            `fit_gaussian` method. If 'com', peaks in the marginal
            distributions are used as an initial guess for the centroid,
            then the location is refined by iteratively calculating the
            "center of mass" until convergence within a given tolerance.
        rtol : float, optional
            Relative tolerance (fractional difference with respect to the
            previous iteration) for the centroid solution in 'com' mode.
            Iteration stops when the centroid position changes by less than
            `rtol`. Default is 0.01.

        Returns
        -------
        2-tuple of floats
            x and y pixel coordinates of the centroid of the source.

        """
        if self.section is None:
            xc, yc = None, None
        else:
            parameters_copy = self._parameters[:]

            if mode == '2dgauss':
                amp, xc, yc, sigmax, sigmay, theta = self.fit_gaussian()
                self.x, self.y = xc, yc

            elif mode == 'com':
                # Use peaks of the marginal distributions as an initial guess
                weights = self.weights()
                sumi = np.sum(self.section * weights, axis=1)
                sumj = np.sum(self.section * weights, axis=0)

                # Pixel centers in array coordinates
                i = np.arange(sumi.size) + 0.5 + self.imin
                j = np.arange(sumj.size) + 0.5 + self.jmin

                # Coordinates of maxima
                ic, jc = i[sumi.argmax()], j[sumj.argmax()]
                xc, yc = jc + 0.5, ic + 0.5
                self.x, self.y = xc, yc  # parameters.setter updates section

                # Optimize
                t = rtol + 1
                rc_prev = self.section.shape[0] + 1  # off image
                while t >= rtol:
                    xc, yc = center_of_mass(self.section, weights=self.weights())
                    xc, yc = xc + self.jmin, yc + self.imin  # image coords
                    self.x, self.y = xc, yc  # The setters update section
                    rc = np.sqrt(xc**2 + yc**2)
                    t = np.abs(rc - rc_prev) / rc_prev
                    rc_prev = rc

            if not adjust:
                self.parameters = parameters_copy

        return (xc, yc)


class EllipticalAperture(object):

    """An elliptical aperture.

    Parameters
    ----------
    xy : 2-tuple of floats
        Initialize the `xy` property.
    a : float
        Initialize the `a` property.
    b : float
        Initialize the `b` property.
    theta : float
        Initialize the `theta` property.
    image : 2d array, optional
        Initialize the `image` property. Default is None.
    label : string, optional
        Initialize the `label` attribute. Default is None.
    nsub : int, optional
        Initialize the `nsub` attribute. Default is 1.

    """

    def __init__(self, xy, a, b, theta, image=None, label=None, nsub=1):
        pass


class PolygonAperture(object):

    """A polygon aperture.

    Parameters
    ----------
    xy : 2-tuple of floats
        Initialize the `xy` property.
    image : 2d array, optional
        Initialize the `image` property. Default is None.
    label : string, optional
        Initialize the `label` attribute. Default is None.
    nsub : int, optional
        Initialize the `nsub` attribute. Default is 1.

    """

    def __init__(self, xy, image=None, label=None, nsub=1):
        pass


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


def center_of_mass(img, weights=None):
    """Calculate the "center of mass" in an image.

    Parameters
    ----------
    img : array
        Image of the source as a 2d array.
    weights : array, optional
        Array of weights (from 0 to 1) for the pixels in `img`. Default is
        None (no weighting).

    Returns
    -------
    tuple
        x and y pixel coordinates of the center of mass. The coordinates
        are given with respect to `img`.

    """
    if weights is None:
        weights = 1

    # i and j coordinates of pixel centers
    i = np.arange(img.shape[0]).reshape(-1, 1) + 0.5
    j = np.arange(img.shape[1]).reshape(1, -1) + 0.5

    # Center of mass
    img_sum = np.sum(img)
    icom = np.sum(img * i, dtype='float') / img_sum
    jcom = np.sum(img * j, dtype='float') / img_sum
    xcom, ycom = jcom + 0.5, icom + 0.5

    return (xcom, ycom)


def gaussian2d(x, y, amp, x0, y0, sigmax, sigmay, theta):
    """2d Gaussian function.

    Parameters
    ----------
    x, y : array
        Input x and y coordinates. Both arrays must be the same shape, or
        be broadcastable.
    amp : float
        Amplitude.
    x0, y0 : float
        Location of the center.
    sigmax, sigmay : float
        Widths (sigma).
    theta : float
        Counter clockwise rotation angle, degrees.

    Returns
    -------
    array
        2d Gaussian function values, same shape as `x` and `y` or the
        result of their broadcasting.

    """
    theta = theta * np.pi/180
    a = np.cos(theta)**2 / (2*sigmax**2) + np.sin(theta)**2 / (2*sigmay**2)
    b = np.sin(2*theta) / (4*sigmax**2) - np.sin(2*theta) / (4*sigmay**2)
    c = np.sin(theta)**2 / (2*sigmax**2) + np.cos(theta)**2 / (2*sigmay**2)
    f = amp * np.exp(-(a*(x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2))
    return f
