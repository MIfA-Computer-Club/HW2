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
`gaussian2d`       2d Gaussian function.
================== ===========================================

"""
import numpy as np
import scipy.optimize


"""
Aperture (CircularAperture, CircularAnnulus,
          EllipticalAperture, EllipticalAnnulus, PolygonAperture)
.parameters (x, y, r, etc.)
.extent (for slicing an image, computing weights; should be linked to parameters)
.weights

fit_gaussian(image, aperture)
find_centroid(image, aperture)
measure_photometry(image, aperture)


# a lot of design inspired by photutils; implementations differ
image = astropy.io.getdata(img_filename)
apertures = from_region_file(reg_filename)  # list of CircularAperture instances
annuli = []
for aperture in apertures:
    aperture.xy = find_centroid(image, aperture)
    annuli.append(CircularAnnulus(...))
apphot = measure_photometry(image, apertures, nsub=50)  # label, total (weighted), area (npix), median, std
anphot = measure_photometry(image, annuli, nsub=50)
total = apphot['total']
background = anphot['median'] * apphot['area']
signal = total - background
noise = np.sqrt(total + background)

"""


class ApertureBase(object):

    """Base class for aperture subclasses.

    Attributes and properties common to all aperture classes are defined
    here.

    Parameters
    ----------
    ijrange : 4-tuple
        Minimum and maximum i and j values of the aperture, ``(imin, imax,
        jmin, jmax)``.
    label : string, optional
        Initialize the `label` attribute. Default is None.
    nsub : int, optional
        Initialize the `nsub` attribute. Default is 50.

    Attributes
    ----------
    centers
    ijextent
    ijslice
    imax
    imin
    jmax
    jmin
    xmax
    xmin
    xyextent
    ymax
    ymin
    label : string
        Label.
    nsub : int
        Number of subpixels (per side) in which to sample each border pixel
        when calculating weights for partial pixels in the aperture. The
        total number of subpixel samples for a border pixel is ``nsub**2``.
        Small apertures require larger `nsub` values to maintain accuracy,
        while smaller `nsub` values will suffice for large apertures.

    Notes
    -----
    i and j denote array coordinates, measuring along the rows and columns,
    respectively, where the origin (0, 0) is the outside corner of the
    first row and column. x and y denote pixel coordinates, corresponding
    to j (columns) and i (rows), respectively, except that the origin is at
    (0.5, 0.5). The conversion is ``(x, y) = (j, i) + (0.5, 0.5)``. Integer
    values mark the pixel edges in the array system, whereas integer values
    mark the pixel centers in the pixel system.

    """

    def __init__(self, ijrange, label=None, nsub=50):
        self._ijrange = ijrange
        self.label = label
        self.nsub = nsub

    @property
    def imin(self):
        """Minimum i value of the aperture; read only."""
        return self._ijrange[0]

    @property
    def imax(self):
        """Maximum i value of the aperture; read only."""
        return self._ijrange[1]

    @property
    def jmin(self):
        """Minimum j value of the aperture; read only."""
        return self._ijrange[2]

    @property
    def jmax(self):
        """Maximum j value of the aperture; read only."""
        return self._ijrange[3]

    @property
    def ijslice(self):
        """Slice from `imin` to `imax` and `jmin` to `jmax`, inclusive;
        read only.

        """
        return (slice(self.imin, self.imax+1), slice(self.jmin, self.jmax+1))

    @property
    def ijextent(self):
        """Minimum and maximum i and j values of the smallest group of
        whole pixels containing the aperture; read only.

        """
        return (int(self.imin), int(self.imax)+1,
                int(self.jmin), int(self.jmax)+1)

    @property
    def xmin(self):
        """Minimum x value of the aperture; read only."""
        return self.jmin + 0.5

    @property
    def xmax(self):
        """Maximum x value of the aperture; read only."""
        return self.jmax + 0.5

    @property
    def ymin(self):
        """Minimum y value of the aperture; read only."""
        return self.imin + 0.5

    @property
    def ymax(self):
        """Maximum y value of the aperture; read only."""
        return self.imax + 0.5

    @property
    def xyextent(self):
        """Minimum and maximum x and y values of the smallest group of
        whole pixels containing the aperture; read only.

        """
        return tuple(n + 0.5 for n in np.roll(self.ijextent, 2))

    @property
    def centers(self):
        """i and j coordinates of pixel centers in the grid defined by
        `ijextent`. The i and j arrays constitute a sparse or open grid,
        which can be broadcast together to form the full grid. Read only.

        """
        imin, imax, jmin, jmax = self.ijextent
        i = np.arange(imin, imax).reshape(-1, 1) + 0.5
        j = np.arange(jmin, jmax).reshape(1, -1) + 0.5
        return i, j


class CircularAperture(ApertureBase):

    """A circular aperture.

    Subclass of `ApertureBase`. A circle is a special case of an ellipse,
    so many of the calculations are simpler. This class therefore performs
    a bit faster than an equivalent `EllipticalAperture` instance.

    Parameters
    ----------
    xy : 2-tuple
        Initialize the `xy` property.
    r : float
        Initialize the `r` property.
    label : string, optional
        Initialize the `ApertureBase.label` attribute. Default is None.
    nsub : int, optional
        Initialize the `ApertureBase.nsub` attribute. Default is 50.

    Attributes
    ----------
    i
    ij
    j
    r
    weights
    x
    xy
    y

    """

    def __init__(self, xy, r, label=None, nsub=50):
        i, j = xy[1] - 0.5, xy[0] - 0.5  # Convert into array coordinates
        self._parameters = [i, j, r]
        super(self.__class__, self).__init__(
            self._calc_ijrange(), label=label, nsub=nsub)

    @property
    def parameters(self):
        """`i`, `j`, and `r` properties of the aperture."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters
        self._ijrange = self._calc_ijrange()  # Update superclass

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
        return (self.i, self.j)

    @ij.setter
    def ij(self, ij):
        self.i, self.j = ij

    @property
    def x(self):
        """x coordinate of the center."""
        return self.j + 0.5

    @x.setter
    def x(self, x):
        self.j = x - 0.5

    @property
    def y(self):
        """y coordinate of the center."""
        return self.i + 0.5

    @y.setter
    def y(self, y):
        self.i = y - 0.5

    @property
    def xy(self):
        """x and y coordinates of the center."""
        return (self.x, self.y)

    @xy.setter
    def xy(self, xy):
        self.x, self.y = xy

    @property
    def r(self):
        """Radius."""
        return self._parameters[2]

    @r.setter
    def r(self, r):
        self._parameters[2] = r
        self.parameters = self._parameters

    @property
    def weights(self):
        """Fraction of each pixel's area, on the grid defined by
        `ApertureBase.ijextent`, that is within the aperture. Areas for
        partial pixels along the aperture border are approximated by
        sampling each border pixel with ``ApertureBase.nsub**2`` subpixels.
        Partial pixels are not computed if `ApertureBase.nsub` is 1.

        """
        # Distances from the aperture center
        i, j = self.centers
        r = np.sqrt((i - self.i)**2 + (j - self.j)**2)

        # Pixels with centers within the aperture
        weights = (r <= self.r).astype('float')

        # Partial pixels
        if self.nsub > 1:
            # Indices (lower-left corners wrt the grid) of border pixels
            i, j = np.where(np.abs(r - self.r) <= np.sqrt(0.5))

            # Centers of subpixels in generic subpixel grid
            isub = (np.arange(self.nsub).reshape(-1, 1) + 0.5) / self.nsub
            jsub = (np.arange(self.nsub).reshape(1, -1) + 0.5) / self.nsub

            # Centers of subpixels
            imin, jmin = self.ijextent[0::2]
            isub = i[:,None,None] + isub + imin  # (len(i), nsub, 1)
            jsub = j[:,None,None] + jsub + jmin  # (len(i), 1, nsub)

            # Distances from aperture center; (len(i), nsub, nsub)
            rsub = np.sqrt((isub - self.i)**2 + (jsub - self.j)**2)

            # Refined pixel weights
            kwargs = dict(axis=(1, 2), dtype='float')
            weights[i,j] = np.sum(rsub <= self.r, **kwargs) / self.nsub**2

        return weights

    def _calc_ijrange(self):
        i, j, r = self.parameters
        imin, imax = i - r, i + r
        jmin, jmax = j - r, j + r
        return (imin, imax, jmin, jmax)



#############


class EllipticalAperture(object):

    """An elliptical aperture.

    Parameters
    ----------
    xy : 2-tuple
        Initialize the `xy` property. Indirectly initializes all other x,
        y, i, and j coordinate-related attributes and properties.
    a : float
        Initialize the `a` attribute.
    b : float
        Initialize the `b` attribute.
    theta : float
        Initialize the `theta` attribute.
    label : string, optional
        Initialize the `label` attribute. Default is None.
    nsub : int, optional
        Initialize the `nsub` attribute. Default is 50.

    Attributes
    ----------
    centers
    ij
    ijextent
    ijslice
    imax
    imin
    jmax
    jmin
    weights
    x
    xmax
    xmin
    xy
    xyextent
    y
    ymax
    ymin
    a : float
        Semimajor axis.
    b : float
        Semiminor axis.
    i : float
        i coordinate of the center.
    j : float
        j coordinate of the center.
    label : string
        Label.
    nsub : int
        Number of subpixels (per side) in which to sample each border pixel
        when calculating weights for partial pixels in the aperture. The
        total number of subpixel samples for a border pixel is ``nsub**2``.
        Small apertures require larger `nsub` values to maintain accuracy,
        while smaller `nsub` values will suffice for large apertures. If 1,
        then partial pixels are not computed.
    theta : float
        Counter clockwise rotation angle in degrees.

    Notes
    -----
    i and j denote array coordinates, measuring along the rows and columns,
    respectively, where the origin (0, 0) is the outside corner of the
    first row and column. x and y denote pixel coordinates, corresponding
    to j (columns) and i (rows), respectively, except that the origin is at
    (0.5, 0.5). The conversion is ``(x, y) = (j, i) + (0.5, 0.5)``. Integer
    values mark the pixel edges in the array system, whereas integer values
    mark the pixel centers in the pixel system.

    General ellipse::

      (dx*cos(t) + dy*sin(t))**2 / a**2 + (-dx*sin(t) + dy*cos(t))**2 / b**2 = 1
      dx = x - x0
      dy = y - y0

    Solving for x and y, ::

      x = (-Bx +/- sqrt(Bx**2 - 4*Ax*Cx)) / (2*Ax)
      Ax = cos(t)**2 / a**2 + sin(t)**2 / b**2
      Bx = 2 * dy * sin(t) * cos(t) * (1/a**2 - 1/b**2)
      Cx = dy**2 * (sin(t)**2 / a**2 + cos(t)**2 / b**2) - 1

      y = (-By +/- sqrt(By**2 - 4*Ay*Cy)) / (2*Ay)
      Ay = sin(t)**2 / a**2 + cos(t)**2 / b**2
      By = 2 * dx * sin(t) * cos(t) * (1/a**2 - 1/b**2)
      Cy = dx**2 * (cos(t)**2 / a**2 + sin(t)**2 / b**2) - 1

    x equals xmin or xmax where the positive and negative y solutions are equal::

      => sqrt(By**2 - 4*Ay*Cy) = -sqrt(By**2 - 4*Ay*Cy)
      => By**2 - 4*Ay*Cy = 0

    After some algebra[1]_ (and repeating the same steps for x(y)), ::

      xmin = -sqrt(a**2 * cos(t)**2 + b**2 * sin(t)**2) + x0
      xmax = sqrt(a**2 * cos(t)**2 + b**2 * sin(t)**2) + x0

      ymin = -sqrt(a**2 * sin(t)**2 + b**2 * cos(t)**2) + y0
      ymax = sqrt(a**2 * sin(t)**2 + b**2 * cos(t)**2) + y0

    .. [1] Start with By**2 = 4*Ay*Cy, the move dx to the left side and
       multiply both sides by a**2*b**2::

         A = a**2
         B = b**2
         C = cos(t)**2
         S = sin(t)**2
         dx**2 * ((A*C+B*S)*(A*S+B*C) - S*C*(A-B)**2) = A*B**2*S + A**2*B*C

       The left side simplifies to dx**2*A*B.

    """

    def __init__(self, xy, a, b, theta, label=None, nsub=50):
        self.label = label
        self.i, self.j = None, None  # Create the i and j attributes...
        self.xy = xy  # ... then set them indirectly
        self.a = a
        self.b = b
        self.theta = theta
        self.nsub = nsub

    @property
    def ij(self):
        """i and j coordinates of the center."""
        return (self.i, self.j)

    @ij.setter
    def ij(self, ij):
        self.i, self.j = ij

    @property
    def imin(self):
        """Minimum i value of the aperture; read only."""
        t = self.theta * np.pi/180
        return self.i - np.sqrt(self.a**2 * np.sin(t)**2 +
                                self.b**2 * np.cos(t)**2)

    @property
    def imax(self):
        """Maximum i value of the aperture; read only."""
        t = self.theta * np.pi/180
        return self.i + np.sqrt(self.a**2 * np.sin(t)**2 +
                                self.b**2 * np.cos(t)**2)

    @property
    def jmin(self):
        """Minimum j value of the aperture; read only."""
        t = self.theta * np.pi/180
        return self.j - np.sqrt(self.a**2 * np.cos(t)**2 +
                                self.b**2 * np.sin(t)**2)

    @property
    def jmax(self):
        """Maximum j value of the aperture; read only."""
        t = self.theta * np.pi/180
        return self.j + np.sqrt(self.a**2 * np.cos(t)**2 +
                                self.b**2 * np.sin(t)**2)

    @property
    def ijslice(self):
        """Slice from `imin` to `imax` and `jmin` to `jmax`, inclusive;
        read only.

        """
        return (slice(self.imin, self.imax+1), slice(self.jmin, self.jmax+1))

    @property
    def ijextent(self):
        """Minimum and maximum i and j values of the smallest group of
        whole pixels containing the aperture; read only.

        """
        return (int(self.imin), int(self.imax)+1,
                int(self.jmin), int(self.jmax)+1)

    @property
    def x(self):
        """x coordinate of the center."""
        return self.j + 0.5

    @x.setter
    def x(self, x):
        self.j = x - 0.5

    @property
    def y(self):
        """y coordinate of the center."""
        return self.i + 0.5

    @y.setter
    def y(self, y):
        self.i = y - 0.5

    @property
    def xy(self):
        """x and y coordinates of the center."""
        return (self.x, self.y)

    @xy.setter
    def xy(self, xy):
        self.x, self.y = xy

    @property
    def xmin(self):
        """Minimum x value of the aperture; read only."""
        return self.jmin + 0.5

    @property
    def xmax(self):
        """Maximum x value of the aperture; read only."""
        return self.jmax + 0.5

    @property
    def ymin(self):
        """Minimum y value of the aperture; read only."""
        return self.imin + 0.5

    @property
    def ymax(self):
        """Maximum y value of the aperture; read only."""
        return self.imax + 0.5

    @property
    def xyextent(self):
        """Minimum and maximum x and y values of the smallest group of
        whole pixels containing the aperture; read only.

        """
        return tuple(n + 0.5 for n in np.roll(self.ijextent, 2))

    @property
    def centers(self):
        """i and j coordinates of pixel centers in the grid defined by
        `ijextent`. The i and j arrays constitute a sparse or open grid,
        which can be broadcast together to form the full grid.

        """
        imin, imax, jmin, jmax = self.ijextent
        i = np.arange(imin, imax).reshape(-1, 1) + 0.5
        j = np.arange(jmin, jmax).reshape(1, -1) + 0.5
        return i, j

    @property
    def weights(self):
        """Fraction of each pixel's area, on the grid defined by
        `ijextent`, that is within the aperture. Areas for partial pixels
        along the aperture border are approximated by sampling each border
        pixel with ``nsub**2`` subpixels. Partial pixels are not computed
        if `nsub` is 1.

        """
        # Elliptical distances from the aperture center
        t = self.theta * np.pi/180
        i, j = self.centers
        di, dj = i - self.i, j - self.j
        r = ((dj * np.cos(t) + di * np.sin(t))**2 / self.a**2 +
             (-dj * np.sin(t) + di * np.cos(t))**2 / self.b**2)

        # Pixels with centers within the aperture
        weights = (r <= 1).astype('float')

        # Partial pixels
        if self.nsub > 1:
            # Indices (lower-left corners wrt the grid) of border pixels
            borderradius = np.sqrt(0.5)
            a1, b1 = self.a - borderradius, self.b - borderradius
            r1 = ((dj * np.cos(t) + di * np.sin(t))**2 / a1**2 +
                  (-dj * np.sin(t) + di * np.cos(t))**2 / b1**2)
            a2, b2 = self.a + borderradius, self.b + borderradius
            r2 = ((dj * np.cos(t) + di * np.sin(t))**2 / a2**2 +
                  (-dj * np.sin(t) + di * np.cos(t))**2 / b2**2)
            i, j = np.where((1 <= r1) & (r2 <= 1))

            # Centers of subpixels in generic subpixel grid
            isub = (np.arange(self.nsub).reshape(-1, 1) + 0.5) / self.nsub
            jsub = (np.arange(self.nsub).reshape(1, -1) + 0.5) / self.nsub

            # Centers of subpixels
            imin, jmin = self.ijextent[0::2]
            isub = i[:,None,None] + isub + imin  # (len(i), nsub, 1)
            jsub = j[:,None,None] + jsub + jmin  # (len(i), 1, nsub)

            # Elliptical distances from aperture center; (len(i), nsub, nsub)
            disub, djsub = isub - self.i, jsub - self.j
            rsub = ((djsub * np.cos(t) + disub * np.sin(t))**2 / self.a**2 +
                    (-djsub * np.sin(t) + disub * np.cos(t))**2 / self.b**2)

            # Refined pixel weights
            kwargs = dict(axis=(1, 2), dtype='float')
            weights[i,j] = np.sum(rsub <= 1, **kwargs) / self.nsub**2

        return weights


class _CircularAperture(object):

    """A circular aperture.

    Parameters
    ----------
    xy : 2-tuple
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
    ijrange
    ijslice
    ijextent
    xmin
    xmax
    ymin
    ymax
    xyrange
    xyextent
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
        return (self.i, self.j)

    @ij.setter
    def ij(self, ij):
        self.i, self.j = ij

    @property
    def x(self):
        """x coordinate of the center."""
        return self.j + 0.5

    @x.setter
    def x(self, x):
        self.j = x - 0.5

    @property
    def y(self):
        """y coordinate of the center."""
        return self.i + 0.5

    @y.setter
    def y(self, y):
        self.i = y - 0.5

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
    def ijrange(self):
        """Minimum and maximum i and j values of the aperture; read only."""
        return (self.imin, self.imax, self.jmin, self.jmax)

    @property
    def ijslice(self):
        """Slice from `imin` to `imax` and `jmin` to `jmax`, inclusive;
        read only.

        """
        return (slice(self.imin, self.imax+1), slice(self.jmin, self.jmax+1))

    @property
    def ijextent(self):
        """Minimum and maximum i and j values of the smallest group of
        whole pixels containing the aperture; read only.

        """
        return (int(self.imin), int(self.imax)+1,
                int(self.jmin), int(self.jmax)+1)

    @property
    def xmin(self):
        """Minimum x value of the aperture; read only."""
        return self.jmin + 0.5

    @property
    def xmax(self):
        """Maximum x value of the aperture; read only."""
        return self.jmax + 0.5

    @property
    def ymin(self):
        """Minimum y value of the aperture; read only."""
        return self.imin + 0.5

    @property
    def ymax(self):
        """Maximum y value of the aperture; read only."""
        return self.imax + 0.5

    @property
    def xyrange(self):
        """Minimum and maximum x and y values of the aperture; read only."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def xyextent(self):
        """Minimum and maximum x and y values of the smallest group of
        whole pixels containing the aperture; read only.

        """
        return tuple(n + 0.5 for n in np.roll(self.ijextent, 2))

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
            imin, imax, jmin, jmax = self.ijextent
            i = np.arange(imin, imax).reshape(-1, 1) + 0.5
            j = np.arange(jmin, jmax).reshape(1, -1) + 0.5

            # Distances from the aperture center
            r = np.sqrt((i - self.i)**2 + (j - self.j)**2)

            # Pixels with centers within r of aperture center
            weights = (r <= self.r).astype('float')

            # Partial pixels
            if self.nsub > 1:
                # Indices (lower-left corners, wrt section) of border pixels
                i, j = np.where(np.abs(r - self.r) <= np.sqrt(0.5))

                # Generic subpixel grid
                gridi = (np.arange(self.nsub).reshape(-1, 1) + 0.5) / self.nsub
                gridj = (np.arange(self.nsub).reshape(1, -1) + 0.5) / self.nsub

                # Centers of subpixels
                isub = gridi + i[:,None,None] + imin  # (len(i), nsub, 1)
                jsub = gridj + j[:,None,None] + jmin  # (len(i), 1, nsub)

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
        current radius, and 0, respectively. Fitting is only done on the
        current `section`, so for the best results the aperture size and
        position should be optimized so that the section contains as much
        of the source as possible without introducting too much confusion
        from other sources.

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
        imin, imax, jmin, jmax = self.ijextent
        i = np.arange(imin, imax).reshape(-1, 1) + 0.5
        j = np.arange(jmin, jmax).reshape(1, -1) + 0.5

        # Find the best fit parameters
        dz = self.section.ravel() - self.section.min()
        p0 = (dz.max(), self.j, self.i, self.r, self.r, 0.0)
        popt, pcov = scipy.optimize.curve_fit(func, (j, i), dz, p0=p0)
        popt[1], popt[2] = popt[1] + 0.5, popt[2] + 0.5  # Pixel coords
        return tuple(popt)

    def centroid(self, adjust=False, mode='com'):
        """Find the centroid of the source in the image section.

        The centroid is iteratively calculated using the image data in
        `section`. The aperture center is updated with each centroid value,
        causing `section` to be updated as well. Iteration continues until
        convergence.

        Parameters
        ----------
        adjust : bool, optional
            If True, set the aperture's center point to the centroid.
            Default is False.
        mode : {'com', '2dgauss', 'marginal'}, optional
            The method for computing the centroid. Default is 'com'.

            - 'com': Calculate the center of mass of `section`. More
              precise than 'marginal'. Not good as `2dgauss`, but faster.
            - '2dgauss': Fit a 2d Gaussian function to `section` using the
              `fit_gaussian` method. Most precise, but also the slowest.
            - 'marginal': Measure the peaks in the marginal distributions.
              This method cannot achieve subpixel precission.

        Returns
        -------
        2-tuple
            x and y pixel coordinates of the centroid of the source.

        """
        tol = 1e-8  # Tolerance in pixels

        if self.section is None:
            xy = (None, None)
        else:
            parameters_copy = self._parameters[:]
            i0, j0 = self.i + tol + 1, self.j + tol + 1

            if mode == 'marginal':
                while np.abs(self.i - i0) > tol or np.abs(self.j - j0) > tol:
                    # Pixel centers in array coordinates
                    imin, imax, jmin, jmax = self.ijextent
                    i = np.arange(imin, imax) + 0.5
                    j = np.arange(jmin, jmax) + 0.5

                    # Peaks of marginal distributions
                    sumi = np.sum(self.section, axis=1)
                    sumj = np.sum(self.section, axis=0)
                    i, j = i[sumi.argmax()], j[sumj.argmax()]

                    i0, j0 = self.ij
                    self.ij = i, j  # setter updates section

            elif mode == 'com':
                while np.abs(self.i - i0) > tol or np.abs(self.j - j0) > tol:
                    # Pixel centers in array coordinates
                    imin, imax, jmin, jmax = self.ijextent
                    i = np.arange(imin, imax).reshape(-1, 1) + 0.5
                    j = np.arange(jmin, jmax).reshape(1, -1) + 0.5

                    # Center of mass (pixel coordinates)
                    total = np.sum(self.section)
                    i = np.sum(self.section * i, dtype='float') / total
                    j = np.sum(self.section * j, dtype='float') / total

                    i0, j0 = self.ij
                    self.ij = i, j  # setter updates section

            elif mode == '2dgauss':
                while np.abs(self.i - i0) > tol or np.abs(self.j - j0) > tol:
                    amp, x, y, sigmax, sigmay, theta = self.fit_gaussian()

                    i0, j0 = self.ij
                    self.xy = x, y  # setter updates section

            xy = self.xy
            if not adjust:
                self.parameters = parameters_copy

        return xy


class _EllipticalAperture(object):

    """An elliptical aperture.

    Parameters
    ----------
    xy : 2-tuple
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

    Attributes
    ----------
    parameters
    i
    j
    ij
    x
    y
    xy
    a
    b
    theta
    imin
    imax
    jmin
    jmax
    ijrange
    ijslice
    ijextent
    xmin
    xmax
    ymin
    ymax
    xyrange
    xyextent
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

    Notes
    -----
    General ellipse::

      (dx*cos(t) + dy*sin(t))**2 / a**2 + (-dx*sin(t) + dy*cos(t))**2 / b**2 = 1
      dx = x - x0
      dy = y - y0

    Solving for x and y, ::

      x = (-Bx +/- sqrt(Bx**2 - 4*Ax*Cx)) / (2*Ax)
      Ax = cos(t)**2 / a**2 + sin(t)**2 / b**2
      Bx = 2 * dy * sin(t) * cos(t) * (1/a**2 - 1/b**2)
      Cx = dy**2 * (sin(t)**2 / a**2 + cos(t)**2 / b**2) - 1

      y = (-By +/- sqrt(By**2 - 4*Ay*Cy)) / (2*Ay)
      Ay = sin(t)**2 / a**2 + cos(t)**2 / b**2
      By = 2 * dx * sin(t) * cos(t) * (1/a**2 - 1/b**2)
      Cy = dx**2 * (cos(t)**2 / a**2 + sin(t)**2 / b**2) - 1

    x equals xmin or xmax where the positive and negative y solutions are equal::

      => sqrt(By**2 - 4*Ay*Cy) = -sqrt(By**2 - 4*Ay*Cy)
      => By**2 - 4*Ay*Cy = 0

    After some algebra[1]_ (and repeating the same steps for x(y)), ::

      xmin = -sqrt(a**2 * cos(t)**2 + b**2 * sin(t)**2) + x0
      xmax = sqrt(a**2 * cos(t)**2 + b**2 * sin(t)**2) + x0

      ymin = -sqrt(a**2 * sin(t)**2 + b**2 * cos(t)**2) + y0
      ymax = sqrt(a**2 * sin(t)**2 + b**2 * cos(t)**2) + y0

    .. [1] Start with By**2 = 4*Ay*Cy, the move dx to the left side and
       multiply both sides by a**2*b**2::

         A = a**2
         B = b**2
         C = cos(t)**2
         S = sin(t)**2
         dx**2 * ((A*C+B*S)*(A*S+B*C) - S*C*(A-B)**2) = A*B**2*S + A**2*B*C

       The left side simplifies to dx**2*A*B.

    """

    def __init__(self, xy, a, b, theta, image=None, label=None, nsub=50):
        i, j = xy[1] - 0.5, xy[0] - 0.5  # Convert into array coordinates
        self.parameters = [i, j, a, b, theta*np.pi/180]
        self.image = image
        self.label = label
        self.nsub = nsub

    @property
    def parameters(self):
        """i, j, a, b, and theta values of the aperture.

        See the `i`, `j`, `a`, `b`, and `theta` properties. Although
        `theta` is exposed in degrees, it is stored here in radians.

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
        return (self.i, self.j)

    @ij.setter
    def ij(self, ij):
        self.i, self.j = ij

    @property
    def x(self):
        """x coordinate of the center."""
        return self.j + 0.5

    @x.setter
    def x(self, x):
        self.j = x - 0.5

    @property
    def y(self):
        """y coordinate of the center."""
        return self.i + 0.5

    @y.setter
    def y(self, y):
        self.i = y - 0.5

    @property
    def xy(self):
        """x and y coordinates of the center."""
        return (self.x, self.y)

    @xy.setter
    def xy(self, xy):
        self.x, self.y = xy

    @property
    def a(self):
        """Semimajor axis of the ellipse."""
        return self._parameters[2]

    @a.setter
    def a(self, a):
        self._parameters[2] = a
        self.parameters = self._parameters

    @property
    def b(self):
        """Semiminor axis of the ellipse."""
        return self._parameters[3]

    @b.setter
    def b(self, b):
        self._parameters[3] = b
        self.parameters = self._parameters

    @property
    def theta(self):
        """Counter clockwise rotation angle in degrees."""
        return self._parameters[4] * 180/np.pi

    @theta.setter
    def theta(self, theta):
        self._parameters[4] = theta * np.pi/180
        self.parameters = self._parameters

    @property
    def imin(self):
        """Minimum i value of the aperture; read only."""
        t = self.theta * np.pi/180
        return self.i - np.sqrt(self.a**2 * np.sin(t)**2 +
                                self.b**2 * np.cos(t)**2)

    @property
    def imax(self):
        """Maximum i value of the aperture; read only."""
        t = self.theta * np.pi/180
        return self.i + np.sqrt(self.a**2 * np.sin(t)**2 +
                                self.b**2 * np.cos(t)**2)

    @property
    def jmin(self):
        """Minimum j value of the aperture; read only."""
        t = self.theta * np.pi/180
        return self.j - np.sqrt(self.a**2 * np.cos(t)**2 +
                                self.b**2 * np.sin(t)**2)

    @property
    def jmax(self):
        """Maximum j value of the aperture; read only."""
        t = self.theta * np.pi/180
        return self.j + np.sqrt(self.a**2 * np.cos(t)**2 +
                                self.b**2 * np.sin(t)**2)

    @property
    def ijrange(self):
        """Minimum and maximum i and j values of the aperture; read only."""
        return (self.imin, self.imax, self.jmin, self.jmax)

    @property
    def ijslice(self):
        """Slice from `imin` to `imax` and `jmin` to `jmax`, inclusive;
        read only.

        """
        return (slice(self.imin, self.imax+1), slice(self.jmin, self.jmax+1))

    @property
    def ijextent(self):
        """Minimum and maximum i and j values of the smallest group of
        whole pixels containing the aperture; read only.

        """
        return (int(self.imin), int(self.imax)+1,
                int(self.jmin), int(self.jmax)+1)

    @property
    def xmin(self):
        """Minimum x value of the aperture; read only."""
        return self.jmin + 0.5

    @property
    def xmax(self):
        """Maximum x value of the aperture; read only."""
        return self.jmax + 0.5

    @property
    def ymin(self):
        """Minimum y value of the aperture; read only."""
        return self.imin + 0.5

    @property
    def ymax(self):
        """Maximum y value of the aperture; read only."""
        return self.imax + 0.5

    @property
    def xyrange(self):
        """Minimum and maximum x and y values of the aperture; read only."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def xyextent(self):
        """Minimum and maximum x and y values of the smallest group of
        whole pixels containing the aperture; read only.

        """
        return tuple(n + 0.5 for n in np.roll(self.ijextent, 2))

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
            imin, imax, jmin, jmax = self.ijextent
            i = np.arange(imin, imax).reshape(-1, 1) + 0.5
            j = np.arange(jmin, jmax).reshape(1, -1) + 0.5

            # Elliptical distances from the aperture center
            t = self.theta * np.pi/180
            di, dj = i - self.i, j - self.j
            r = ((dj * np.cos(t) + di * np.sin(t))**2 / self.a**2 +
                 (-dj * np.sin(t) + di * np.cos(t))**2 / self.b**2)

            # Pixels with centers within the aperture
            weights = (r <= 1).astype('float')

            # Partial pixels
            if self.nsub > 1:
                # Indices (lower-left corners, wrt section) of border pixels
                borderradius = np.sqrt(0.5)
                a1, b1 = self.a - borderradius, self.b - borderradius
                r1 = ((dj * np.cos(t) + di * np.sin(t))**2 / a1**2 +
                      (-dj * np.sin(t) + di * np.cos(t))**2 / b1**2)
                a2, b2 = self.a + borderradius, self.b + borderradius
                r2 = ((dj * np.cos(t) + di * np.sin(t))**2 / a2**2 +
                      (-dj * np.sin(t) + di * np.cos(t))**2 / b2**2)
                i, j = np.where((1 <= r1) & (r2 <= 1))

                # Generic subpixel grid
                gridi = (np.arange(self.nsub).reshape(-1, 1) + 0.5) / self.nsub
                gridj = (np.arange(self.nsub).reshape(1, -1) + 0.5) / self.nsub

                # Centers of subpixels
                isub = gridi + i[:,None,None] + imin  # (len(i), nsub, 1)
                jsub = gridj + j[:,None,None] + jmin  # (len(i), 1, nsub)

                # Elliptical distances from aperture center; (len(i), nsub, nsub)
                disub, djsub = isub - self.i, jsub - self.j
                rsub = ((djsub * np.cos(t) + disub * np.sin(t))**2 / self.a**2 +
                        (-djsub * np.sin(t) + disub * np.cos(t))**2 / self.b**2)

                # Refined pixel weights
                kwargs = dict(axis=(1, 2), dtype='float')
                weights[i,j] = np.sum(rsub <= 1, **kwargs) / self.nsub**2

        return weights

    def fit_gaussian(self):
        """Fit a 2d Gaussian function to the source.

        Initial guesses for the amplitude, center, width, and rotation
        parameters are the difference between the maximum and minimum
        values in `section`, the current x and y of the aperture, the
        current radius, and 0, respectively. Fitting is only done on the
        current `section`, so for the best results the aperture size and
        position should be optimized so that the section contains as much
        of the source as possible without introducting too much confusion
        from other sources.

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
        imin, imax, jmin, jmax = self.ijextent
        i = np.arange(imin, imax).reshape(-1, 1) + 0.5
        j = np.arange(jmin, jmax).reshape(1, -1) + 0.5

        # Find the best fit parameters
        dz = self.section.ravel() - self.section.min()
        p0 = (dz.max(), self.j, self.i, self.a, self.b, 0.0)
        popt, pcov = scipy.optimize.curve_fit(func, (j, i), dz, p0=p0)
        popt[1], popt[2] = popt[1] + 0.5, popt[2] + 0.5  # Pixel coords
        return tuple(popt)

    def centroid(self, adjust=False, mode='com'):
        """Find the centroid of the source in the image section.

        The centroid is iteratively calculated using the image data in
        `section`. The aperture center is updated with each centroid value,
        causing `section` to be updated as well. Iteration continues until
        convergence.

        Parameters
        ----------
        adjust : bool, optional
            If True, set the aperture's center point to the centroid.
            Default is False.
        mode : {'com', '2dgauss', 'marginal'}, optional
            The method for computing the centroid. Default is 'com'.

            - 'com': Calculate the center of mass of `section`. More
              precise than 'marginal'. Not good as `2dgauss`, but faster.
            - '2dgauss': Fit a 2d Gaussian function to `section` using the
              `fit_gaussian` method. Most precise, but also the slowest.
            - 'marginal': Measure the peaks in the marginal distributions.
              This method cannot achieve subpixel precission.

        Returns
        -------
        2-tuple
            x and y pixel coordinates of the centroid of the source.

        """
        tol = 1e-8  # Tolerance in pixels

        if self.section is None:
            xy = (None, None)
        else:
            parameters_copy = self._parameters[:]
            i0, j0 = self.i + tol + 1, self.j + tol + 1

            if mode == 'marginal':
                while np.abs(self.i - i0) > tol or np.abs(self.j - j0) > tol:
                    # Pixel centers in array coordinates
                    imin, imax, jmin, jmax = self.ijextent
                    i = np.arange(imin, imax) + 0.5
                    j = np.arange(jmin, jmax) + 0.5

                    # Peaks of marginal distributions
                    sumi = np.sum(self.section, axis=1)
                    sumj = np.sum(self.section, axis=0)
                    i, j = i[sumi.argmax()], j[sumj.argmax()]

                    i0, j0 = self.ij
                    self.ij = i, j  # setter updates section

            elif mode == 'com':
                while np.abs(self.i - i0) > tol or np.abs(self.j - j0) > tol:
                    # Pixel centers in array coordinates
                    imin, imax, jmin, jmax = self.ijextent
                    i = np.arange(imin, imax).reshape(-1, 1) + 0.5
                    j = np.arange(jmin, jmax).reshape(1, -1) + 0.5

                    # Center of mass (pixel coordinates)
                    total = np.sum(self.section)
                    i = np.sum(self.section * i, dtype='float') / total
                    j = np.sum(self.section * j, dtype='float') / total

                    i0, j0 = self.ij
                    self.ij = i, j  # setter updates section

            elif mode == '2dgauss':
                while np.abs(self.i - i0) > tol or np.abs(self.j - j0) > tol:
                    amp, x, y, sigmax, sigmay, theta = self.fit_gaussian()

                    i0, j0 = self.ij
                    self.xy = x, y  # setter updates section

            xy = self.xy
            if not adjust:
                self.parameters = parameters_copy

        return xy






class PolygonAperture(object):

    """A polygon aperture.

    Parameters
    ----------
    xy : 2-tuple
        Initialize the `xy` property.
    image : 2d array, optional
        Initialize the `image` property. Default is None.
    label : string, optional
        Initialize the `label` attribute. Default is None.
    nsub : int, optional
        Initialize the `nsub` attribute. Default is 1.

    """

    ### Weights for PolygonAperture: use matplotlib paths and "points in path"
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
        Widths defined as the standard deviation in each direction.
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


def fit_gaussian(image, p0=None):
    """Fit a 2d Gaussian function to the source in an image.

    For the best results, `image` should be optimally cropped and panned so
    that it contains as much of the source as possible without introducing
    too much confusion from other sources.

    Parameters
    ----------
    image : 2d array
        Image data containing (ideally only) the source of interest.
    p0 : tuple, optional
        Initial guesses for the `amp`, `x0` and `y0` (in pixel
        coordinates), `sigmax` and `sigmay`, and `theta`. See
        `fit_gaussian` for the definitions of these parameters. Any of
        these may be None, in which case a default initial guess is used:

        - `amp`: the difference between the maximum and minimum values in
          `image`.
        - `x` and/or `y`: the center of `image`.
        - `sigmax` and/or `sigmay`: half the size of `image` in each
          direction.
        - `theta`: 0 degrees.

        If None (default), the default guess for each parameter is used.

    Returns
    -------
    tuple
        Best-fit 2d Gaussian parameters (see `fit_gaussian`): `amp`, `x`
        and `y` (in pixel coordinates), `sigmax` and `sigmay`, and `theta`.

    """
    def func(xy, amp, x0, y0, sigmax, sigmay, theta):
        f = gaussian2d(xy[0], xy[1], amp, x0, y0, sigmax, sigmay, theta)
        return f.ravel()

    if p0 is None:
        p0 = [None, None, None, None, None, None]
    p0 = list(p0)

    # Pixel centers in array coordinates
    i = np.arange(image.shape[0]).reshape(-1, 1) + 0.5
    j = np.arange(image.shape[1]).reshape(1, -1) + 0.5

    # Find the best fit parameters
    dz = image.ravel() - image.min()
    p0[0] = dz.max() if p0[0] is None else p0[0]
    p0[1] = image.shape[1]/2.0 if p0[1] is None else p0[1] - 0.5  # Array coords
    p0[2] = image.shape[0]/2.0 if p0[2] is None else p0[2] - 0.5  # Array coords
    p0[3] = image.shape[1]/2.0 if p0[3] is None else p0[3]
    p0[4] = image.shape[0]/2.0 if p0[4] is None else p0[4]
    p0[5] = 0.0 if p0[5] is None else p0[5]
    popt, pcov = scipy.optimize.curve_fit(func, (j, i), dz, p0=p0)
    popt[1], popt[2] = popt[1] + 0.5, popt[2] + 0.5  # Pixel coords
    return tuple(popt)


def centroid(image, box=None, mode='com'):
    """Find the centroid of the source in an image.

    Parameters
    ----------
    image : 2d array
        Image data containing (ideally only) the source of interest.
    box : tuple, optional
        ``(i, j, di, dj)``, the position and size of a box in `image`
        within which to calculate the centroid, where ``i`` and ``j`` are
        the array coordinates of the lower left corner. It is essentially a
        box-shaped aperture around the source, and is useful if `image`
        contains multiple sources. If given, the centroid will be
        calculated iteratively, with the box recentering on each new
        centroid value until convergence. If None (default), the centroid
        is calculated only once, and therefore `image` should be optimally
        cropped and panned so that it contains as much of the source as
        possible without introducing too much confusion from other sources. 
    mode : {'com', '2dgauss', 'marginal'}, optional
        The method for computing the centroid. Default is 'com'.

        - 'com': Center of mass. More precise than 'marginal'. Faster than
          '2dgauss', but not always as precise.
        - '2dgauss': Fit a 2d Gaussian function using `fit_gaussian`. Most
          precise, but also the slowest.
        - 'marginal': Measure the peaks in the x and y marginal
          distributions. This method cannot achieve subpixel precision.

    Returns
    -------
    2-tuple
        x and y pixel coordinates of the centroid of the source.

    """
    if box is None:
        maxit = 1
        i0, j0, di, dj = 0, 0, image.shape[0], image.shape[1]
    else:
        maxit = np.inf
        i0, j0, di, dj = [int(n) for n in box]

    tol = 1e-8  # Tolerance in pixels

    # Box center, loop setup
    i, j = i0 + di / 2.0, j0 + dj / 2.0
    i_prev, j_prev = i + tol + 1, j + tol + 1
    it = 0

    if mode == 'marginal':
        while (i - i_prev)**2 + (j - j_prev)**2 > tol**2 and it < maxit:
            # Update image section
            ijslice = (slice(i0, i0 + di), slice(j0, j0 + dj))
            section = image[ijslice]

            # Pixel centers in array coordinates
            i = np.arange(i0, ) + 0.5
            j = np.arange() + 0.5

            # Peaks of marginal distributions
            sumi = np.sum(image, axis=1)
            sumj = np.sum(image, axis=0)
            i, j = i[sumi.argmax()], j[sumj.argmax()]
            x, y = j + 0.5, i + 0.5

            it += 1

    elif mode == 'com':
        # Pixel centers in array coordinates
        i = np.arange(image.shape[0]).reshape(-1, 1) + 0.5
        j = np.arange(image.shape[1]).reshape(1, -1) + 0.5

        # Center of mass
        total = np.sum(image)
        i = np.sum(image * i, dtype='float') / total
        j = np.sum(image * j, dtype='float') / total
        x, y = j + 0.5, i + 0.5

    elif mode == '2dgauss':
        amp, x, y, sigmax, sigmay, theta = fit_gaussian()

    return (x, y)


def centroid(self, adjust=False, mode='com'):
    """Find the centroid of the source in the image section.

    The centroid is iteratively calculated using the image data in
    `section`. The aperture center is updated with each centroid value,
    causing `section` to be updated as well. Iteration continues until
    convergence.

    Parameters
    ----------
    adjust : bool, optional
        If True, set the aperture's center point to the centroid.
        Default is False.
    mode : {'com', '2dgauss', 'marginal'}, optional
        The method for computing the centroid. Default is 'com'.

        - 'com': Calculate the center of mass of `section`. More
          precise than 'marginal'. Not good as `2dgauss`, but faster.
        - '2dgauss': Fit a 2d Gaussian function to `section` using the
          `fit_gaussian` method. Most precise, but also the slowest.
        - 'marginal': Measure the peaks in the marginal distributions.
          This method cannot achieve subpixel precission.

    Returns
    -------
    2-tuple
        x and y pixel coordinates of the centroid of the source.

    """
    tol = 1e-8  # Tolerance in pixels

    if self.section is None:
        xy = (None, None)
    else:
        parameters_copy = self._parameters[:]
        i0, j0 = self.i + tol + 1, self.j + tol + 1

        if mode == 'marginal':
            while np.abs(self.i - i0) > tol or np.abs(self.j - j0) > tol:
                # Pixel centers in array coordinates
                imin, imax, jmin, jmax = self.ijextent
                i = np.arange(imin, imax) + 0.5
                j = np.arange(jmin, jmax) + 0.5

                # Peaks of marginal distributions
                sumi = np.sum(self.section, axis=1)
                sumj = np.sum(self.section, axis=0)
                i, j = i[sumi.argmax()], j[sumj.argmax()]

                i0, j0 = self.ij
                self.ij = i, j  # setter updates section

        elif mode == 'com':
            while np.abs(self.i - i0) > tol or np.abs(self.j - j0) > tol:
                # Pixel centers in array coordinates
                imin, imax, jmin, jmax = self.ijextent
                i = np.arange(imin, imax).reshape(-1, 1) + 0.5
                j = np.arange(jmin, jmax).reshape(1, -1) + 0.5

                # Center of mass (pixel coordinates)
                total = np.sum(self.section)
                i = np.sum(self.section * i, dtype='float') / total
                j = np.sum(self.section * j, dtype='float') / total

                i0, j0 = self.ij
                self.ij = i, j  # setter updates section

        elif mode == '2dgauss':
            while np.abs(self.i - i0) > tol or np.abs(self.j - j0) > tol:
                amp, x, y, sigmax, sigmay, theta = self.fit_gaussian()

                i0, j0 = self.ij
                self.xy = x, y  # setter updates section

        xy = self.xy
        if not adjust:
            self.parameters = parameters_copy

    return xy
