"""

============
`galaxyphot`
============

Utilities for galaxy photometry.


Classes
-------

==================== ===================================
`ApertureBase`       Base class for aperture subclasses.
`CircularAperture`   A circular aperture.
`EllipticalAperture` An elliptical aperture.
`PolygonAperture`    A polygon aperture.
==================== ===================================


Functions
---------

================== ===================================================
`extract`          Extract image data for the given aperture.
`find_centroid`    Find the centroid of a source in an image.
`fit_gaussian`     Fit a 2d Gaussian function to a source in an image.
`from_region_file` Create apertures from a DS9 region file.
`gaussian2d`       2d Gaussian function.
================== ===================================================

"""
try:
    import astropy.table
except ImportError:
    astropy_installed = False
else:
    astropy_installed = True
import matplotlib.path
import numpy as np
import scipy.optimize



### apertures

class ApertureBase(object):

    """Base class for aperture subclasses.

    Attributes and properties common to all aperture classes are defined
    here.

    Parameters
    ----------
    limits : iterable, optional
        Initialize the `limits` property. Default is None. (See the note on
        subclassing in the Notes section for more details about this
        keyword.)
    label : str, optional
        Initialize the `label` attribute. Default is None.
    nsub : int, optional
        Initialize the `nsub` attribute. Default is 50.

    Attributes
    ----------
    grid
    extent
    ijextent
    ijlimits
    limits
    label : str
        Label.
    nsub : int
        Number of subpixels (per side) in which to sample each border pixel
        when calculating weights for partial pixels in the aperture. The
        total number of subpixel samples for a border pixel is ``nsub**2``.
        Small apertures require larger `nsub` values to maintain accuracy,
        while smaller `nsub` values will suffice for large apertures.

    Notes
    -----
    Coordinate systems for pixel arrays: i and j denote array coordinates,
    measuring along the rows and columns, respectively, where the origin
    (0, 0) is the outside corner of the first row and column. x and y
    denote pixel coordinates, corresponding to j and i, respectively,
    except that the origin is at (0.5, 0.5). The conversion is ``(x, y) =
    (j, i) + (0.5, 0.5)``. Integer values mark the pixel edges in the array
    system, whereas integer values mark the pixel centers in the pixel
    system.

    Subclassing: Many properties are derived either directly or indirectly
    from the `limits` property, which, internally, accesses a `_limits`
    attribute. This attribute may be set in the `__init__` method by
    specifying the `limits` keyword argument. By default, however, the
    `limits` keyword is None and no internal `_limits` attribute is
    created. This allows `ApertureBase` to be used by a subclass which
    defines its own `_limits` attribute.

    """

    def __init__(self, limits=None, label=None, nsub=50):
        if limits:
            self._limits = tuple(limits)
        self.label = label
        self.nsub = nsub

    @property
    def limits(self):
        """Minimum and maximum x and y values of the aperture, (xmin, xmax,
        ymin, ymax), defining the aperture's minimum bounding box; read
        only.

        """
        return self._limits

    @property
    def ijlimits(self):
        """Minimum and maximum i and j values of the aperture (imin, imax,
        jmin, jmax), defining the aperture's minimum bounding box; read
        only.

        """
        xmin, xmax, ymin, ymax = self.limits
        return (ymin-0.5, ymax-0.5, xmin-0.5, xmax-0.5)

    @property
    def ijextent(self):
        """Minimum and maximum i and j values of the smallest group of
        whole pixels containing the aperture (`ijlimits` rounded to the
        outside pixel edges). Defines an MxN grid, where there are M rows
        between ``extent[2]`` and ``extent[3]`` and N columns between
        ``extent[0]`` and ``extent[1]``. Read only.

        """
        imin, imax, jmin, jmax = self.ijlimits
        imin = int(imin) - 1 if imin < 0 else int(imin)
        imax = int(imax) if imax < 0 else int(imax) + 1
        jmin = int(jmin) - 1 if jmin < 0 else int(jmin)
        jmax = int(jmax) if jmax < 0 else int(jmax) + 1
        return (imin, imax, jmin, jmax)

    @property
    def extent(self):
        """Minimum and maximum x and y values of the smallest group of
        whole pixels containing the aperture (`limits` rounded to the
        outside pixel edges). Defines an MxN grid, where there are M rows
        between ``extent[2]`` and ``extent[3]`` and N columns between
        ``extent[0]`` and ``extent[1]``. Read only.

        """
        # Derive from ijextent because the pixel edges are easier to
        # calculate in ij space.
        imin, imax, jmin, jmax = self.ijextent
        return (jmin+0.5, jmax+0.5, imin+0.5, imax+0.5)

    @property
    def grid(self):
        """(1,N) and (M,1) arrays containing the x and y coordinates of
        pixel centers in the MxN grid defined by `extent`. The x and y
        arrays constitute a sparse or open grid, which can be broadcast
        together to form the full grid. Read only.

        """
        imin, imax, jmin, jmax = self.ijextent
        x = np.arange(jmin, jmax).reshape(1, -1) + 1.0
        y = np.arange(imin, imax).reshape(-1, 1) + 1.0
        return (x, y)

    def extract(self, image):
        return extract(image, self)


class CircularAperture(ApertureBase):

    """A circular aperture.

    Subclass of `ApertureBase`.

    Parameters
    ----------
    xy0 : (2,) array_like
        Initialize the `xy0` property.
    r : float
        Initialize the `r` property.
    label : str, optional
        Initialize the `ApertureBase.label` attribute. Default is None.
    nsub : int, optional
        Initialize the `ApertureBase.nsub` attribute. Default is 50.

    Attributes
    ----------
    r
    weights
    xy0

    Methods
    -------
    copy

    Notes
    -----
    A circle is a special case of an ellipse, so many of the calculations
    are simpler. This class therefore performs a bit faster than an
    equivalent `EllipticalAperture` instance.

    """

    def __init__(self, xy0, r, label=None, nsub=50):
        self.xy0 = xy0
        self.r = r
        super(self.__class__, self).__init__(label=label, nsub=nsub)

    @property
    def xy0(self):
        """(2,) ndarray. x and y pixel coordinates of the center."""
        return self._xy0

    @xy0.setter
    def xy0(self, xy0):
        xy0 = np.array(xy0)
        if xy0.shape != (2,):
            raise ValueError('xy0 is not (2,) array_like')
        self._xy0 = xy0

    @property
    def r(self):
        """float. Radius in pixels."""
        return self._r

    @r.setter
    def r(self, r):
        self._r = float(r)

    @property
    def _limits(self):
        # Required by ApertureBase
        x0, y0 = self.xy0
        r = self.r
        xmin, xmax = x0 - r, x0 + r
        ymin, ymax = y0 - r, y0 + r
        return (xmin, xmax, ymin, ymax)

    @property
    def weights(self):
        """(M,N) ndarray. Fractions of pixel areas within the aperture for
        the MxN grid defined by `ApertureBase.extent`. Areas for partial
        pixels along the aperture border are approximated by dividing each
        border pixel into ``ApertureBase.nsub**2`` subpixels. Partial
        pixels are not computed if `ApertureBase.nsub` is 1. Read only.

        """
        # Distances of pixel centers from the aperture center
        x0, y0 = self.xy0
        x, y = self.grid
        r0 = self.r
        r = np.sqrt((x - x0)**2 + (y - y0)**2)

        # Pixels with centers within the aperture
        weights = (r <= r0).astype('float')

        # Partial pixels
        if self.nsub > 1:
            # Lower-left corners of border pixels
            borderradius = np.sqrt(0.5)
            i, j = np.where(np.absolute(r - self.r) <= borderradius)
            xmin, ymin = self.extent[0::2]
            x, y = xmin + j, ymin + i

            # Centers of subpixels with respect to a generic pixel
            nsub = self.nsub
            subx = (np.arange(nsub).reshape(1, -1) + 0.5) / nsub
            suby = (np.arange(nsub).reshape(-1, 1) + 0.5) / nsub

            # Centers of subpixels in all border pixels
            subx = x[:,None,None] + subx  # (len(i), 1, nsub)
            suby = y[:,None,None] + suby  # (len(i), nsub, 1)

            # Distances from aperture center; (len(i), nsub, nsub)
            subr = np.sqrt((subx - x0)**2 + (suby - y0)**2)

            # Refined pixel weights along the border
            kwargs = dict(axis=(1, 2), dtype='float')
            weights[i,j] = np.sum(subr <= r0, **kwargs) / nsub**2

        return weights

    def copy(self):
        """Return a copy of the aperture."""
        xy0 = self.xy0.copy()
        r = self.r
        label = self.label
        nsub = self.nsub
        return CircularAperture(xy0, r, label=label, nsub=nsub)


class CircularAnnulus(object):

    """A ciruclar annulus aperture.

    """

    def __init__(self, xy0, r1, r2, label=None, nsub=50):
        # Which should be anchor?
        # Constraints on r1 and r2? e.g., 0 < r1 < r2
        self._outer = CircularAperture(xy0, r2, label=label, nsub=nsub)
        self._inner = CircularAperture(xy0, r1, label=label, nsub=nsub)

    @property
    def xy0(self):
        return


class EllipticalAperture(ApertureBase):

    """An elliptical aperture.

    Subclass of `ApertureBase`.

    Parameters
    ----------
    xy0 : (2,) array_like
        Initialize the `xy0` property.
    a : float
        Initialize the `a` property.
    b : float
        Initialize the `b` property.
    theta : float
        Initialize the `theta` property.
    label : string, optional
        Initialize the `ApertureBase.label` attribute. Default is None.
    nsub : int, optional
        Initialize the `ApertureBase.nsub` attribute. Default is 50.

    Attributes
    ----------
    a
    b
    theta
    weights
    xy0

    Methods
    -------
    copy

    Notes
    -----
    The general formula for an ellipse centered at ``(x0, y0)`` with
    semimajor axis ``a``, semiminor axis ``b``, and counter clockwise
    rotation angle ``t`` is::

      (( dx*cos(t) + dy*sin(t))**2 / a**2 +
       (-dx*sin(t) + dy*cos(t))**2 / b**2) = 1

      dx = x - x0
      dy = y - y0

    The x and y limits of the minimum bounding box are found by solving for
    x(y) and y(x)::

      x(y) = (-Bx +/- sqrt(Bx**2 - 4*Ax*Cx)) / (2*Ax)
      Ax = cos(t)**2 / a**2 + sin(t)**2 / b**2
      Bx = 2 * dy * sin(t) * cos(t) * (1/a**2 - 1/b**2)
      Cx = dy**2 * (sin(t)**2 / a**2 + cos(t)**2 / b**2) - 1

      y(x) = (-By +/- sqrt(By**2 - 4*Ay*Cy)) / (2*Ay)
      Ay = sin(t)**2 / a**2 + cos(t)**2 / b**2
      By = 2 * dx * sin(t) * cos(t) * (1/a**2 - 1/b**2)
      Cy = dx**2 * (cos(t)**2 / a**2 + sin(t)**2 / b**2) - 1

    x equals xmin or xmax where the positive and negative roots of y(x) are
    equal::

      sqrt(By**2 - 4*Ay*Cy) = -sqrt(By**2 - 4*Ay*Cy)
      => By**2 - 4*Ay*Cy = 0

    After some algebra [1]_ and repeating the same steps for x(y)::

      (xmin, xmax) = (-1, 1) * sqrt(a**2 * cos(t)**2 + b**2 * sin(t)**2) + x0
      (ymin, ymax) = (-1, 1) * sqrt(a**2 * sin(t)**2 + b**2 * cos(t)**2) + y0

    .. [1] Start with ``By**2 = 4*Ay*Cy``, then collect dx on the left side
       and multiply both sides by ``(a*b)**2``::

         A = a**2
         B = b**2
         C = cos(t)**2
         S = sin(t)**2
         dx**2 * ((A*C+B*S)*(A*S+B*C) - S*C*(A-B)**2) = A*B**2*S + A**2*B*C

       The left side simplifies to ``(dx*a*b)**2``, and the equation is
       easily solved for xmin and xmax.

    """

    def __init__(self, xy0, a, b, theta, label=None, nsub=50):
        self.xy0 = xy0
        self.a = a
        self.b = b
        self.theta = theta
        super(self.__class__, self).__init__(label=label, nsub=nsub)

    @property
    def xy0(self):
        """(2,) ndarray. x and y pixel coordinates of the center."""
        return self._xy0

    @xy0.setter
    def xy0(self, xy0):
        xy0 = np.array(xy0)
        if xy0.shape != (2,):
            raise ValueError('xy0 is not (2,) array_like')
        self._xy0 = xy0

    @property
    def a(self):
        """float. Semimajor axis in pixels."""
        return self._a

    @a.setter
    def a(self, a):
        self._a = float(a)

    @property
    def b(self):
        """float. Semiminor axis in pixels."""
        return self._b

    @b.setter
    def b(self, b):
        self._b = float(b)

    @property
    def theta(self):
        """float. Counter clockwise rotation angle in degrees."""
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = float(theta)

    @property
    def _limits(self):
        # Required by ApertureBase
        t = np.radians(self.theta)
        dx = np.sqrt(self.a**2 * np.cos(t)**2 + self.b**2 * np.sin(t)**2)
        dy = np.sqrt(self.a**2 * np.sin(t)**2 + self.b**2 * np.cos(t)**2)
        xmin, xmax = self.xy0[0] - dx, self.xy0[0] + dx
        ymin, ymax = self.xy0[1] - dy, self.xy0[1] + dy
        return (xmin, xmax, ymin, ymax)

    @property
    def weights(self):
        """(M,N) ndarray. Fractions of pixel areas within the aperture for
        the MxN grid defined by `ApertureBase.extent`. Areas for partial
        pixels along the aperture border are approximated by dividing each
        border pixel into ``ApertureBase.nsub**2`` subpixels. Partial
        pixels are not computed if `ApertureBase.nsub` is 1. Read only.

        """
        # Elliptical distances from the aperture center
        t = np.radians(self.theta)
        x, y = self.grid
        dx, dy = x - self.xy0[0], y - self.xy0[1]
        r = (( dx * np.cos(t) + dy * np.sin(t))**2 / self.a**2 +
             (-dx * np.sin(t) + dy * np.cos(t))**2 / self.b**2)

        # Pixels with centers within the aperture
        weights = (r <= 1).astype('float')

        # Partial pixels
        if self.nsub > 1:
            # Lower-left corners of border pixels
            borderradius = np.sqrt(0.5)
            amin, bmin = self.a - borderradius, self.b - borderradius
            rmin = (( dx * np.cos(t) + dy * np.sin(t))**2 / amin**2 +
                    (-dx * np.sin(t) + dy * np.cos(t))**2 / bmin**2)
            amax, bmax = self.a + borderradius, self.b + borderradius
            rmax = (( dx * np.cos(t) + dy * np.sin(t))**2 / amax**2 +
                    (-dx * np.sin(t) + dy * np.cos(t))**2 / bmax**2)
            i, j = np.where((1 <= rmin) & (rmax <= 1))
            x, y = self.extent[0] + j, self.extent[2] + i

            # Centers of subpixels with respect to a generic pixel
            subx = (np.arange(self.nsub).reshape(1, -1) + 0.5) / self.nsub
            suby = (np.arange(self.nsub).reshape(-1, 1) + 0.5) / self.nsub

            # Centers of subpixels in all border pixels
            subx = x[:,None,None] + subx  # (len(i), 1, nsub)
            suby = y[:,None,None] + suby  # (len(i), nsub, 1)

            # Elliptical distances from aperture center; (len(i), nsub, nsub)
            subdx, subdy = subx - self.xy0[0], suby - self.xy0[1]
            subr = (( subdx * np.cos(t) + subdy * np.sin(t))**2 / self.a**2 +
                    (-subdx * np.sin(t) + subdy * np.cos(t))**2 / self.b**2)

            # Refined pixel weights along the border
            kwargs = dict(axis=(1, 2), dtype='float')
            weights[i,j] = np.sum(subr <= 1, **kwargs) / self.nsub**2

        return weights

    def copy(self):
        """Return a copy of the aperture."""
        xy0 = self.xy0.copy()
        a, b = self.a, self.b
        theta = self.theta
        label = self.label
        nsub = self.nsub
        return EllipticalAperture(xy0, a, b, theta, label=label, nsub=nsub)


class EllipticalAnnulus(object):

    """An elliptical annulus aperture.

    """

    def __init__(self, xy0, a1, b1, a2, b2, theta, label=None, nsub=50):
        # Annulus is completely described by three axis parameters; the
        # fourth is derived. If all are given, ignore b2 for consistency
        if not None in (a1, b1, a2, b2): b2 = None
        if a1 is None:
            a1 = a2 * b1 / b2
        elif b1 is None:
            b1 = b2 * a1 / a2
        elif a2 is None:
            a2 = a1 * b2 / b1
        elif b2 is None:
            b2 = b1 * a2 / a1
        pass


class PolygonAperture(ApertureBase):

    """A polygon aperture.

    Subclass of `ApertureBase`.

    Parameters
    ----------
    xy : (k,2) or (k+1,2) array_like
        Initialize the `xy` property.
    label : string, optional
        Initialize the `ApertureBase.label` attribute. Default is None.
    nsub : int, optional
        Initialize the `ApertureBase.nsub` attribute. Default is 50.

    Attributes
    ----------
    area
    path
    weights
    xy
    xy0

    Methods
    -------
    copy

    """

    def __init__(self, xy, label=None, nsub=50):
        self.xy = xy
        super(self.__class__, self).__init__(label=label, nsub=nsub)

    @property
    def xy(self):
        """(k,2) ndarray. x and y pixel coordinates of the vertices. The
        last vertex is automatically truncated if set from a sequence in
        which the first and last vertices are equal (k+1 vertices).

        """
        return self._xy[:-1]

    @xy.setter
    def xy(self, xy):
        if np.all(xy[0] == xy[-1]):  # xy is closed
            xy = np.array(xy)
        else:  # xy is open
            xy = np.vstack((xy, xy[0]))
        if xy.shape != (len(xy), 2):
            raise ValueError('xy is not (k,2) array_like')
        self._xy = xy
        self._path = matplotlib.path.Path(self._xy, closed=True)  # Update path

    @property
    def _area_segments(self):
        # Used for area and centroid calculations
        return (self._xy[:-1,0] * self._xy[1:,1] -
                self._xy[1:,0] * self._xy[:-1,1])

    @property
    def area(self):
        """float. Area of the aperture in pix**2; read only."""
        return 0.5 * np.sum(self._area_segments)

    @property
    def xy0(self):
        """(2,) ndarray. x and y pixel coordinates of the centroid. Setting
        this property shifts the polygon vertices so that the centroid is
        at the given location.

        """
        xy0 = np.sum((self._xy[:-1] + self._xy[1:]) *
                     self._area_segments[:,None], axis=0) / 6 / self.area
        return xy0

    @xy0.setter
    def xy0(self, xy0):
        xy0 = np.array(xy0)
        if xy0.shape != (2,):
            raise ValueError('xy0 is not (2,) array_like')
        self._xy += xy0 - self.xy0

    @property
    def _limits(self):
        xmin, ymin = np.amin(self.xy, axis=0)
        xmax, ymax = np.amax(self.xy, axis=0)
        return (xmin, xmax, ymin, ymax)

    @property
    def path(self):
        """matplotlib.path.Path. Path representation of the aperture. The
        underlying ``path.vertices`` array is a reference to `xy`, i.e.,
        the path automatically updates when the aperture changes. A new
        path instance is created whenever `xy` is set explicitly. Read
        only.

        """
        return self._path

    @property
    def weights(self):
        """(M,N) ndarray. Fractions of pixel areas within the aperture for
        the MxN grid defined by `ApertureBase.extent`. Areas for partial
        pixels along the aperture border are approximated by dividing each
        border pixel into ``ApertureBase.nsub**2`` subpixels. Partial
        pixels are not computed if `ApertureBase.nsub` is 1. Read only.

        """
        # Pixels with centers within the aperture
        x, y = self.grid
        pix = np.vstack(np.broadcast_arrays(x, y)).reshape(2, -1).T
        weights = (self.path.contains_points(pix)
                   .reshape(y.size, x.size).astype('float'))

        # Partial pixels
        if self.nsub > 1:
            # Shift the aperture to determine the border pixels.
            #
            # Ideally, additional contains tests would only have to be
            # performed for a dialated and an eroded version of the
            # aperture. Implementing a buffering algorithm for arbitrary
            # polygons is not easy, however, and I don't want to add
            # Shapely as a dependency for this module. Instead, the method
            # below shifts the aperture in eight directions and records the
            # changes in the pixels' coverage. The code is much simpler,
            # but performing eight contains tests instead of two means that
            # the code is also much slower.
            shift_size = 0.5
            shift_list = np.array(
                [[1, 0], [0, 1], [-1, 0], [-1, 0],
                 [0, -1], [0, -1], [1, 0], [1, 0]]) * shift_size
            w = weights.ravel().astype('bool')
            bordertest = np.zeros(w.shape, dtype='bool')
            xy = self._xy.copy()
            path = matplotlib.path.Path(xy, closed=True)
            for shift in shift_list:
                path.vertices += shift
                bordertest += bordertest | w != path.contains_points(pix)
            bordertest = bordertest.reshape(y.size, x.size).astype('float')

            # Lower-left corners of border pixels
            i, j = np.where(bordertest)
            x, y = self.extent[0] + j, self.extent[2] + i

            # Centers of subpixels with respect to a generic pixel
            subx = (np.arange(self.nsub).reshape(1, -1) + 0.5) / self.nsub
            suby = (np.arange(self.nsub).reshape(-1, 1) + 0.5) / self.nsub

            # Centers of subpixels in all border pixels
            subx = x[:,None,None] + subx  # (len(i), 1, nsub)
            suby = y[:,None,None] + suby  # (len(i), nsub, 1)

            # Subpixels with centers within the aperture
            subpix = np.vstack(np.broadcast_arrays(subx, suby)).reshape(2, -1).T
            subweights = (self.path.contains_points(subpix)
                          .reshape(np.broadcast(subx, suby).shape))

            # Refined pixel weights along the border
            kwargs = dict(axis=(1, 2), dtype='float')
            weights[i, j] = np.sum(subweights, **kwargs) / self.nsub**2

        return weights

    def copy(self):
        """Return a copy of the aperture."""
        xy = self.xy.copy()
        label = self.label
        nsub = self.nsub
        return PolygonAperture(xy, label=label, nsub=nsub)


def extract(image, aperture):
    """Extract image data for the given aperture.

    Parameters
    ----------
    image : 2d ndarray
        Image data.
    aperture : ApertureBase
        The aperture.

    Returns
    -------
    2d ndarray
        Image data for the minimum bounding box of the aperture.
        `numpy.nan` where the aperture is outside of the image.

    """
    # Aperture and image limits
    imin, imax, jmin, jmax = aperture.ijextent
    ilim, jlim = image.shape

    if 0 <= imin and imax <= ilim and 0 <= jmin and jmax <= jlim:
        # Aperture is contained within image
        image_slice = (slice(imin, imax), slice(jmin, jmax))
        data = image[image_slice]

    else:
        # Aperture is not contained within image
        data = np.zeros((imax-imin, jmax-jmin)) * np.nan

        # Insert data where aperture and image intersect
        di, dj = -imin, -jmin  # Offset of image with respect to data
        imin, imax = max(imin, 0), min(imax, ilim)
        jmin, jmax = max(jmin, 0), min(jmax, jlim)
        if imin < ilim and 0 < imax and jmin < jlim and 0 < jmax:
            image_slice = (slice(imin, imax), slice(jmin, jmax))
            data_slice = (slice(imin+di, imax+di), slice(jmin+dj, jmax+dj))
            data[data_slice] = image[image_slice]

    return data


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
    filename : str
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






### util

def gaussian2d(x, y, amp, x0, y0, sigmax, sigmay, theta):
    """2d Gaussian function.

    Parameters
    ----------
    x, y : ndarray
        Input x and y coordinates. Both arrays must be either the same
        shape or broadcastable.
    amp : float
        Amplitude.
    x0, y0 : float
        Location of the center.
    sigmax, sigmay : float
        Widths defined as the standard deviation in the x and y directions.
    theta : float
        Counter clockwise rotation angle in degrees.

    Returns
    -------
    ndarray
        2d Gaussian function values, same shape as `x` and `y` or the
        result of their broadcasting.

    """
    theta = np.radians(theta)
    a = np.cos(theta)**2 / (2*sigmax**2) + np.sin(theta)**2 / (2*sigmay**2)
    b = np.sin(2*theta) / (4*sigmax**2) - np.sin(2*theta) / (4*sigmay**2)
    c = np.sin(theta)**2 / (2*sigmax**2) + np.cos(theta)**2 / (2*sigmay**2)
    f = amp * np.exp(-(a*(x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2))
    return f


def median(x, weights=None, thresh=None):
    """Calculate the median of a set of values.

    This is just `numpy.median`, with an option to calculate the weighted
    median.

    Parameters
    ----------
    x : ndarray
    weights : ndarray, optional
        Weights for the x values. If given, and if `thresh` is None, the
        weighted median is calculated by interpolating to half the
        cumulative weight between the minimum and maximum x values. If None
        (default), the median is calculated normally using `numpy.median`,
        which gives equal weight to all values.
    thresh : float, optional
        Threshold weight value. If `weights` is given, calculate the median
        using `numpy.median`, but only for x values with weights equal to
        or above the threshold. Ignored if `weights` is None. Default is
        None.

    Returns
    -------
    int or float

    """
    if x.size == 1:
        x_median = x[0]
    else:
        if weights is not None:
            if thresh is not None:
                x_median = np.median(x[thresh <= weights])
            else:
                x, weights = x.ravel(), weights.ravel()
                idx = np.argsort(x)
                x, weights = x[idx], weights[idx]
                cumulative = np.cumsum(weights) - weights[0]
                cumulative /= cumulative[-1]  # Normalize to 1
                x_median = np.interp(0.5, cumulative, x)
        else:
            x_median = np.median(x)
    return x_median




### main

def fit_gaussian(image, aperture, p0=None):
    """Fit a 2d Gaussian function to a source in an image.

    Parameters
    ----------
    image : 2d ndarray
        Image data.
    aperture : ApertureBase
        The aperture containing the source of interest [1]_.
    p0 : (6,) iterable, optional
        Initial guesses for the `amp`, `x0` and `y0` (in pixel
        coordinates), `sigmax` and `sigmay`, and `theta` (see
        `gaussian2d`). Any or all of these may be None, in which case the
        initial guess is based on the aperture. For `amp`, the default
        guess is the difference between the maximum and minimum values in
        the aperture's bounding box. Default is None.

    Returns
    -------
    (6,) tuple
        Best-fit 2d Gaussian parameters: `amp`, `x` and `y` (in pixel
        coordinates), `sigmax` and `sigmay`, and `theta`. See `gaussian2d`.

    Notes
    -----
    .. [1] The fitting is actually performed on all pixels in the
       aperture's bounding box defined by its `extent` property. For the
       best results, the aperture should be tuned so that the bounding box
       contains as much of the source as possible without introducting too
       much confusion from other sources.

    """
    def func(xy, amp, x0, y0, sigmax, sigmay, theta):
        f = gaussian2d(xy[0], xy[1], amp, x0, y0, sigmax, sigmay, theta)
        return f.ravel()

    # Initial guesses
    p0 = [None] * 6 if p0 is None else list(p0)

    data = aperture.extract(image)
    dz = data.ravel() - np.nanmin(data)
    p0[0] = np.nanmax(dz) if p0[0] is None else p0[0]

    if p0[1] is None:
        try:
            p0[1] = aperture.centroid[0]
        except AttributeError:
            p0[1] = aperture.xy0[0]

    if p0[2] is None:
        try:
            p0[2] = aperture.centroid[1]
        except AttributeError:
            p0[2] = aperture.xy0[1]

    if p0[3] is None:
        try:
            p0[3] = aperture.r
        except AttributeError:
            try:
                p0[3] = aperture.a
            except AttributeError:
                p0[3] = (aperture.limits[1] - aperture.limits[0]) / 2.0

    if p0[4] is None:
        try:
            p0[4] = aperture.r
        except AttributeError:
            try:
                p0[4] = aperture.b
            except AttributeError:
                p0[4] = (aperture.limits[3] - aperture.limits[2]) / 2.0

    if p0[5] is None:
        try:
            p0[5] = aperture.theta
        except AttributeError:
            p0[5] = 0.0

    if np.all(np.isnan(data)):  # Can't fit all nans, so return intial guesses
        popt = p0

    else:
        # Pixel centers in array coordinates
        x, y = aperture.grid

        # Find the best fit parameters
        popt, pcov = scipy.optimize.curve_fit(func, (x, y), dz, p0=p0)

    return tuple(popt)


def find_centroid(image, aperture, mode='com', adjust=True):
    """Find the centroid of a source in an image.

    The centroid is calculated iteratively by shifting the aperture to each
    new centroid position. Iteration continues until convergence.

    Parameters
    ----------
    image : 2d ndarray
        Image data.
    aperture : ApertureBase
        The aperture containing the source of interest [1]_.
    mode : {'com', '2dgauss', 'marginal'}, optional
        The method for computing the centroid. Default is 'com'.

        - 'com': Center of mass. More precise than 'marginal'. Faster than
          '2dgauss', but not always as precise.
        - '2dgauss': Fit a 2d Gaussian function using `fit_gaussian`. Most
          precise, but also the slowest.
        - 'marginal': Measure the peaks in the x and y marginal
          distributions. This method cannot achieve subpixel precision.

    adjust : bool, optional
        If True (default), the aperture's `xy0` attribute is set to the
        calculated centroid. Otherwise the aperture is unmodified.

    Returns
    -------
    (2,) ndarray
        x and y pixel coordinates of the centroid of the source.

    Notes
    -----
    .. [1] The centroid calculation is actually performed on all pixels in
       the aperture's bounding box defined by its `extent` property. For
       the best results, the aperture should be tuned so that its bounding
       box contains as much of the source as possible without introducting
       too much confusion from other sources.

    """
    def calc_centroid_marginal(image, aperture):
        # Pixel centers
        x, y = aperture.grid

        # Peaks of marginal distributions
        data = aperture.extract(image)
        xmarg, ymarg = np.sum(data, axis=0), np.sum(data, axis=1)
        if np.all(np.isnan(xmarg)):
            x0 = aperture.xy0[0]  # Return the original x value
        else:
            x0 = x[0,np.nanargmax(xmarg)]
        if np.all(np.isnan(ymarg)):
            y0 = aperture.xy0[1]  # Return the original y value
        else:
            y0 = y[np.nanargmax(ymarg),0]

        return np.array([x0, y0])

    def calc_centroid_com(image, aperture):
        # Pixel centers
        x, y = aperture.grid

        # Center of mass
        data = aperture.extract(image)
        total = np.sum(data)
        if np.isnan(total):  # Return the original x and y values
            x0 = aperture.xy0[0]
            y0 = aperture.xy0[1]
        else:
            x0 = np.sum(data * x, dtype='float') / total
            y0 = np.sum(data * y, dtype='float') / total

        return np.array([x0, y0])

    def calc_centroid_2dgauss(image, aperture):
            amp, x0, y0, sigmax, sigmay, theta = fit_gaussian(image, aperture)
            return np.array([x0, y0])

    func_dict = {
        'marginal': calc_centroid_marginal,
        'com': calc_centroid_com,
        '2dgauss': calc_centroid_2dgauss,
        }
    centroid_func = func_dict[mode]

    tol = 1e-8  # Tolerance in pixels

    aperture_copy = aperture.copy()
    xy0_prev = aperture_copy.xy0 + tol + 1
    while np.sum((aperture_copy.xy0 - xy0_prev)**2) > tol**2:
        xy0_prev = aperture_copy.xy0
        xy0 = centroid_func(image, aperture_copy)
        aperture_copy.xy0 = xy0
    xy0 = aperture_copy.xy0.copy()  # No lingering refs to the aperture copy

    if adjust:
        aperture.xy0 = xy0
    return xy0


def apphot(image, aperture_list, median_type='weighted', unbiased=False):
    """Perform aperture photometry on a CCD image.

    Parameters
    ----------
    image : 2d ndarray
        Image data.
    aperture_list : 1d iterable
        Collection of `ApertureBase` instances.
    median_type : {'weighted', float, None}, optional
        Method for calculating the median intensity value in the aperture.
        If 'weighted' (default) or a float, `median` is used in either
        "weighted median" or threshold mode. If None, the median is
        calculated normally from all pixels overlapping the aperture
        (partial pixels are treated as whole pixels).
    unbiased : bool, optional
        Calculate the (weighted) standard deviation using the unbiased
        form. Default is False.

    Returns
    -------
    astropy.table.Table or ndarray
        Photometry table, one row per aperture. An `ndarray` is returned if
        `astropy` is not available. See Notes for the columns.

    Notes
    -----
    The columns in the output table are,

    ====== ======== =======================================================
    column units    description
    ====== ======== =======================================================
    label           `label` attribute of the aperture.
    area   pix      Total area (number of pixels), including partial pixels
                    along the aperture border.
    total  flux     Total flux in the aperture.
    median flux/pix Median intensity.
    mean   flux/pix Mean intensity.
    std    flux/pix Standard deviation of the intensities.
    ====== ======== =======================================================

    The value of a CCD pixel represents a total flux. A pixel is a unit of
    area, so the value of a pixel is also an intensity (surface brightness,
    flux per area on the sky). The intensity of a pixel is constant, but
    the total flux it contributes to the aperture is determined by its
    in-aperture area::

        flux_i = area_i * intensity_i

    The total flux from all pixels in the aperture is::

        total_flux = sum(flux_i)

    The (area weighted) mean intensity in the aperture and the standard
    deviation are [1]_ ::

        mean_intensity = sum(area_i * intensity_i) / sum(area_i)
                       = sum(flux_i) / sum(area_i)
                       = total_flux / total_area

        std_intensity**2 = sum(area_i * (intensity_i - mean_intensity)**2) /
                           denominator
        denominator =  V1                (biased)
                    or (V1**2 - V2) / V1 (unbiased)
        V1 = sum(area_i) = total_area
        V2 = sum(area_i**2)

    Flux and intensity are numerically identical for a single, whole pixel
    as long as area is measured in pixel units (i.e., 1 pixel). By
    extension, if there are no partial pixels in the aperture, then the
    mean intensity is equal to the mean flux.

    The median pixel value in an aperture is somewhat ambiguous when
    partial pixels are involved. Options are to only consider whole pixels
    or pixels with areas above a certain threshold, or to calculated a
    "weighted median". All of these options are possible with `median`.

    .. [1] https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    """
    rows = []
    for aperture in aperture_list:
        data = aperture.extract(image)
        area = aperture.weights
        total_area = np.sum(area)

        mask = np.isnan(data) | area == 0
        if np.sum(mask) == mask.size:
            total_flux = np.nan
            median_intensity = np.nan
            mean_intensity = np.nan
            std_intensity = np.nan

        else:
            intensity, area = data[-mask], aperture.weights[-mask]
            flux = intensity * area

            total_flux = np.sum(flux)
            if median_type == 'weighted':
                kwargs = dict(weights=area)
            elif median_type is not None:
                kwargs = dict(weights=area, thresh=median_type)
            else:
                kwargs = dict()
            median_intensity = median(intensity, **kwargs)

            mean_intensity = total_flux / total_area
            if unbiased:
                denominator = (total_area**2 - np.sum(area**2)) / total_area
            else:
                denominator = total_area
            std_intensity = np.sqrt(
                np.sum(area * (intensity - mean_intensity)**2) / denominator)

        row = (aperture.label, total_area, total_flux,
               median_intensity, mean_intensity, std_intensity)
        rows.append(row)

    names = ('label', 'area', 'total', 'median', 'mean', 'std')
    dtype = (str, float, float, float, float, float)
    if astropy_installed:
        table = astropy.table.Table(data=zip(*rows), names=names, dtype=dtypes)

    return table


