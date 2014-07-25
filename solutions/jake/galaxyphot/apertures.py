"""

======================
`galaxyphot.apertures`
======================

Various aperture classes.


Classes
-------

==================== ===================================
`ApertureBase`       Base class for aperture subclasses.
`CircularAnnulus`    A circular annulus aperture.
`CircularAperture`   A circular aperture.
`EllipticalAnnulus`  An elliptical annulus aperture.
`EllipticalAperture` An elliptical aperture.
`PolygonAperture`    A polygon aperture.
==================== ===================================


Functions
---------

================== ==========================================
`extract`          Extract image data for the given aperture.
`from_region_file` Create apertures from a DS9 region file.
================== ==========================================

"""
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np


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
    extent
    grid
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

    Methods
    -------
    extract
    plot

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

    def plot(self, ax=None, draw=False, **kwargs):
        """Plot `patch`, the patch representation of the aperture. Accepts
        all valid patch keyword arguments. Default is to create a new plot,
        but the patch can be added to an existing axis with the `ax`
        keyword. If `draw` is True, ``plt.draw()`` is called after adding
        the patch (default is False).

        """
        if ax is None:
            ax = plt.subplot(1, 1, 1)
        ax.add_patch(self.patch(**kwargs))
        if draw:
            plt.draw()
        return


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
    area
    r
    weights
    xy0

    Methods
    -------
    copy
    patch

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
        xy0 = np.array(xy0, dtype=float)
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
    def area(self):
        """Exact area of the aperture in pixels. Can be compared with the
        sum of `weights` to determine if `nsub` is sufficiently large.

        """
        return np.pi * self.r**2

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
        weights = (r <= r0).astype(float)

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
            kwargs = dict(axis=(1, 2), dtype=float)
            weights[i,j] = np.sum(subr <= r0, **kwargs) / nsub**2

        return weights

    def copy(self):
        """Return a copy of the aperture."""
        xy0 = self.xy0.copy()
        r = self.r
        label = self.label
        nsub = self.nsub
        return CircularAperture(xy0, r, label=label, nsub=nsub)

    def patch(self, **kwargs):
        """Return a `matplotlib.patches.Circle` patch representation of the
        aperture for plotting. Accepts all valid patch keyword arguments.

        """
        kwargs['radius'] = self.r
        return matplotlib.patches.Circle(self.xy0, **kwargs)


class CircularAnnulus(object):

    """A ciruclar annulus aperture.

    This is essentially a wrapper around two `CircularAperture` instances,
    with some extra functionality to make them work together.

    Parameters
    ----------
    xy0 : (2,) array_like
        Initialize the `xy0` property.
    r1 : float
        Initialize the `r1` property.
    r2 : float
        Initialize the `r2` property.
    label : str, optional
        Initialize the `ApertureBase.label` property. Default is None.
    nsub : int, optional
        Initialize the `ApertureBase.nsub` property. Default is 50.

    Attributes
    ----------
    area
    extent
    grid
    ijextent
    ijlimits
    label
    limits
    nsub
    r1
    r2
    weights
    xy0

    Methods
    -------
    copy
    extract
    patch

    """

    def __init__(self, xy0, r1, r2, label=None, nsub=50):
        self._inner = CircularAperture(xy0, r1, label=label, nsub=nsub)
        self._outer = CircularAperture(xy0, r2, label=label, nsub=nsub)
        self.xy0 = xy0
        self.r1 = r1
        self.r2 = r2
        self.nsub = nsub
        self.label = label

    @property
    def xy0(self):
        """(2,) ndarray. x and y pixel coordinates of the center."""
        return self._inner.xy0

    @xy0.setter
    def xy0(self, xy0):
        self._inner.xy0 = xy0
        self._outer._xy0 = self._inner._xy0  # Anchor to inner

    @property
    def r1(self):
        """float. Inner radius in pixels."""
        return self._inner.r

    @r1.setter
    def r1(self, r):
        if not 0 < r < self._outer.r:
            raise ValueError('r1 must be between 0 and r2.')
        self._inner.r = r

    @property
    def r2(self):
        """float. Outer radius in pixels."""
        return self._outer.r

    @r2.setter
    def r2(self, r):
        if not self._inner.r < r:
            raise ValueError('r2 must be greater than r1.')
        self._outer.r = r

    @property
    def area(self):
        """Exact area of the aperture in pixels. Can be compared with the
        sum of `weights` to determine if `nsub` is sufficiently large.

        """
        return np.pi * (self.r2**2 - self.r1**2)

    @property
    def nsub(self):
        """See `ApertureBase.nsub`."""
        return self._inner.nsub

    @nsub.setter
    def nsub(self, nsub):
        self._inner.nsub = nsub
        self._outer.nsub = nsub

    @property
    def label(self):
        """See `ApertureBase.label`."""
        return self._inner.label

    @label.setter
    def label(self, label):
        self._inner.label = label
        self._outer.label = label

    @property
    def limits(self):
        """x and y limits of the outer circle. See `ApertureBase.limits`."""
        return self._outer.limits

    @property
    def ijlimits(self):
        """i and j limits of the outer circle. See
        `ApertureBase.ijlimits`.

        """
        return self._outer.ijlimits

    @property
    def ijextent(self):
        """i and j extent of the outer circle. See
        `ApertureBase.ijextent`.

        """
        return self._outer.ijextent

    @property
    def extent(self):
        """x and y extent of the outer circle. See `ApertureBase.extent`."""
        return self._outer.extent

    @property
    def grid(self):
        """See `ApertureBase.grid`."""
        return self._outer.grid

    @property
    def weights(self):
        """Difference between the outer and the outer weight arrays. See
        `CircularAperture.weights`.

        """
        weights1 = self._inner.weights
        weights2 = self._outer.weights
        i1, j1 = self._inner.ijextent[0::2]
        i2, j2 = self._outer.ijextent[0::2]
        i0, j0 = i1 - i2, j1 - j2
        di, dj = weights1.shape
        ijslice = (slice(i0, i0+di), slice(j0, j0+dj))
        weights2[ijslice] -= weights1
        return weights2

    def extract(self, image):
        return self._outer.extract(image)

    def copy(self):
        """Return a copy of the aperture."""
        xy0 = self.xy0.copy()
        r1 = self.r1
        r2 = self.r2
        label = self.label
        nsub = self.nsub
        return CircularAnnulus(xy0, r1, r2, label=label, nsub=nsub)

    def patch(self, res=1000, **kwargs):
        """Return a `matplotlib.patches.PathPatch` patch representation of
        the aperture for plotting (`matplotlib.patches.Circle` does not
        currently support holes). Accepts all valid patch keyword
        arguments. The number of samples along the inner and outer curves
        is set using the `res` keyword (default is 1000).

        """
        phi = np.linspace(0, 2*np.pi, res)
        x1 = self.r1 * np.cos(phi)  # Counter clockwise
        y1 = self.r1 * np.sin(phi)
        x2 = self.r2 * np.cos(phi[::-1])  # Clockwise
        y2 = self.r2 * np.sin(phi[::-1])
        x, y = np.hstack((x1, x2)), np.hstack((y1, y2))
        xy = np.vstack((x, y)).T + self.xy0
        codes = [matplotlib.path.Path.LINETO] * len(x1)
        codes[0] = matplotlib.path.Path.MOVETO
        codes = codes * 2
        path = matplotlib.path.Path(xy, codes)
        return matplotlib.patches.PathPatch(path, **kwargs)

    def plot(self, ax=None, draw=False, **kwargs):
        """Plot `patch`, the patch representation of the aperture. Accepts
        all valid patch keyword arguments. Default is to create a new plot,
        but the patch can be added to an existing axis with the `ax`
        keyword. If `draw` is True, ``plt.draw()`` is called after adding
        the patch (default is False).

        """
        if ax is None:
            ax = plt.subplot(1, 1, 1)
        ax.add_patch(self.patch(**kwargs))
        if draw:
            plt.draw()
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
    area
    b
    theta
    weights
    xy0

    Methods
    -------
    copy
    patch

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
        xy0 = np.array(xy0, dtype=float)
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
    def area(self):
        """Exact area of the aperture in pixels. Can be compared with the
        sum of `weights` to determine if `nsub` is sufficiently large.

        """
        return np.pi * self.a * self.b

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
        weights = (r <= 1).astype(float)

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
            kwargs = dict(axis=(1, 2), dtype=float)
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

    def patch(self, **kwargs):
        """Return a `matplotlib.patches.Ellipse` patch representation of the
        aperture for plotting. Accepts all valid patch keyword arguments.

        """
        kwargs['angle'] = self.theta
        return matplotlib.patches.Ellipse(
            self.xy0, 2*self.a, 2*self.b, **kwargs)


class EllipticalAnnulus(object):

    """An elliptical annulus aperture.

    This is essentially a wrapper around two `EllipticalAperture`
    instances, with some extra functionality to make them work together.

    Parameters
    ----------
    xy0 : (2,) array_like
        Initialize the `xy0` property.
    a1 : float
        Initialize the `a1` property.
    b1 : float
        Initialize the `b1` property.
    a2 : float
        Initialize the `a2` property.
    b2 : float
        Initialize the `b2` property.
    theta : float
        Initialize the `theta` property.
    label : str, optional
        Initialize the `ApertureBase.label` property. Default is None.
    nsub : int, optional
        Initialize the `ApertureBase.nsub` property. Default is 50.

    Attributes
    ----------
    a1
    a2
    area
    b1
    b2
    extent
    grid
    ijextent
    ijlimits
    label
    limits
    nsub
    theta
    weights
    xy0

    Methods
    -------
    copy
    extract
    patch

    """

    def __init__(self, xy0, a1, b1, a2, b2, theta, label=None, nsub=50):
        self._inner = EllipticalAperture(
            xy0, a1, b1, theta=theta, label=label, nsub=nsub)
        self._outer = EllipticalAperture(
            xy0, a2, b2, theta=theta, label=label, nsub=nsub)
        self.xy0 = xy0
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.theta = theta
        self.nsub = nsub
        self.label = label

    @property
    def xy0(self):
        """(2,) ndarray. x and y pixel coordinates of the center."""
        return self._inner.xy0

    @xy0.setter
    def xy0(self, xy0):
        self._inner.xy0 = xy0
        self._outer._xy0 = self._inner._xy0  # Anchor to inner

    @property
    def a1(self):
        """float. Inner semimajor axis in pixels."""
        return self._inner.a

    @a1.setter
    def a1(self, a):
        if not 0 < a < self._outer.a:
            raise ValueError('a1 must be between 0 and a2.')
        self._inner.a = a

    @property
    def b1(self):
        """float. Inner semiminor axis in pixels."""
        return self._inner.b

    @b1.setter
    def b1(self, b):
        if not 0 < b < self._outer.b:
            raise ValueError('b1 must be between 0 and b2.')
        self._inner.b = b

    @property
    def a2(self):
        """float. Outer semimajor axis in pixels."""
        return self._outer.a

    @a2.setter
    def a2(self, a):
        if not self._inner.a < a:
            raise ValueError('a2 must be greater than a1.')
        self._outer.a = a

    @property
    def b2(self):
        """float. Outer semiminor axis in pixels."""
        return self._outer.b

    @b2.setter
    def b2(self, b):
        if not self._inner.b < b:
            raise ValueError('b2 must be greater than b1.')
        self._outer.b = b

    @property
    def theta(self):
        """float. Counter clockwise rotation angle in degrees."""
        return self._inner.theta

    @theta.setter
    def theta(self, theta):
        self._inner.theta = theta
        self._outer.theta = theta

    @property
    def area(self):
        """Exact area of the aperture in pixels. Can be compared with the
        sum of `weights` to determine if `nsub` is sufficiently large.

        """
        return np.pi * (self.a2 * self.b2 - self.a1 * self.b1)

    @property
    def nsub(self):
        """See `ApertureBase.nsub`."""
        return self._inner.nsub

    @nsub.setter
    def nsub(self, nsub):
        self._inner.nsub = nsub
        self._outer.nsub = nsub

    @property
    def label(self):
        """See `ApertureBase.label`."""
        return self._inner.label

    @label.setter
    def label(self, label):
        self._inner.label = label
        self._outer.label = label

    @property
    def limits(self):
        """x and y limits of the outer ellipse. See `ApertureBase.limits`."""
        return self._outer.limits

    @property
    def ijlimits(self):
        """i and j limits of the outer ellipse. See
        `ApertureBase.ijlimits`.

        """
        return self._outer.ijlimits

    @property
    def ijextent(self):
        """i and j extent of the outer ellipse. See
        `ApertureBase.ijextent`.

        """
        return self._outer.ijextent

    @property
    def extent(self):
        """x and y extent of the outer ellipse. See
        `ApertureBase.extent`.

        """
        return self._outer.extent

    @property
    def grid(self):
        """See `ApertureBase.grid`."""
        return self._outer.grid

    @property
    def weights(self):
        """Difference between the outer and the outer weight arrays. See
        `EllipticalAperture.weights`.

        """
        weights1 = self._inner.weights
        weights2 = self._outer.weights
        i1, j1 = self._inner.ijextent[0::2]
        i2, j2 = self._outer.ijextent[0::2]
        i0, j0 = i1 - i2, j1 - j2
        di, dj = weights1.shape
        ijslice = (slice(i0, i0+di), slice(j0, j0+dj))
        weights2[ijslice] -= weights1
        return weights2

    def extract(self, image):
        return self._outer.extract(image)

    def copy(self):
        """Return a copy of the aperture."""
        xy0 = self.xy0.copy()
        a1 = self.a1
        b1 = self.b1
        a2 = self.a2
        b2 = self.b2
        theta = self.theta
        label = self.label
        nsub = self.nsub
        return EllipticalAnnulus(
            xy0, a1, b1, a2, b2, theta, label=label, nsub=nsub)

    def patch(self, res=1000, **kwargs):
        """Return a `matplotlib.patches.PathPatch` patch representation of
        the aperture for plotting (`matplotlib.patches.Ellipse` does not
        currently support holes). Accepts all valid patch keyword
        arguments. The number of samples along the inner and outer curves
        is set using the `res` keyword (default is 1000).

        """
        a1, b1 = self.a1, self.b1
        a2, b2 = self.a2, self.b2
        theta = np.radians(self.theta)
        phi = np.linspace(0, 2*np.pi, res)
        x1 = (a1 * np.cos(phi) * np.cos(theta) -
              b1 * np.sin(phi) * np.sin(theta))  # Counter clockwise
        y1 = (a1 * np.cos(phi) * np.sin(theta) +
              b1 * np.sin(phi) * np.cos(theta))
        x2 = (a2 * np.cos(phi[::-1]) * np.cos(theta) -
              b2 * np.sin(phi[::-1]) * np.sin(theta))  # Clockwise
        y2 = (a2 * np.cos(phi[::-1]) * np.sin(theta) +
              b2 * np.sin(phi[::-1]) * np.cos(theta))
        x, y = np.hstack((x1, x2)), np.hstack((y1, y2))
        xy = np.vstack((x, y)).T + self.xy0
        codes = [matplotlib.path.Path.LINETO] * len(x1)
        codes[0] = matplotlib.path.Path.MOVETO
        codes = codes * 2
        path = matplotlib.path.Path(xy, codes)
        return matplotlib.patches.PathPatch(path, **kwargs)

    def plot(self, ax=None, draw=False, **kwargs):
        """Plot `patch`, the patch representation of the aperture. Accepts
        all valid patch keyword arguments. Default is to create a new plot,
        but the patch can be added to an existing axis with the `ax`
        keyword. If `draw` is True, ``plt.draw()`` is called after adding
        the patch (default is False).

        """
        if ax is None:
            ax = plt.subplot(1, 1, 1)
        ax.add_patch(self.patch(**kwargs))
        if draw:
            plt.draw()
        return


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
    patch

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
            xy = np.array(xy, dtype=float)
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
        xy0 = np.array(xy0, dtype=float)
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
                   .reshape(y.size, x.size).astype(float))

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
            bordertest = bordertest.reshape(y.size, x.size).astype(float)

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
            kwargs = dict(axis=(1, 2), dtype=float)
            weights[i, j] = np.sum(subweights, **kwargs) / self.nsub**2

        return weights

    def copy(self):
        """Return a copy of the aperture."""
        xy = self.xy.copy()
        label = self.label
        nsub = self.nsub
        return PolygonAperture(xy, label=label, nsub=nsub)

    def patch(self, **kwargs):
        """Return a `matplotlib.patches.PathPatch` patch representation of
        the aperture for plotting. Accepts all valid patch keyword
        arguments.

        """
        return matplotlib.patches.PathPatch(self.path, **kwargs)


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
                xy = np.array(line.split()[1:], dtype=float).reshape(-1, 2)
                aperture = PolygonAperture(xy, label=label)
                aperture_list.append(aperture)
            else:
                pass

    return aperture_list
