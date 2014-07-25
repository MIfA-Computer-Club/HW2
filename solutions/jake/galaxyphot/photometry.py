"""

=======================
`galaxyphot.photometry`
=======================

Utilities for galaxy photometry.


Functions
---------

================== ===================================================
`apphot`           Perform aperture photometry on an image.
`find_centroid`    Find the centroid of a source in an image.
`fit_gaussian`     Fit a 2d Gaussian function to a source in an image.
================== ===================================================

"""
try:
    import astropy.table
except ImportError:
    ASTROPY_INSTALLED = False
else:
    ASTROPY_INSTALLED = True
import numpy as np
import scipy.optimize

from . import util


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
        `galaxyphot.util.gaussian2d`). Any or all of these may be None, in
        which case the initial guess is based on the aperture. For `amp`,
        the default guess is the difference between the maximum and minimum
        values in the aperture's bounding box. Default is None.

    Returns
    -------
    (6,) tuple
        Best-fit 2d Gaussian parameters: `amp`, `x` and `y` (in pixel
        coordinates), `sigmax` and `sigmay`, and `theta`. See
        `galaxyphot.util.gaussian2d`.

    Notes
    -----
    .. [1] The fitting is actually performed on all pixels in the
       aperture's bounding box defined by its `extent` property. For the
       best results, the aperture should be tuned so that the bounding box
       contains as much of the source as possible without introducting too
       much confusion from other sources.

    """
    def func(xy, amp, x0, y0, sigmax, sigmay, theta):
        f = util.gaussian2d(xy[0], xy[1], amp, x0, y0, sigmax, sigmay, theta)
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
    """Perform aperture photometry on an image.

    Parameters
    ----------
    image : 2d ndarray
        Image data.
    aperture_list : 1d iterable
        Collection of `ApertureBase` instances.
    median_type : {'weighted', float, None}, optional
        Method for calculating the median intensity value in the aperture.
        If 'weighted' (default) or a float, `galaxyphot.util.median` is
        used in either "weighted median" or threshold mode. If None, the
        median is calculated normally from all pixels overlapping the
        aperture (partial pixels are treated as whole pixels).
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
    "weighted median". All of these options are possible with
    `galaxyphot.util.median`.

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
            median_intensity = util.median(intensity, **kwargs)

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
    dtypes = (str, float, float, float, float, float)
    if ASTROPY_INSTALLED:
        table = astropy.table.Table(data=zip(*rows), names=names, dtype=dtypes)
    else:
        table = np.array(rows, dtype=zip(names, dtypes))
    return table
