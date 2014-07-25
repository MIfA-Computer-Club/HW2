"""

=================
`galaxyphot.util`
=================

Utilities for the `galaxyphot` package.


Functions
---------

============ ========================================
`gaussian2d` 2d Gaussian function.
`median`     Calculate the median of a set of values.
============ ========================================

"""
import numpy as np


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
