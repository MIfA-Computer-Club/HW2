"""

============
`galaxyphot`
============

A small set of tools for basic galaxy analysis, including photometry.


Modules
-------

=========== =======================================
`util`      Utilities for the `galaxyphot` package.
=========== =======================================


Classes
-------

==================== ===================================
|ApertureBase|       Base class for aperture subclasses.
|CircularAnnulus|    A circular annulus aperture.
|CircularAperture|   A circular aperture.
|EllipticalAnnulus|  An elliptical annulus aperture.
|EllipticalAperture| An elliptical aperture.
|PolygonAperture|    A polygon aperture.
==================== ===================================


Functions
---------

================== ===================================================
|apphot|           Perform aperture photometry on an image.
|extract|          Extract image data for the given aperture.
|find_centroid|    Find the centroid of a source in an image.
|fit_gaussian|     Fit a 2d Gaussian function to a source in an image.
|from_region_file| Create apertures from a DS9 region file.
================== ===================================================


============
Module Index
============

- `galaxyphot.apertures`
- `galaxyphot.photometry`
- `galaxyphot.util`


.. references

.. |ApertureBase| replace:: `~galaxyphot.apertures.ApertureBase`
.. |CircularAnnulus| replace:: `~galaxyphot.apertures.CircularAnnulus`
.. |CircularAperture| replace:: `~galaxyphot.apertures.CircularAperture`
.. |EllipticalAnnulus| replace:: `~galaxyphot.apertures.EllipticalAnnulus`
.. |EllipticalAperture| replace:: `~galaxyphot.apertures.EllipticalAperture`
.. |PolygonAperture| replace:: `~galaxyphot.apertures.PolygonAperture`
.. |extract| replace:: `~galaxyphot.apertures.extract`
.. |from_region_file| replace:: `~galaxyphot.apertures.from_region_file`

.. |apphot| replace:: `~galaxyphot.photometry.apphot`
.. |find_centroid| replace:: `~galaxyphot.photometry.find_centroid`
.. |fit_gaussian| replace:: `~galaxyphot.photometry.fit_gaussian`

"""
from .apertures import (
    ApertureBase, CircularAnnulus, CircularAperture, EllipticalAnnulus,
    EllipticalAperture, PolygonAperture, extract, from_region_file
    )
from .photometry import apphot, find_centroid, fit_gaussian
from . import util
