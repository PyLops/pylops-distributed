.. _changlog:

Changelog
=========


Version 0.1.0
-------------

*Released on: 09/02/2020*

* Added :py:class:`pylops_distributed.Restriction` operator
* Added :py:class:`pylops_distributed.signalprocessing.Convolve1D`
  and :py:class:`pylops_distributed.signalprocessing.FFT2D` operators
* Improved efficiency of
  :py:class:`pylops_distributed.signalprocessing.Fredholm1` when
  ``saveGt=False``
* Adapted :py:func:`pylops_distributed.optimization.cg.cg` and
  :py:func:`pylops_distributed.optimization.cg.cgls` solvers for
  complex numbers


Version 0.0.0
-------------

*Released on: 01/09/2019*

* First official release.
