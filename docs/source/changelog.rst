.. _changlog:

Changelog
=========


Version 0.2.0
-------------

*Released on: 06/06/2020*

* Added ``prescaled`` input parameter to :class:`pylops_distributed.waveeqprocessing.MDC`
  and :class:`pylops_distributed.waveeqprocessing.Marchenko`
* Added ``dtype`` parameter to the ``FFT`` calls in the definition of the
  :class:`pylops_distributed.waveeqprocessing.MDD` operation. This ensure that the type
  of the real part of ``G`` input is enforced to the output vectors of the
  forward and adjoint operations.
* Changed handling of ``dtype`` in :class:`pylops_distributed.signalprocessing.FFT`
  to ensure that the type of the input vector is retained when applying forward and adjoint.
* Added ``PBS`` backend to :func:`pylops_distributed.utils.backend.dask`


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
