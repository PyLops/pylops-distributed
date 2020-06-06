# 0.2.0
* Added ``prescaled`` input parameter to ``pylops_distributed.waveeqprocessing.MDC``
  and ``pylops_distributed.waveeqprocessing.Marchenko``
* Added ``dtype`` parameter to the ``FFT`` calls in the definition of the
  ``pylops_distributed.waveeqprocessing.MDD`` operation. This ensure that the type
  of the real part of ``G`` input is enforced to the output vectors of the
  forward and adjoint operations.
* Changed handling of ``dtype`` in ``pylops_distributed.signalprocessing.FFT``
  to ensure that the type of the input vector is retained when applying forward and adjoint.
* Added ``PBS`` backend to ``pylops_distributed.utils.backend.dask``


# 0.1.0
* Added ``pylops_distributed.Restriction`` operator
* Added ``pylops_distributed.signalprocessing.Convolve1D``
  and ``pylops_distributed.signalprocessing.FFT2D`` operators
* Improved efficiency of
  ``pylops_distributed.signalprocessing.Fredholm1`` when
  ``saveGt=False``
* Adapted ``pylops_distributed.optimization.cg.cg`` and
  ``pylops_distributed.optimization.cg.cgls`` solvers for
  complex numbers



# 0.0.0
* First official release.
