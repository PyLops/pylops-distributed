.. _api:

PyLops-distributed API
======================


Linear operators
----------------

Basic operators
~~~~~~~~~~~~~~~

.. currentmodule:: pylops_distributed

.. autosummary::
   :toctree: generated/

   MatrixMult
   Identity
   Diagonal
   Transpose
   Roll
   Restriction
   Spread
   VStack
   HStack
   BlockDiag


Smoothing and derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Smoothing1D
   FirstDerivative
   SecondDerivative
   Laplacian


Signal processing
~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops_distributed.signalprocessing

.. autosummary::
   :toctree: generated/

   FFT
   Convolve1D
   Fredholm1


Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops_distributed.waveeqprocessing

.. autosummary::
   :toctree: generated/

   MDC
   Marchenko
   Demigration


Solvers
-------

Low-level solvers
~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops_distributed.optimization.cg

.. autosummary::
   :toctree: generated/

    cg
    cgls


Applications
------------

Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops_distributed.waveeqprocessing

.. autosummary::
   :toctree: generated/

    LSM