PyLops-distributed
==================

.. note:: This library is under early development.

   Expect things to constantly change until version v1.0.0.

This library is an extension of `PyLops <https://pylops.readthedocs.io/en/latest/>`_
for distributed operators.

As much as `numpy <http://www.numpy.org>`_ and
`scipy <http://www.scipy.org/scipylib/index.html>`_ lie at the core of the parent project
PyLops, PyLops-distributed heavily builds on top of `Dask <https://dask.org>`_,
and more specifically Dask arrays.

Doing so, linear operators can be parallelized across several
processes on a single node or across multiple nodes. Their forward and adjoint
are first lazily built as directed acyclic graphs and evaluated only when requested by
the user (or automatically within one of our solvers).

Most of the operators and solvers in PyLops-distributed mirror their equivalents
in PyLops and users can seamlessly switch between PyLops and PyLops-distributed
or even combine operators acting locally with distributed operators.

Here is a simple example showing how a diagonal operator can be created,
applied and inverted using PyLops:

.. code-block:: python

   import numpy as np
   from pylops import Diagonal

   n = 10
   x = np.ones(n)
   d = np.arange(n) + 1

   Dop = Diagonal(d)

   # y = Dx
   y = Dop*x
   # x = D'y
   xadj = Dop.H*y
   # xinv = D^-1 y
   xinv = Dop / y

and similarly using PyLops-distributed:

.. code-block:: python

   import numpy as np
   import dask.array as da
   import pylops_distributed
   from pylops_distributed import Diagonal

   # set-up client
   client = pylops_distributed.utils.backend.dask()

   n = 10
   x = da.ones(n, chunks=(n//2,))
   d = da.from_array(np.arange(n) + 1, chunks=(n//2, n//2))

   Dop = Diagonal(d)

   # y = Dx
   y = Dop*x
   # x = D'y
   xadj = Dop.H*y
   # xinv = D^-1 y
   xinv = Dop / y

   da.compute((y, xadj, xinv))
   client.close()

It is worth noticing two things at this point:

- In this specific case we did not even need to reimplement the ``Diagonal`` operator.
  Calling numpy operations as methods (e.g., ``x.sum()``) instead of functions (e.g., ``np.sum(x)``) makes it
  automatic for our operator to act as a distributed operator when a dask array is provided instead.
  Unfortunately not all numpy functions are also implemented as methods: in those cases we reimplement the o
  perator directly within PyLops-distributed.
- Using ``*`` and ``.H*`` is still possible also within PyLops-distributed, however when initializing an
  operator we will need to decide whether we want to simply create dask graph or also evaluation.
  This gives flexibility as we can decide if and when apply evaluation using the ``compute`` method
  on a dask array of choice.


History
-------
PyLops-Distributed was initially written and it is currently maintained by `Equinor <https://www.equinor.com>`_
It is an extension of `PyLops <https://pylops.readthedocs.io/en/latest/>`_ for large-scale optimization with
*distributed* linear operators that can be tailored to our needs, and as contribution to the free software community.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started:

   installation.rst
   tutorials/index.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation:

   api/index.rst
   api/others.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting involved:

   Contributing <contributing.rst>
   Changelog <changelog.rst>
   Roadmap <roadmap.rst>
   Credits <credits.rst>

