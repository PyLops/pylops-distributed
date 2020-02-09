"""
Convolution
===========
This example shows how to use the
:py:class:`pylops_distributed.signalprocessing.Convolve1D` operator to perform
convolution between two signals.

Note that when the input dataset is distributed across multiple nodes,
additional care should be taken when applying convolution.
In PyLops-distributed, we leverage the :func:`dask.array.map_block` and
:func:`dask.array.map_overlap` functionalities of dask when chunking is
performed along the dimension of application of the convolution and along any
other dimension, respectively. This is however handled by the inner working of
:py:class:`pylops_distributed.signalprocessing.Convolve1D`.

"""
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr

import pylops_distributed
from pylops.utils.wavelets import ricker

plt.close('all')

###############################################################################
# We will start by creating a zero signal of lenght :math:`nt` and we will
# place a unitary spike at its center. We also create our filter to be
# applied by means of
# :py:class:`pylops-distributed.signalprocessing.Convolve1D` operator.
# Following the seismic example mentioned above, the filter is a
# `Ricker wavelet <http://subsurfwiki.org/wiki/Ricker_wavelet>`_
# with dominant frequency :math:`f_0 = 30 Hz`.
nt = 1001
dt = 0.004
t = np.arange(nt)*dt
x = np.zeros(nt)
x[int(nt/2)] = 1
x = da.from_array(x, chunks=nt // 2 + 1)
h, th, hcenter = ricker(t[:101], f0=30)

Cop = pylops_distributed.signalprocessing.Convolve1D(nt, h=h, offset=hcenter,
                                                     chunks=(nt // 2 + 1,
                                                             nt // 2 + 1),
                                                     dtype='float32')
y = Cop * x
xinv = Cop / y

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t, x, 'k', lw=2, label=r'$x$')
ax.plot(t, y, 'r', lw=2, label=r'$y=Ax$')
ax.plot(t, xinv, '--g', lw=2, label=r'$x_{ext}$')
ax.set_title('Convolve 1d data', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xlim(1.9, 2.1)

###############################################################################
# We show now that also a filter with mixed phase (i.e., not centered
# around zero) can be applied and inverted for using the
# :py:class:`pylops.signalprocessing.Convolve1D`
# operator.
Cop = pylops_distributed.signalprocessing.Convolve1D(nt, h=h, offset=hcenter - 3,
                                                     chunks=(nt // 2 + 1,
                                                             nt // 2 + 1),
                                                     dtype='float32')
y = Cop * x
y1 = Cop.H * x
xinv = Cop / y

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t, x, 'k', lw=2, label=r'$x$')
ax.plot(t, y, 'r', lw=2, label=r'$y=Ax$')
ax.plot(t, y1, 'b', lw=2, label=r'$y=A^Hx$')
ax.plot(t, xinv, '--g', lw=2, label=r'$x_{ext}$')
ax.set_title('Convolve 1d data with non-zero phase filter', fontsize=14,
             fontweight='bold')
ax.set_xlim(1.9, 2.1)
ax.legend()
