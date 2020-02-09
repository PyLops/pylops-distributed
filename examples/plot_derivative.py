"""
Derivatives
===========
This example shows how to use the suite of derivative operators, namely
:py:class:`pylops_distributed.FirstDerivative`,
:py:class:`pylops_distributed.SecondDerivative`, and
:py:class:`pylops_distributed.Laplacian`.

The derivative operators can be applied along any dimension of a N-dimensional
input. When the input is chuncked along any other direction(s) than the one
the derivative is applied, the derivative is efficiently performed without
neither communication between workers nor duplication of part of the input
array. On the other hand when the input is chuncked along the direction where
the derivative is applied, the chunks are partially overlapping such that no
communication is required between the workers when applying the derivative.

In some applications, the user cannot avoid this second scenario to happen
(e.g, when the derivative should be applied to all the dimensions of the
dataset). Nevertheless our implementation makes this possible in a fully
transparent way and with very little additional overhead.

"""
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

import pylops
import pylops_distributed

plt.close('all')
np.random.seed(0)

###############################################################################
# Let's start by looking at a simple first-order centered derivative. We
# chunck the vector in 3 chunks.
nx = 100
nchunks = 3

x = np.zeros(nx)
x[int(nx/2)] = 1
xd = da.from_array(x, chunks=nx // nchunks + 1)
print('x:', xd)

Dop = pylops.FirstDerivative(nx, dtype='float32')
dDop = pylops_distributed.FirstDerivative(nx, compute=(True, True),
                                          dtype='float32')

y = Dop * x
xadj = Dop.H * y

yd = Dop * xd
xadjd = Dop.H * yd


fig, axs = plt.subplots(3, 1, figsize=(13, 8))
axs[0].stem(np.arange(nx), x, linefmt='k', markerfmt='ko',
            use_line_collection=True)
axs[0].set_title('Input', size=20, fontweight='bold')
axs[1].stem(np.arange(nx), y, linefmt='--r', markerfmt='ro',
            label='Numpy', use_line_collection=True)
axs[1].stem(np.arange(nx), yd, linefmt='--r', markerfmt='ro',
            label='Dask', use_line_collection=True)
axs[1].set_title('Forward', size=20, fontweight='bold')
axs[1].legend()
axs[2].stem(np.arange(nx), xadj, linefmt='k', markerfmt='ko',
            label='Numpy', use_line_collection=True)
axs[2].stem(np.arange(nx), xadjd, linefmt='--r', markerfmt='ro',
            label='Dask', use_line_collection=True)
axs[2].set_title('Adjoint', size=20, fontweight='bold')
axs[2].legend()
plt.tight_layout()

###############################################################################
# As expected we obtain the same result, with the only difference that
# in the second case we did not need to explicitly create a matrix,
# saving memory and computational time.
#
# Let's move onto applying the same first derivative to a 2d array in
# the first direction. Now we consider two cases, first when the data is
# chunked along the first direction and second when its chunked along the
# second direction.
nx, ny = 11, 21
nchunks = 2

A = np.zeros((nx, ny))
A[nx//2, ny//2] = 1.
A1d = da.from_array(A, chunks= (nx // nchunks + 1, ny))
A2d = da.from_array(A, chunks= (nx , ny // nchunks + 1))
print('A1d:', A1d)
print('A2d:', A2d)

Dop = pylops_distributed.FirstDerivative(nx * ny, dims=(nx, ny),
                                         compute=(True, True),
                                         dir=0, dtype='float64')

B1d = np.reshape(Dop * A1d.flatten(), (nx, ny))
B2d = np.reshape(Dop * A2d.flatten(), (nx, ny))

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
fig.suptitle('First Derivative in 1st direction', fontsize=12,
             fontweight='bold', y=0.95)
im = axs[0].imshow(A, interpolation='nearest', cmap='rainbow')
axs[0].axis('tight')
axs[0].set_title('x')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(B1d, interpolation='nearest', cmap='rainbow')
axs[1].axis('tight')
axs[1].set_title('y (1st dir chunks)')
plt.colorbar(im, ax=axs[1])
im = axs[2].imshow(B2d, interpolation='nearest', cmap='rainbow')
axs[2].axis('tight')
axs[2].set_title('y (2nd dir chunks)')
plt.colorbar(im, ax=axs[2])
plt.tight_layout()
plt.subplots_adjust(top=0.8)

###############################################################################
# We can now do the same for the second derivative
A = np.zeros((nx, ny))
A[nx//2, ny//2] = 1.
A1d = da.from_array(A, chunks= (nx // nchunks + 1, ny))
A2d = da.from_array(A, chunks= (nx , ny // nchunks + 1))
print('A1d:', A1d)
print('A2d:', A2d)

Dop = pylops_distributed.SecondDerivative(nx * ny, dims=(nx, ny),
                                          compute=(True, True),
                                          dir=0, dtype='float64')

B1d = np.reshape(Dop * A1d.flatten(), (nx, ny))
B2d = np.reshape(Dop * A2d.flatten(), (nx, ny))

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
fig.suptitle('Second Derivative in 1st direction', fontsize=12,
             fontweight='bold', y=0.95)
im = axs[0].imshow(A, interpolation='nearest', cmap='rainbow')
axs[0].axis('tight')
axs[0].set_title('x')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(B1d, interpolation='nearest', cmap='rainbow')
axs[1].axis('tight')
axs[1].set_title('y (1st dir chunks)')
plt.colorbar(im, ax=axs[1])
im = axs[2].imshow(B2d, interpolation='nearest', cmap='rainbow')
axs[2].axis('tight')
axs[2].set_title('y (2nd dir chunks)')
plt.colorbar(im, ax=axs[2])
plt.tight_layout()
plt.subplots_adjust(top=0.8)


###############################################################################
# We use the symmetrical Laplacian operator as well
# as a asymmetrical version of it (by adding more weight to the
# derivative along one direction)

# symmetrical
L2symop = pylops_distributed.Laplacian(dims=(nx, ny), weights=(1, 1),
                                       compute=(True, True), dtype='float64')

# asymmetrical
L2asymop = pylops_distributed.Laplacian(dims=(nx, ny), weights=(3, 1),
                                        compute=(True, True), dtype='float64')

Bsym = np.reshape(L2symop * A1d.flatten(), (nx, ny))
Basym = np.reshape(L2asymop * A2d.flatten(), (nx, ny))

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
fig.suptitle('Laplacian', fontsize=12,
             fontweight='bold', y=0.95)
im = axs[0].imshow(A, interpolation='nearest', cmap='rainbow')
axs[0].axis('tight')
axs[0].set_title('x')
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(Bsym, interpolation='nearest', cmap='rainbow')
axs[1].axis('tight')
axs[1].set_title('y sym')
plt.colorbar(im, ax=axs[1])
im = axs[2].imshow(Basym, interpolation='nearest', cmap='rainbow')
axs[2].axis('tight')
axs[2].set_title('y asym')
plt.colorbar(im, ax=axs[2])
plt.tight_layout()
plt.subplots_adjust(top=0.8)
