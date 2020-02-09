import numpy as np
import dask.array as da

from scipy.signal import convolve, fftconvolve
from pylops_distributed import LinearOperator


class Convolve1D(LinearOperator):
    r"""1D convolution operator.

    Apply one-dimensional convolution with a compact filter to model (and data)
    along a specific direction of a multi-dimensional array depending on the
    choice of ``dir``.

    Note that if a multi-dimensional array is provided the array can also
    be chuncked along the direction ``dir`` where convolution is performed.
    In this case, ``dask`` handles the communication of borders between
    neighboring blocks.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    h : :obj:`numpy.ndarray`
        1d compact filter to be convolved to input signal
    offset : :obj:`int`
        Index of the center of the compact filter
    dims : :obj:`tuple`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which convolution is applied
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array.array`
    chunks : :obj:`tuple`, optional
        Chunk size for model and data. If provided it will rechunk the model
        before applying the forward pass and the data before applying the
        adjoint pass
    todask : :obj:`tuple`, optional
        Apply :func:`dask.array.from_array` to model and data before applying
        forward and adjoint respectively
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Raises
    ------
    ValueError
        If ``offset`` is bigger than ``len(h) - 1``

    Notes
    -----
    Refer to :class:`pylops.signalprocessing.Convolve1D` for implementation
    details.

    """
    def __init__(self, N, h, offset=0, dims=None, dir=0,
                 compute=(False, False), chunks=(None, None),
                 todask=(False, False), dtype='float64'):
        if offset > len(h) - 1:
            raise ValueError('offset must be smaller than len(h) - 1')
        self.h = h
        self.hstar = np.flip(self.h)
        self.nh = len(h)
        self.offset = 2 * (self.nh // 2 - int(offset))
        if self.nh % 2 == 0:
            self.offset -= 1
        if self.offset != 0:
            self.h = \
                np.pad(self.h, (self.offset if self.offset > 0 else 0,
                                -self.offset if self.offset < 0 else 0),
                       mode='constant')
        self.hstar = np.flip(self.h)
        if dims is not None:
            # add dimensions to filter to match dimensions of model and data
            hdims = [1] * len(dims)
            hdims[dir] = len(self.h)
            self.h = self.h.reshape(hdims)
            self.hstar = self.hstar.reshape(hdims)
        self.dir = dir
        if dims is None:
            self.dims = np.array([N, 1])
            self.reshape = False
        else:
            if np.prod(dims) != N:
                raise ValueError('product of dims must equal N!')
            else:
                self.dims = np.array(dims)
                self.reshape = True
        self.shape = (np.prod(self.dims), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.compute = compute
        self.chunks = chunks
        self.todask = todask
        self.Op = None
        self.explicit = False

    def _matvec(self, x):
        if not self.reshape:
            y = da.map_overlap(x, lambda x: np.convolve(x, self.h,
                                                        mode='same'),
                               depth=len(self.h) // 2, boundary=0, trim=True)
        else:
            x = da.reshape(x, self.dims)
            if self.chunks[0] is not None:
                x = x.rechunk(self.chunks[0])
            if x.numblocks[self.dir] == 1:
                # apply the convolution for each block indipendently
                y = da.map_blocks(fftconvolve, x, self.h, mode='same',
                                  axes=self.dir)
            else:
                # communicate borders between chunks and perform convolution
                # (trimming away the edges at the end)
                depth = [1] * len(self.dims)
                depth[self.dir] = self.h.shape[self.dir] // 2
                y = da.map_overlap(x, lambda x: fftconvolve(x, self.h,
                                                            mode='same',
                                                            axes=self.dir),
                                   depth=tuple(depth), boundary=0,
                                   trim=True)
            y = y.ravel()
        return y

    def _rmatvec(self, x):
        if not self.reshape:
            y = da.map_overlap(x, lambda x: np.convolve(x, self.hstar,
                                                        mode='same'),
                               depth=len(self.h) // 2, boundary=0, trim=True)
        else:
            x = da.reshape(x, self.dims)
            if self.chunks[1] is not None:
                x = x.rechunk(self.chunks[1])
            if x.numblocks[self.dir] == 1:
                # apply the convolution for each block indipendently
                y = da.map_blocks(fftconvolve, x, self.hstar,
                                  mode='same', axes=self.dir)
            else:
                # communicate borders between chunks and perform convolution
                # (trimming away the edges at the end)
                depth = [1] * len(self.dims)
                depth[self.dir] = self.hstar.shape[self.dir] // 2
                y = da.map_overlap(x, lambda x: fftconvolve(x, self.hstar,
                                                            mode='same',
                                                            axes=self.dir),
                                   depth=tuple(depth), boundary=0,
                                   trim=True)
        y = y.ravel()
        return y