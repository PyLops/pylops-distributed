import numpy as np
import dask.array as da
from math import sqrt
from pylops_distributed import LinearOperator


class FFT2D(LinearOperator):
    r"""Two dimensional Fast-Fourier Transform.

    Apply two dimensional Fast-Fourier Transform (FFT) to any pair of axes of a
    multi-dimensional array depending on the choice of ``dirs``.
    Note that the FFT2D operator is a simple overload to the numpy
    :py:func:`numpy.fft.fft2` in forward mode and to the numpy
    :py:func:`numpy.fft.ifft2` in adjoint mode, however scaling is taken
    into account differently to guarantee that the operator is passing the
    dot-test.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    dirs : :obj:`tuple`, optional
        Pair of directions along which FFT2D is applied
    nffts : :obj:`tuple`, optional
        Number of samples in Fourier Transform for each direction (same as
        input if ``nffts=(None, None)``)
    sampling : :obj:`tuple`, optional
        Sampling steps ``dy`` and ``dx``
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
        Operator contains a matrix that can be solved explicitly
        (True) or not (False)

    Raises
    ------
    ValueError
        If ``dims`` has less than two elements, and if ``dirs``, ``nffts``,
        or ``sampling`` has more or less than two elements.

    Notes
    -----
    Refer to :class:`pylops.signalprocessing.FFT2D` for implementation
    details.


    """
    def __init__(self, dims, dirs=(0, 1), nffts=(None, None),
                 sampling=(1., 1.), compute=(False, False),
                 chunks=(None, None), todask=(None, None),
                 dtype='complex128'):
        # checks
        if len(dims) < 2:
            raise ValueError('provide at least two dimensions')
        if len(dirs) != 2:
            raise ValueError('provide at two directions along which fft is applied')
        if len(nffts) != 2:
            raise ValueError('provide at two nfft dimensions')
        if len(sampling) != 2:
            raise ValueError('provide two sampling steps')

        self.dirs = dirs
        self.nffts = np.array([int(nffts[0]) if nffts[0] is not None
                               else dims[self.dirs[0]],
                               int(nffts[1]) if nffts[1] is not None
                               else dims[self.dirs[1]]])
        self.f1 = np.fft.fftfreq(self.nffts[0], d=sampling[0])
        self.f2 = np.fft.fftfreq(self.nffts[1], d=sampling[1])

        self.dims = np.array(dims)
        self.dims_fft = self.dims.copy()
        self.dims_fft[self.dirs[0]] = self.nffts[0]
        self.dims_fft[self.dirs[1]] = self.nffts[1]

        self.shape = (int(np.prod(self.dims_fft)), int(np.prod(self.dims)))
        self.dtype = np.dtype(dtype)
        self.compute = compute
        self.chunks = chunks
        self.todask = todask
        self.Op = None
        self.explicit = False

    def _matvec(self, x):
        x = da.reshape(x, self.dims)
        if self.chunks[0] is not None:
            x = x.rechunk(self.chunks[0])
        y = sqrt(1./np.prod(self.nffts)) * da.fft.fft2(x, s=self.nffts,
                                                          axes=(self.dirs[0],
                                                                self.dirs[1]))
        y = y.ravel()
        return y

    def _rmatvec(self, x):
        x = da.reshape(x, self.dims_fft)
        if self.chunks[1] is not None:
            x = x.rechunk(self.chunks[1])
        y = sqrt(np.prod(self.nffts)) * da.fft.ifft2(x, s=self.nffts,
                                                        axes=(self.dirs[0],
                                                              self.dirs[1]))
        y = da.take(y, np.arange(self.dims[self.dirs[0]]), axis=self.dirs[0])
        y = da.take(y, np.arange(self.dims[self.dirs[1]]), axis=self.dirs[1])
        y = y.ravel()
        return y

