import numpy as np
import dask.array as da
from math import sqrt
from pylops_distributed import LinearOperator


class FFT(LinearOperator):
    r"""One dimensional Fast-Fourier Transform.

    Apply Fast-Fourier Transform (FFT) along a specific direction ``dir`` of a
    multi-dimensional array of size ``dim``.

    Note that the FFT operator is an overload to the dask
    :py:func:`dask.array.fft.fft` (or :py:func:`dask.array.fft.rfft` for
    real models) in forward mode and to the dask :py:func:`dask.array.fft.ifft`
    (or :py:func:`dask.array.fft.irfft` for real models) in adjoint mode.

    Scaling is properly taken into account to guarantee
    that the operator is passing the dot-test.

    .. note:: For a real valued input signal, it is possible to store the
      values of the Fourier transform at positive frequencies only as values
      at negative frequencies are simply their complex conjugates.
      However as the operation of removing the negative part of the frequency
      axis in forward mode and adding the complex conjugates in adjoint mode
      is nonlinear, the Linear Operator FTT with ``real=True`` is not expected
      to pass the dot-test. It is thus *only* advised to use this flag when a
      forward and adjoint FFT is used in the same chained operator
      (e.g., ``FFT.H*Op*FFT``) such as in
      :py:func:`pylops_distributed.waveeqprocessing.mdd.MDC`.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    dir : :obj:`int`, optional
        Direction along which FFT is applied.
    nfft : :obj:`int`, optional
        Number of samples in Fourier Transform (same as input if ``nfft=None``)
    sampling : :obj:`float`, optional
        Sampling step ``dt``.
    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real.
    fftshift : :obj:`bool`, optional
        Apply fftshift/ifftshift (``True``) or not (``False``)
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
        If ``dims`` is not provided and if ``dir`` is bigger than ``len(dims)``

    Notes
    -----
    Refer to :class:`pylops.signalprocessing.FFT` for implementation
    details.

    """
    def __init__(self, dims, dir=0, nfft=None, sampling=1.,
                 real=False, fftshift=False, compute=(False, False),
                 chunks=(None, None), todask=(None, None), dtype='float64'):
        if isinstance(dims, int):
            dims = (dims,)
        if dir > len(dims) - 1:
            raise ValueError('dir=%d must be smaller than '
                             'number of dims=%d...' % (dir, len(dims)))
        self.dir = dir
        self.nfft = nfft if nfft is not None else dims[self.dir]
        self.real = real
        self.fftshift = fftshift
        self.f = np.fft.rfftfreq(self.nfft, d=sampling) if real \
            else np.fft.fftfreq(self.nfft, d=sampling)
        if len(dims) == 1:
            self.dims = np.array([dims[0], 1])
            self.dims_fft = self.dims.copy()
            self.dims_fft[self.dir] = self.nfft // 2 + 1 if \
                self.real else self.nfft
            self.reshape = False
        else:
            self.dims = np.array(dims)
            self.dims_fft = self.dims.copy()
            self.dims_fft[self.dir] = self.nfft // 2 + 1 if \
                self.real else self.nfft
            self.reshape = True
        self.shape = (int(np.prod(dims) * (self.nfft // 2 + 1 if self.real
                                           else self.nfft) / self.dims[dir]),
                      int(np.prod(dims)))
        # Find types to enforce to forward and adjoint outputs. This is
        # required as np.fft.fft always returns complex128 even if input is
        # float32 or less
        self.dtype = np.dtype(dtype)
        self.cdtype = (np.ones(1, dtype=self.dtype) +
                       1j*np.ones(1, dtype=self.dtype)).dtype
        self.compute = compute
        self.chunks = chunks
        self.todask = todask
        self.Op = None
        self.explicit = False

    def _matvec(self, x):
        if self.reshape:
            x = da.reshape(x, self.dims)
        if self.chunks[0] is not None:
            x = x.rechunk(self.chunks[0])
        if not self.reshape:
            if self.fftshift:
                x = da.fft.ifftshift(x)
            if self.real:
                y = sqrt(1. / self.nfft) * da.fft.rfft(da.real(x),
                                                       n=self.nfft, axis=-1)
            else:
                y = sqrt(1. / self.nfft) * da.fft.fft(x, n=self.nfft,
                                                      axis=-1)
        else:
            if self.fftshift:
                x = da.fft.ifftshift(x, axes=self.dir)
            if self.real:
                y = sqrt(1. / self.nfft) * da.fft.rfft(da.real(x),
                                                       n=self.nfft,
                                                       axis=self.dir)
            else:
                y = sqrt(1. / self.nfft) * da.fft.fft(x, n=self.nfft,
                                                      axis=self.dir)
            y = y.ravel()
        y = y.astype(self.cdtype)
        return y

    def _rmatvec(self, x):
        if self.reshape:
            x = da.reshape(x, self.dims_fft)
        if self.chunks[1] is not None:
            x = x.rechunk(self.chunks[1])
        if not self.reshape:
            if self.real:
                y = sqrt(self.nfft) * da.fft.irfft(x, n=self.nfft, axis=-1)
                y = da.real(y)
            else:
                y = sqrt(self.nfft) * da.fft.ifft(x, n=self.nfft, axis=-1)
            if self.nfft != self.dims[self.dir]:
                y = y[:self.dims[self.dir]]
            if self.fftshift:
                y = da.fft.fftshift(y)
        else:
            if self.real:
                y = sqrt(self.nfft) * da.fft.irfft(x, n=self.nfft,
                                                   axis=self.dir)
                y = da.real(y)
            else:
                y = sqrt(self.nfft) * da.fft.ifft(x, n=self.nfft,
                                                  axis=self.dir)
            if self.nfft != self.dims[self.dir]:
                y = da.take(y, np.arange(0, self.dims[self.dir]),
                            axis=self.dir)
            if self.fftshift:
                y = da.fft.fftshift(y, axes=self.dir)
            y = y.ravel()
        y = y.astype(self.dtype)
        return y