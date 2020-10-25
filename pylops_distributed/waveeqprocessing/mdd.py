import logging
import numpy as np
import dask.array as da
from math import sqrt
from pylops.waveeqprocessing.mdd import _MDC

from pylops_distributed import LinearOperator
from pylops_distributed.utils import dottest as Dottest
from pylops_distributed import Identity, Transpose
from pylops_distributed.signalprocessing import FFT, Fredholm1
from pylops_distributed.optimization.cg import cgls

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class MDC(LinearOperator):
    r"""Multi-dimensional convolution.

    Apply multi-dimensional convolution between two datasets.
    Model and data should be provided after flattening 2- or 3-dimensional
    arrays of size :math:`[n_t \times n_r (\times n_{vs})]` and
    :math:`[n_t \times n_s (\times n_{vs})]` (or :math:`2*n_t-1` for
    ``twosided=True``), respectively.

    Parameters
    ----------
    G : :obj:`dask.array.ndarray`
        Multi-dimensional convolution kernel in frequency domain of size
        :math:`[n_{fmax} \times n_s \times n_r]`
    nt : :obj:`int`
        Number of samples along time axis
    nv : :obj:`int`
        Number of samples along virtual source axis
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    twosided : :obj:`bool`, optional
        MDC operator has both negative and positive time (``True``) or
        only positive (``False``)
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint of
        :class:`pylops_distributed.signalprocessing.Fredholm1` (``True``) or create
        ``G^H`` on-the-fly (``False``) Note that ``saveGt=True`` will be
        faster but double the amount of required memory
    conj : :obj:`str`, optional
        Perform Fredholm integral computation with complex conjugate of ``G``
    prescaled : :obj:`bool`, optional
        Apply scaling to kernel (``False``) or not (``False``) when performing
        spatial and temporal summations. In case ``prescaled=True``, the
        kernel is assumed to have been pre-scaled when passed to the MDC
        routine.
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array`
    todask : :obj:`tuple`, optional
        Apply :func:`dask.array.from_array` to model and data before applying
        forward and adjoint respectively
    dtype : :obj:`str`, optional
        Type of elements in input array. If ``None``, automatically inferred
        from ``G``

    Notes
    -----
    Refer to :class:`pylops.waveeqprocessing.MDC` for implementation
    details.

    """
    def __init__(self, G, nt, nv, dt=1., dr=1., twosided=True,
                 saveGt=False, conj=False, prescaled=False,
                 chunks=(None, None), compute=(False, False),
                 todask=(False, False), dtype=None):

        if twosided and nt % 2 == 0:
            raise ValueError('nt must be odd number')

        # store G
        self.G = G
        self.nfmax, self.ns, self.nr = self.G.shape
        self.saveGt = saveGt
        if self.saveGt:
            self.GT = (G.transpose((0, 2, 1)).conj()).persist()

        # ensure that nfmax is not bigger than allowed
        self.nfft = int(np.ceil((nt + 1) / 2))
        if self.nfmax > self.nfft:
            self.nfmax = self.nfft
            logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % self.nfmax)

        # store other input parameters
        self.nt, self.nv = nt, nv
        self.dt, self.dr = dt, dr
        self.twosided = twosided
        self.conj = conj
        self.prescaled = prescaled
        self.dims = (self.nt, self.nr, self.nv)
        self.dimsd = (self.nt, self.ns, self.nv)
        self.dimsdf = (self.nfft, self.ns, self.nv)

        # find out dtype of G
        self.cdtype = self.G[0, 0, 0].dtype
        if dtype is None:
            self.dtype = np.real(np.ones(1, dtype=self.cdtype)).dtype
        else:
            self.dtype = dtype

        self.shape = (np.prod(self.dimsd), np.prod(self.dims))
        self.compute = compute
        self.chunks = chunks
        self.todask = todask
        self.Op = None
        self.explicit = False

    def _matvec(self, x):
        # apply forward fft
        x = da.reshape(x, self.dims)
        if self.twosided:
            x = da.fft.ifftshift(x, axes=0)
        y = sqrt(1. / self.nt) * da.fft.rfft(x, n=self.nt, axis=0)
        y = y.astype(self.cdtype)
        y = y[:self.nfmax]

        # apply batched matrix mult
        y = y.rechunk((self.G.chunks[0], self.nr, self.nv))
        if self.conj:
            y = y.conj()
        y = da.matmul(self.G, y)
        if self.conj:
            y = y.conj()
        if not self.prescaled:
            y *= self.dr * self.dt * np.sqrt(self.nt)

        # apply inverse fft
        y = da.pad(y, ((0, self.nfft - self.nfmax), (0, 0), (0, 0)), mode='constant')
        y = y.rechunk(self.dimsdf)
        y = sqrt(self.nt) * da.fft.irfft(y, n=self.nt, axis=0)
        y = y.astype(self.dtype)
        y = da.real(y)
        return y.ravel()

    def _rmatvec(self, x):
        # apply forward fft
        x = da.reshape(x, self.dimsd)
        y = sqrt(1. / self.nt) * da.fft.rfft(x, n=self.nt, axis=0)
        y = y.astype(self.cdtype)
        y = y[:self.nfmax]

        # apply batched matrix mult
        y = y.rechunk((self.G.chunks[0], self.nr, self.nv))
        if self.saveGt:
            if self.conj:
                y = y.conj()
            y = da.matmul(self.GT, y)
            if self.conj:
                y = y.conj()
        else:
            if self.conj:
                y = da.matmul(y.transpose(0, 2, 1), self.G).transpose(0, 2, 1)
            else:
                y = da.matmul(y.transpose(0, 2, 1).conj(), self.G).transpose(0, 2, 1).conj()
        if not self.prescaled:
            y *= self.dr * self.dt * np.sqrt(self.nt)

        # apply inverse fft
        y = da.pad(y, ((0, self.nfft - self.nfmax), (0, 0), (0, 0)), mode='constant')
        y = y.rechunk(self.dimsdf)
        y = sqrt(self.nt) * da.fft.irfft(y, n=self.nt, axis=0)
        if self.twosided:
            y = da.fft.fftshift(y, axes=0)
        y = y.astype(self.dtype)
        y = da.real(y)
        return y.ravel()


def MDD(G, d, dt=0.004, dr=1., nfmax=None, wav=None,
        twosided=True, adjoint=False, dottest=False,
        saveGt=False, add_negative=True, **kwargs_cgls):
    r"""Multi-dimensional deconvolution.

    Solve multi-dimensional deconvolution problem using
    :py:func:`scipy.sparse.linalg.lsqr` iterative solver.

    Parameters
    ----------
    G : :obj:`dask.array.ndarray`
        Multi-dimensional convolution kernel in frequency domain of size
        :math:`[n_{f,max} \times n_s \times n_r]`
    d : :obj:`dask.array.ndarray`
        Data in time domain :math:`[n_t \times n_s (\times n_vs)]` if
        ``twosided=False`` or ``twosided=True`` and ``add_negative=True``
        (with only positive times) or size
        :math:`[2*n_t-1 \times n_s (\times n_vs)]` if ``twosided=True``
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    nfmax : :obj:`int`, optional
        Index of max frequency to include in deconvolution process
    wav : :obj:`numpy.ndarray`, optional
        Wavelet to convolve to the inverted model and psf
        (must be centered around its index in the middle of the array).
        If ``None``, the outputs of the inversion are returned directly.
    twosided : :obj:`bool`, optional
        MDC operator and data both negative and positive time (``True``)
        or only positive (``False``)
    add_negative : :obj:`bool`, optional
        Add negative side to MDC operator and data (``True``) or not
        (``False``)- operator and data are already provided with both positive
        and negative sides. To be used only with ``twosided=True``.
    adjoint : :obj:`bool`, optional
        Compute and return adjoint(s)
    dottest : :obj:`bool`, optional
        Apply dot-test
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint of
        :class:`pylops_distributed.signalprocessing.Fredholm1` (``True``) or
        create ``G^H`` on-the-fly (``False``) Note that ``saveGt=True`` will be
        faster but double the amount of required memory
    **kwargs_cgls
        Arbitrary keyword arguments for
        :py:func:`pylops_distributed.optimization.cg.cgls` solver

    Returns
    -------
    minv : :obj:`dask.array.ndarray`
        Inverted model of size :math:`[n_t \times n_r (\times n_{vs})]`
        for ``twosided=False`` or
        :math:`[2*n_t-1 \times n_r (\times n_vs)]` for ``twosided=True``
    madj : :obj:`dask.array.ndarray`
        Adjoint model of size :math:`[n_t \times n_r (\times n_{vs})]`
        for ``twosided=False`` or
        :math:`[2*n_t-1 \times n_r (\times n_r) ]` for ``twosided=True``

    See Also
    --------
    MDC : Multi-dimensional convolution

    Notes
    -----
    Refer to :class:`pylops.waveeqprocessing.MDD` for implementation
    details. Note that this implementation is currently missing the ``wav``
    and ``causality_precond=False`` options.

    """
    nf, ns, nr = G.shape
    nt = d.shape[0]
    if len(d.shape) == 2:
        nv = 1
    else:
        nv = d.shape[2]
    if twosided:
        if add_negative:
            nt2 = 2 * nt - 1
        else:
            nt2 = nt
            nt = (nt2 + 1) // 2
        nfmax_allowed = int(np.ceil((nt2+1)/2))
    else:
        nt2 = nt
        nfmax_allowed = nt

    # Fix nfmax to be at maximum equal to half of the size of fft samples
    if nfmax is None or nfmax > nfmax_allowed:
        nfmax = nfmax_allowed
        logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % nfmax)

    # Add negative part to data and model
    if twosided and add_negative:
        if nv == 1:
            d = da.concatenate((da.zeros((nt - 1, ns)), d), axis=0)
        else:
            d = da.concatenate((da.zeros((nt - 1, ns, nv)), d), axis=0)
        d = d.rechunk(d.shape)

    # Define MDC linear operator
    MDCop = MDC(G, nt2, nv=nv, dt=dt, dr=dr,
                twosided=twosided, saveGt=saveGt)
    if dottest:
        Dottest(MDCop, nt2 * ns * nv, nt2 * nr * nv, verb=True)

    # Adjoint
    if adjoint:
        madj = MDCop.H * d.flatten()
        madj = da.squeeze(madj.reshape(nt2, nr, nv))

    # Inverse
    minv = cgls(MDCop, d.flatten(), **kwargs_cgls)[0]
    minv = da.squeeze(minv.reshape(nt2, nr, nv))
    #if wav is not None:
    #    minv = sp_convolve1d(minv, wav, axis=-1)

    if adjoint:
        return minv, madj
    else:
        return minv
