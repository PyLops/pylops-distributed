import logging
from pylops.waveeqprocessing.mdd import _MDC

from pylops_distributed import Identity, Transpose
from pylops_distributed.signalprocessing import FFT, Fredholm1

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def MDC(G, nt, nv, dt=1., dr=1., twosided=True,
        saveGt=True, conj=False, prescaled=False,
        compute=(False, False), todask=(False, False)):
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

    Notes
    -----
    Refer to :class:`pylops.waveeqprocessing.MDC` for implementation
    details.

    """
    return _MDC(G, nt, nv, dt=dt, dr=dr, twosided=twosided,
                transpose=False, saveGt=saveGt, conj=conj, prescaled=prescaled,
                _Identity=Identity, _Transpose=Transpose,
                _FFT=FFT, _Fredholm1=Fredholm1,
                args_Fredholm1={'chunks': ((G.chunks[0], G.shape[2], nv),
                                           (G.chunks[0], G.shape[1], nv))},
                args_FFT={'chunks': ((nt, G.shape[2], nv),
                                     (nt, G.shape[2], nv)),
                          'todask':(todask[0], False),
                          'compute': (False, compute[1])},
                args_FFT1={'chunks': ((nt, G.shape[1], nv),
                                      (nt, G.shape[1], nv)),
                           'todask': (todask[1], False),
                           'compute':(False, compute[0])})
