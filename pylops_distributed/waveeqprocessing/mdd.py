import logging
import warnings
import numpy as np

from scipy.sparse.linalg import lsqr
from scipy.ndimage.filters import convolve1d as sp_convolve1d

from pylops_distributed import Diagonal, Identity, Transpose
from pylops_distributed.signalprocessing import FFT, Fredholm1
from pylops.utils import dottest as Dottest
from pylops.optimization.leastsquares import PreconditionedInversion

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)



def _MDC(G, nt, nv, dt=1., dr=1., twosided=True, fast=None,
         dtype=None, transpose=True, saveGt=True, conj=False,
         _Identity=Identity, _Transpose=Transpose, _FFT=FFT,
         _Fredholm1=Fredholm1, args_Identity={}, args_Transpose={},
         args_FFT={}, args_Identity1={}, args_Transpose1={},
         args_FFT1={}, args_Fredholm1={}):
    r"""Multi-dimensional convolution.

    Used to be able to provide operators from different libraries to
    MDC. It operates in the same way as public method
    (PoststackLinearModelling) but has additional input parameters allowing
    passing a different operator and additional arguments to be passed to such
    operator.

    """
    warnings.warn('A new implementation of MDC is provided in v1.5.0. This '
                  'currently affects only the inner working of the operator, '
                  'end-users can continue using the operator in the same way. '
                  'Nevertheless, it is now recommended to start using the '
                  'operator with transpose=True, as this behaviour will '
                  'become default in version v2.0.0 and the behaviour with '
                  'transpose=False will be deprecated.',
                  FutureWarning)

    if twosided and nt % 2 == 0:
        raise ValueError('nt must be odd number')

    # transpose G
    if transpose:
        G = np.transpose(G, axes=(2, 0, 1))

    # create Fredholm operator
    dtype = G[0, 0, 0].dtype
    fdtype = (G[0, 0, 0] + 1j * G[0, 0, 0]).dtype
    Frop = _Fredholm1(dr * dt * np.sqrt(nt) * G, nv, saveGt=saveGt,
                      dtype=fdtype, **args_Fredholm1)
    if conj:
        Frop = Frop.conj()

    # create FFT operators
    nfmax, ns, nr = G.shape
    # ensure that nfmax is not bigger than allowed
    nfft = int(np.ceil((nt + 1) / 2))
    if nfmax > nfft:
        nfmax = nfft
        logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % nfmax)

    Fop = _FFT(dims=(nt, nr, nv), dir=0, real=True,
               fftshift=twosided, dtype=fdtype, **args_FFT)
    F1op = _FFT(dims=(nt, ns, nv), dir=0, real=True,
                fftshift=False, dtype=fdtype, **args_FFT1)

    # create Identity operator to extract only relevant frequencies
    Iop = _Identity(N=nfmax * nr * nv, M=nfft * nr * nv,
                    inplace=True, dtype=dtype, **args_Identity)
    I1op = _Identity(N=nfmax * ns * nv, M=nfft * ns * nv,
                     inplace=True, dtype=dtype, **args_Identity1)
    F1opH = F1op.H
    I1opH = I1op.H

    # create transpose operator
    if transpose:
        dims = [nr, nt] if nv == 1 else [nr, nv, nt]
        axes = (1, 0) if nv == 1 else (2, 0, 1)
        Top = _Transpose(dims, axes, dtype=dtype, **args_Transpose)

        dims = [nt, ns] if nv == 1 else [nt, ns, nv]
        axes = (1, 0) if nv == 1 else (1, 2, 0)
        TopH = _Transpose(dims, axes, dtype=dtype, **args_Transpose1)

    # create MDC operator
    MDCop = F1opH * I1opH * Frop * Iop * Fop
    print(Fop.todask)
    if transpose:
        MDCop = TopH * MDCop * Top
    return MDCop


def MDC(G, nt, nv, dt=1., dr=1., twosided=True,
        saveGt=True, conj=False, todask=(False, False)):
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
    dtype : :obj:`str`, optional
        *Deprecated*, will be removed in v2.0.0
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint of
        :class:`pylops_distributed.signalprocessing.Fredholm1` (``True``) or create
        ``G^H`` on-the-fly (``False``) Note that ``saveGt=True`` will be
        faster but double the amount of required memory
    conj : :obj:`str`, optional
        Perform Fredholm integral computation with complex conjugate of ``G``
    todask : :obj:`tuple`, optional
        Apply :func:`dask.array.from_array` to model and data before applying
        forward and adjoint respectively

    See Also
    --------
    MDD : Multi-dimensional deconvolution

    Notes
    -----
    Refer to :class:`pylops.waveeqprocessing.MDC` for implementation
    details.

    """
    return _MDC(G, nt, nv, dt=dt, dr=dr, twosided=twosided,
                transpose=False, saveGt=saveGt, conj=conj,
                _Identity=Identity, _Transpose=Transpose,
                _FFT=FFT, _Fredholm1=Fredholm1,
                args_Fredholm1={'chunks': ((G.chunks[0], G.shape[0], nv),
                                           (G.chunks[0], G.shape[0],
                                            G.shape[1]))},
                args_FFT={'chunks': (None, (nt, G.shape[0], nv)),
                          'todask':(todask[0], False)},
                args_FFT1={'chunks': (None, (nt, G.shape[0], nv))},
                args_Identity={'chunks': ((nt * nv * G.shape[0]),
                                          (nt * nv * G.shape[0]))},
                args_Identity1={'chunks': ((nt * nv * G.shape[0]),
                                           (nt * nv * G.shape[0])),
                                'todask': (False, todask[1])})
