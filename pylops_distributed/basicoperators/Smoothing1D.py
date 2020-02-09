import numpy as np
from pylops_distributed.signalprocessing import Convolve1D


def Smoothing1D(nsmooth, dims, dir=0, compute=(False, False),
                chunks=(None, None), todask=(False, False), dtype='float64'):
    r"""1D Smoothing.

    Apply smoothing to model (and data) along a specific direction of a
    multi-dimensional array depending on the choice of ``dir``.

    Parameters
    ----------
    nsmooth : :obj:`int`
        Lenght of smoothing operator (must be odd)
    dims : :obj:`tuple` or :obj:`int`
        Number of samples for each dimension
    dir : :obj:`int`, optional
        Direction along which smoothing is applied
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
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Smoothing1D` for implementation
    details.

    """
    if isinstance(dims, int):
        dims = (dims,)
    if nsmooth % 2 == 0:
        nsmooth += 1

    return Convolve1D(np.prod(np.array(dims)), dims=dims, dir=dir,
                      h=np.ones(nsmooth)/float(nsmooth), offset=(nsmooth-1)/2,
                      compute=compute, chunks=chunks, todask=todask,
                      dtype=dtype)
