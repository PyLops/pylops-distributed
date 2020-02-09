import numpy as np

from pylops_distributed import LinearOperator
from pylops_distributed.signalprocessing import Convolve1D


class SecondDerivative(LinearOperator):
    r"""Second derivative.

    Apply second-order centered second derivative.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which smoothing is applied.
    sampling : :obj:`float`, optional
        Sampling step ``dx``.
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
    Refer to :class:`pylops.basicoperators.SecondDerivative` for implementation
    details.

    """
    def __init__(self, N, dims=None, dir=0, sampling=1.,
                 compute=(False, False), chunks=(None, None),
                 todask=(False, False), dtype='float64'):
        h = np.array([1., -2, 1.], dtype=dtype) / sampling**2
        self.compute = compute
        self.chunks = chunks
        self.todask = todask
        self.shape = (N, N)
        self.dtype = dtype
        self.explicit = False
        self.Op = Convolve1D(N, h, offset=1, dims=dims, dir=dir,
                             compute=compute, chunks=chunks,
                             todask=todask, dtype='float64')
