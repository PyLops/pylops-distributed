import numpy as np
import dask.array as da

from pylops_distributed import LinearOperator


class Roll(LinearOperator):
    r"""Roll along an axis.

    Roll a multi-dimensional array along a specified direction ``dir`` for
    a chosen number of samples (``shift``).

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which rolling is applied.
    shift : :obj:`int`, optional
        Number of samples by which elements are shifted
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array`
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
        (``True``) or not (``False``)

    Raises
    ------
    ValueError
        If ``M`` is different from ``N`` and ``chunks`` is not provided

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Roll` for implementation
    details.

    """
    def __init__(self, N, dims=None, dir=0, shift=1, compute=(False, False),
                 todask=(False, False), dtype='float64'):
        self.N = N
        self.dir = dir
        if dims is None:
            self.dims = (self.N,)
            self.reshape = False
        else:
            if np.prod(dims) != self.N:
                raise ValueError('product of dims must equal N')
            else:
                self.dims = dims
                self.reshape = True
        self.shift = shift
        self.shape = (self.N, self.N)
        self.dtype = dtype
        self.compute = compute
        self.todask = todask
        self.Op = None
        self.explicit = False

    def _matvec(self, x):
        if self.reshape:
            x = da.reshape(x, self.dims)
        y = da.roll(x, shift=self.shift, axis=self.dir)
        y = y.rechunk(x.chunks)
        return y.ravel()

    def _rmatvec(self, x):
        if self.reshape:
            x = da.reshape(x, self.dims)
        y = da.roll(x, shift=-self.shift, axis=self.dir)
        y = y.rechunk(x.chunks)
        return y.ravel()
