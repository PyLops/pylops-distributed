import numpy as np
import dask.array as da

from pylops_distributed import LinearOperator


class Restriction(LinearOperator):
    r"""Restriction (or sampling) operator.

    Extract subset of values from input vector at locations ``iava``
    in forward mode and place those values at locations ``iava``
    in an otherwise zero vector in adjoint mode.

    Parameters
    ----------
    M : :obj:`int`
        Number of samples in model.
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Integer indices of available samples for data selection.
    dims : :obj:`list`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which restriction is applied.
    inplace : :obj:`bool`, optional
        Work inplace (``True``) or make a new copy (``False``). By default,
        data is a reference to the model (in forward) and model is a reference
        to the data (in adjoint).
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
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)


    Notes
    -----
    Refer to :class:`pylops.basicoperators.Restriction` for implementation
    details.

    """
    def __init__(self, M, iava, dims=None, dir=0, inplace=True,
                 compute=(False, False), todask=(False, False),
                 dtype='float64'):
        self.M = M
        self.dir = dir
        self.iava = iava

        if dims is None:
            self.N = len(iava)
            self.dims = (self.M, )
            self.reshape = False
        else:
            if np.prod(dims) != self.M:
                raise ValueError('product of dims must equal M!')
            else:
                self.dims = dims # model dimensions
                self.dimsd = list(dims) # data dimensions
                self.dimsd[self.dir] = len(iava)
                self.iavareshape = [1] * self.dir + [len(self.iava)] + \
                                   [1] * (len(self.dims) - self.dir - 1)
                self.N = np.prod(self.dimsd)
                self.reshape = True

        # find out indices to insert zero by da.inster
        diava = np.diff(self.iava) - 1
        diava = np.insert(diava, 0, iava[0])
        diava = np.append(diava, self.dims[self.dir] - self.iava[-1] - 1)
        self.fill = \
            np.concatenate([np.array([i] * nfill)
                            for i, nfill in enumerate(diava)]).astype(np.int)
        self.inplace = inplace
        self.shape = (self.N, self.M)
        self.dtype = np.dtype(dtype)
        self.explicit = False
        self.compute = compute
        self.todask = todask
        self.Op = None

    def _matvec(self, x):
        if not self.inplace:
            x = x.copy()
        if not self.reshape:
            y = x[self.iava]
        else:
            x = da.reshape(x, self.dims)
            y = da.take(x, self.iava, axis=self.dir)
        return y

    def _rmatvec(self, x):
        if not self.inplace:
            x = x.copy()
        if not self.reshape:
            y = da.insert(x, self.fill, 0, axis=-1)
        else:
            x = da.reshape(x, self.dimsd)
            y = da.insert(x, self.fill, 0, axis=self.dir)
        return y