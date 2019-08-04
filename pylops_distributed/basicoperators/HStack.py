import numpy as np
import dask.array as da

from scipy.sparse.linalg.interface import _get_dtype
from pylops_distributed import LinearOperator


class HStack(LinearOperator):
    r"""Horizontal stacking.

    Stack a set of N linear operators horizontally.

    Parameters
    ----------
    ops : :obj:`list`
        Linear operators to be stacked
    chunks : :obj:`tuple`, optional
        Chunks for model and data (an array with a single chunk is created
        if ``chunks`` is not provided)
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
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    Refer to :class:`pylops.basicoperators.HStack` for implementation
    details.

    """
    def __init__(self, ops, chunks=None, compute=(False, False),
                 todask=(False, False), dtype=None):
        self.ops = ops
        mops = np.zeros(len(ops), dtype=np.int)
        for iop, op in enumerate(ops):
            mops[iop] = op.shape[1]
        self.mops = mops.sum()
        self.nops = ops[0].shape[0]
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        self.shape = (self.nops, self.mops)
        if dtype is None:
            self.dtype = _get_dtype(ops)
        else:
            self.dtype = np.dtype(dtype)
        self.chunks = (self.nops, self.mops) if chunks is None else chunks
        self.compute = compute
        self.todask = todask
        self.Op = None
        self.explicit = False

    def _matvec(self, x):
        y = da.zeros(self.nops, chunks=self.chunks[0], dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y = y + oper.matvec(
                x[self.mmops[iop]:self.mmops[iop + 1]]).squeeze()
        return y

    def _rmatvec(self, x):
        y = []
        for iop, oper in enumerate(self.ops):
            y.append(oper.rmatvec(x).squeeze())
        y = da.concatenate(y)
        y = y.rechunk(self.chunks[0])
        return y
