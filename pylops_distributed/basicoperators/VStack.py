import dask
import numpy as np
import dask.array as da

from scipy.sparse.linalg.interface import _get_dtype
from pylops_distributed import LinearOperator


class VStack(LinearOperator):
    r"""Vertical stacking.

    Stack a set of N linear operators vertically.

    Parameters
    ----------
    ops : :obj:`list`
        Linear operators to be stacked. Operators must be
        of :obj:`pylops_distributed.LinearOperator` type for
        ``usedelayed=False`` and :obj:`pylops.LinearOperator`
        for ``usedelayed=True``
    chunks : :obj:`tuple`, optional
        Chunks for model and data (an array with a single chunk is created
        if ``chunks`` is not provided)
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array`
    todask : :obj:`tuple`, optional
        Apply :func:`dask.array.from_array` to model and data before applying
        forward and adjoint respectively
    usedelayed : :obj:`bool`, optional
        Use :func:`dask.delayed` to parallelize over the N operators. Note that
        when this is enabled the input model and data should be passed as
        :obj:`numpy.ndarray`
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
    Refer to :class:`pylops.basicoperators.VStack` for implementation
    details.

    """
    def __init__(self, ops, chunks=None, compute=(False, False),
                 todask=(False, False), usedelayed=False, dtype=None):
        self.ops = ops
        nops = np.zeros(len(ops), dtype=np.int)
        for iop, oper in enumerate(ops):
            nops[iop] = oper.shape[0]
        self.nops = nops.sum()
        self.mops = ops[0].shape[1]
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.shape = (self.nops, self.mops)
        if dtype is None:
            self.dtype = _get_dtype(ops)
        else:
            self.dtype = np.dtype(dtype)
        self.chunks = (self.nops, self.mops) if chunks is None else chunks
        self.compute = compute
        self.todask = todask
        self.usedelayed = usedelayed
        if self.usedelayed:
            self.todask = (False, False)
        self.Op = None
        self.explicit = False

    def _matvec_dask(self, x):
        y = []
        for iop, oper in enumerate(self.ops):
            y.append(oper.matvec(x).squeeze())
        y = da.concatenate(y)
        y = y.rechunk(self.chunks[0])
        return y

    def _matvec_delayed(self, x):
        ytmp = []
        for iop, oper in enumerate(self.ops):
            ytmp.append(dask.delayed(oper.matvec)(x))
        y = dask.delayed(np.concatenate, traverse=False)(ytmp).compute()
        return y

    def _rmatvec_dask(self, x):
        y = da.zeros(self.mops, chunks=self.chunks[1], dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y = y + oper.rmatvec(
                x[self.nnops[iop]:self.nnops[iop + 1]]).squeeze()
        return y

    def _rmatvec_delayed(self, x):
        ytmp = []
        for iop, oper in enumerate(self.ops):
            ytmp.append(dask.delayed(oper.rmatvec)(
                x[self.nnops[iop]:self.nnops[iop + 1]]))
        y = dask.delayed(np.sum, traverse=False)(ytmp, axis=0).compute()
        return y

    def _matvec(self, x):
        if self.usedelayed:
            y = self._matvec_delayed(x)
        else:
            y = self._matvec_dask(x)
        return y

    def _rmatvec(self, x):
        if self.usedelayed:
            y = self._rmatvec_delayed(x)
        else:
            y = self._rmatvec_dask(x)
        return y
