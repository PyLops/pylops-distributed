import numpy as np
import dask.array as da

from dask.array.linalg import inv
from pylops_distributed import LinearOperator


class MatrixMult(LinearOperator):
    r"""Matrix multiplication.

    Simple wrapper to :py:func:`dask.array.dot` for
    an input matrix :math:`\mathbf{A}`.

    Parameters
    ----------
    A : :obj:`dask.array.ndarray`
        Matrix.
    dims : :obj:`tuple`, optional
        Number of samples for each other dimension of model
        (model/data will be reshaped and ``A`` applied multiple times
        to each column of the model/data).
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

    Notes
    -----
    Refer to :class:`pylops.basicoperators.MatrixMult` for implementation
    details.

    """
    def __init__(self, A, dims=None, compute=(False, False),
                 todask=(False, False), dtype='float64'):
        self.A = A
        if dims is None:
            self.reshape = False
            self.shape = A.shape
            self.explicit = True
        else:
            if isinstance(dims, int):
                dims = (dims, )
            self.reshape = True
            self.dims = np.array(dims, dtype=np.int)
            self.shape = (A.shape[0]*np.prod(self.dims),
                          A.shape[1]*np.prod(self.dims))
            self.explicit = False
        self.dtype = np.dtype(dtype)
        self.compute = compute
        self.todask = todask
        self.Op = None

    def _matvec(self, x):
        if self.reshape:
            x = da.reshape(x, da.insert([np.prod(self.dims)], 0,
                           self.A.shape[1]))
        y = self.A.dot(x)
        if self.reshape:
            return y.ravel()
        else:
            return y

    def _rmatvec(self, x):
        if self.reshape:
            x = da.reshape(x, da.insert([np.prod(self.dims)], 0,
                           self.A.shape[0]))
        y = self.A.conj().T.dot(x)
        if self.reshape:
            return y.ravel()
        else:
            return y

    def inv(self):
        r"""Return the inverse of :math:`\mathbf{A}`.

        Returns
        ----------
        Ainv : :obj:`numpy.ndarray`
            Inverse matrix.

        """
        Ainv = inv(self.A)
        return Ainv
