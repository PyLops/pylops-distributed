import numpy as np
import dask.array as da

from pylops_distributed import LinearOperator


class Identity(LinearOperator):
    r"""Identity operator.

    Simply move model to data in forward model and viceversa in adjoint mode if
    :math:`M = N`. If :math:`M > N` removes last :math:`M - N` elements from
    model in forward and pads with :math:`0` in adjoint. If :math:`N > M`
    removes last :math:`N - M` elements from data in adjoint and pads with
    :math:`0` in forward.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in data (and model, if ``M`` is not provided).
    M : :obj:`int`, optional
        Number of samples in model.
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
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Raises
    ------
    ValueError
        If ``M`` is different from ``N``

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Identity` for implementation
    details.

    """
    def __init__(self, N, M=None, inplace=True, compute=(False, False),
                 todask=(False, False), dtype='float64'):
        M = N if M is None else M
        self.inplace = inplace
        self.shape = (N, M)
        self.dtype = np.dtype(dtype)
        self.compute = compute
        self.todask = todask
        self.Op = None
        self.explicit = False

    def _matvec(self, x):
        if not self.inplace:
            x = x.copy()
        if self.shape[0] == self.shape[1]:
            y = x
        elif self.shape[0] < self.shape[1]:
            y = x[:self.shape[0]]
        else:
            y = da.pad(x, (0, self.shape[0] - self.shape[1]), mode='constant')
        return y

    def _rmatvec(self, x):
        if not self.inplace:
            x = x.copy()
        if self.shape[0] == self.shape[1]:
            y = x
        elif self.shape[0] < self.shape[1]:
            y = da.pad(x, (0, self.shape[1] - self.shape[0]), mode='constant')
        else:
            y = x[:self.shape[1]]
        return y
