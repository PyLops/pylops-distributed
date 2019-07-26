from pylops import Diagonal as pDiagonal
from pylops_distributed import LinearOperator


class Diagonal(LinearOperator):
    r"""Diagonal operator.

    Applies element-wise multiplication of the input vector with the vector
    ``diag`` in forward and with its complex conjugate in adjoint mode.

    This operator can also broadcast; in this case the input vector is
    reshaped into its dimensions ``dims`` and the element-wise multiplication
    with ``diag`` is perfomed on the direction ``dir``. Note that the
    vector ``diag`` will need to have size equal to ``dims[dir]``.

    Parameters
    ----------
    diag : :obj:`dask.array.ndarray`
        Vector to be used for element-wise multiplication.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which multiplication is applied.
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array.array`
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
    Refer to :class:`pylops.basicoperators.Diagonal` for implementation
    details.

    """
    def __init__(self, diag, dims=None, dir=0,
                 compute=(False, False), todask=(False, False),
                 dtype='float64'):
        Op = pDiagonal(diag, dims=dims, dir=dir, dtype=dtype)
        super().__init__(Op.shape, Op.dtype, Op, explicit=False,
                         compute=compute, todask=todask)
