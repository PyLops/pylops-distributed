from pylops import Diagonal as pDiagonal
from pylops_distributed import LinearOperator


class Diagonal(LinearOperator):
    r"""Diagonal operator.

    Applies element-wise multiplication of the input vector with a vector
    ``diag`` in forward and with its complex conjugate in
    adjoint mode.

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Diagonal` for implementation
    details.

    """
    def __init__(self, diag, dims=None, dir=0,
                 dtype='float64', compute=(False, False)):
        Op = pDiagonal(diag, dims=dims, dir=dir, dtype=dtype)
        super().__init__(Op.shape, Op.dtype, Op, explicit=False,
                         compute=compute)
