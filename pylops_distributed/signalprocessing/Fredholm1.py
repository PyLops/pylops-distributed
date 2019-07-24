import logging
import numpy as np
import dask.array as da

from pylops_distributed import LinearOperator

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class Fredholm1(LinearOperator):
    r"""Fredholm integral of first kind.

    Implement a multi-dimensional Fredholm integral of first kind. Note that if
    the integral is two dimensional, this can be directly implemented using
    :class:`pylops.basicoperators.MatrixMult`. A multi-dimensional
    Fredholm integral can be performed as a :class:`pylops.basicoperators.BlockDiag`
    operator of a series of :class:`pylops.basicoperators.MatrixMult`. However,
    here we take advantage of the structure of the kernel and perform it in a
    more efficient manner.

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel of size
        :math:`[n_{slice} \times n_x \times n_y]`
    nz : :obj:`numpy.ndarray`, optional
        Additional dimension of model
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint
        (``True``) or create ``G^H`` on-the-fly (``False``)
        Note that ``saveGt=True`` will double the amount of required memory
    usematmul : :obj:`bool`, optional
        Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
        (``False``). As it is not possible to define which approach is more
        performant (this is highly dependent on the size of ``G`` and input
        arrays as well as the hardware used in the compution), we advise users
        to time both methods for their specific problem prior to making a
        choice.
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
    A multi-dimensional Fredholm integral of first kind can be expressed as

    .. math::

        d(sl, x, z) = \int{G(sl, x, y) m(sl, y, z) dy}
        \quad \forall sl=1,n_{slice}

    on the other hand its adjoin is expressed as

    .. math::

        m(sl, y, z) = \int{G^*(sl, y, x) d(sl, x, z) dx}
        \quad \forall sl=1,n_{slice}

    In discrete form, this operator can be seen as a block-diagonal
    matrix multiplication:

    .. math::
        \begin{bmatrix}
            \mathbf{G}_{sl1}  & \mathbf{0}       &  ... & \mathbf{0} \\
            \mathbf{0}        & \mathbf{G}_{sl2} &  ... & \mathbf{0} \\
            ...               & ...              &  ... & ...        \\
            \mathbf{0}        & \mathbf{0}       &  ... & \mathbf{G}_{slN}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{m}_{sl1}  \\
            \mathbf{m}_{sl2}  \\
            ...     \\
            \mathbf{m}_{slN}
        \end{bmatrix}

    """
    def __init__(self, G, nz=1, saveGt=True, usematmul=True, dtype='float64'):
        pass

    def _matvec(self, x):
        pass

    def _rmatvec(self, x):
        pass