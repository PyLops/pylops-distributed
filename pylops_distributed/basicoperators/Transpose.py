import numpy as np
from pylops import LinearOperator as pLinearOperator
from pylops_distributed import LinearOperator
from pylops.basicoperators import Transpose as _Transpose


class Transpose(LinearOperator):
    r"""Transpose operator.

    Transpose axes of a multi-dimensional array. This operator works with
    flattened input model (or data), which are however multi-dimensional in
    nature and will be reshaped and treated as such in both forward and adjoint
    modes.

    Parameters
    ----------
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    axes : :obj:`tuple`, optional
        Direction along which transposition is applied
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array.array`
    todask : :obj:`tuple`, optional
        Apply :func:`dask.array.from_array` to model and data before applying
        forward and adjoint respectively
    dtype : :obj:`str`, optional
        Type of elements in input array

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
        If ``axes`` contains repeated dimensions (or a dimension is missing)

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Transpose` for implementation
    details.

    """
    def __init__(self, dims, axes, compute=(False, False),
                 todask=(False, False), dtype='float64'):
        Op = _Transpose(dims, axes, dtype=dtype)
        super().__init__(Op.shape, Op.dtype, Op, explicit=False,
                         compute=compute, todask=todask)
