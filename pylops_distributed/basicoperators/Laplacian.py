import numpy as np
from pylops_distributed.LinearOperator import aslinearoperator
from pylops_distributed.basicoperators import SecondDerivative


def Laplacian(dims, dirs=(0, 1), weights=(1, 1), sampling=(1, 1),
              compute=(False, False), chunks=(None, None),
              todask=(False, False), dtype='float64'):
    r"""Laplacian.

    Apply second-order centered Laplacian operator to a multi-dimensional
    array (at least 2 dimensions are required)

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    dirs : :obj:`tuple`, optional
        Directions along which laplacian is applied.
    weights : :obj:`tuple`, optional
        Weight to apply to each direction (real laplacian operator if
        ``weights=[1,1]``)
    sampling : :obj:`tuple`, optional
        Sampling steps for each direction
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array.array`
    chunks : :obj:`tuple`, optional
        Chunk size for model and data. If provided it will rechunk the model
        before applying the forward pass and the data before applying the
        adjoint pass
    todask : :obj:`tuple`, optional
        Apply :func:`dask.array.from_array` to model and data before applying
        forward and adjoint respectively
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    l2op : :obj:`pylops.LinearOperator`
        Laplacian linear operator

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Laplacian` for implementation
    details.

    """
    l2op = weights[0]*SecondDerivative(np.prod(dims), dims=dims, dir=dirs[0],
                                       sampling=sampling[0],
                                       compute=compute,
                                       chunks=chunks, todask=todask,
                                       dtype=dtype)
    l2op += weights[1]*SecondDerivative(np.prod(dims), dims=dims, dir=dirs[1],
                                        sampling=sampling[1],
                                        compute=compute,
                                        chunks=chunks, todask=todask,
                                        dtype=dtype)
    return aslinearoperator(l2op)
