import dask.array as da


def cg(A, y, x=None, niter=10, compute=False):
    r"""Conjugate gradient

    Solve a system of equations given the square operator ``A`` and data ``y``
    using conjugate gradient iterations.

    Parameters
    ----------
    A : :obj:`pylops_gpu.LinearOperator`
        Operator to invert of size :math:`[N \times N]`
    y : :obj:`torch.Tensor`
        Data of size :math:`[N \times 1]`
    x0 : :obj:`torch.Tensor`, optional
        Initial guess
    niter : :obj:`int`
        Number of iterations
    compute : :obj:`tuple`, optional
        Compute intermediate results at the end of every iteration

    Returns
    -------
    x : :obj:`torch.Tensor`
        Estimated model

    """
    if x is None:
        x = da.zeros_like(y)
    r = y - A.matvec(x)
    d = r
    kold = r.dot(r)
    if compute:
        x, d, r, kold = da.compute(x, d, r, kold)
    for iter in range(niter):
        a = kold / d.dot(A.matvec(d))
        x = x + a * d
        r = r - a * A.matvec(d)
        k = r.dot(r)
        b = k / kold
        d = r + b * d
        if compute:
            x, d, r, a, b, k = da.compute(x, d, r, a, b, k)
        kold = k
        # print('x', x)
        # print('b', b)
    return x