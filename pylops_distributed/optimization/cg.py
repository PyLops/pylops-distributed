import dask.array as da


def cg(A, y, x=None, niter=10, compute=False):
    r"""Conjugate gradient

    Solve a system of equations given the square operator ``A`` and data ``y``
    using conjugate gradient iterations.

    Parameters
    ----------
    A : :obj:`pylops_distributed.LinearOperator`
        Operator to invert of size :math:`[N \times N]`
    y : :obj:`dask.array`
        Data of size :math:`[N \times 1]`
    x0 : :obj:`dask.array`, optional
        Initial guess
    niter : :obj:`int`, optional
        Number of iterations
    compute : :obj:`tuple`, optional
        Compute intermediate results at the end of every iteration

    Returns
    -------
    x : :obj:`dask.array`
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
    return x


def cgls(A, y, x=None, niter=10, damp=0., compute=False):
    r"""Conjugate gradient least squares

    Solve an overdetermined system of equations given an operator ``A`` and
    data ``y`` using conjugate gradient iterations.

    Parameters
    ----------
    A : :obj:`pylops_distributed.LinearOperator`
        Operator to invert of size :math:`[N \times N]`
    y : :obj:`dask.array`
        Data of size :math:`[N \times 1]`
    x0 : :obj:`dask.array`, optional
        Initial guess
    niter : :obj:`int`, optional
        Number of iterations
    damp : :obj:`float`, optional
        Damping coefficient
    compute : :obj:`tuple`, optional
        Compute intermediate results at the end of every iteration

    Returns
    -------
    x : :obj:`dask.array`
        Estimated model

    Notes
    -----
    Minimize the following functional using conjugate gradient
    iterations:

    .. math::
        J = || \mathbf{y} -  \mathbf{Ax} ||^2 + \epsilon || \mathbf{x} ||^2

    where :math:`\epsilon` is the damping coefficient.
    """
    if x is None:
        x = da.zeros(A.shape[1], dtype=y.dtype)
    s = y - A.matvec(x)
    r = A.rmatvec(s) - damp * x
    c = r
    q = A.matvec(c)
    kold = r.dot(r)
    if compute:
        x, s, r, c, q, kold = da.compute(x, s, r, c, q, kold)
    for iter in range(niter):
        a = kold / q.dot(q)
        x = x + a * c
        s = s - a * q
        r = A.rmatvec(s) - damp * x
        k = r.dot(r)
        b = k / kold
        c = r + b * c
        q = A.matvec(c)
        kold = k
        if compute:
            x, s, r, a, b, c, q, k = da.compute(x, s, r, a, b, c, q, k)
    return x