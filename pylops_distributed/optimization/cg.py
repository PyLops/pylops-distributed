import numpy as np
import dask.array as da


def cg(A, y, x=None, niter=10, tol=1e-5, compute=False,
       client=None):
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
    tol : :obj:`float`, optional
        Tolerance on residual norm
    compute : :obj:`tuple`, optional
        Compute intermediate results at the end of every iteration
    client : :obj:`dask.distributed.client.Client`, optional
        Dask client. If provided when ``compute=None`` each iteration
        is persisted. This is the preferred method to avoid repeating
        computations.

    Returns
    -------
    x : :obj:`dask.array`
        Estimated model
    iter : :obj:`int`
        Number of executed iterations

    Notes
    -----
    Solve the the following problem using conjugate gradient
    iterations:

    .. math::
        \mathbf{y} = \mathbf{Ax}

    Note that early stopping based on ``tol`` is activated only when
    client is provided or ``compute=True``. The formed approach is preferred
    as it avoid repeating computations along the compute tree.

    """
    if x is None:
        x = da.zeros_like(y)
        r = y.copy()
    else:
        r = y - A.matvec(x)
    d = r.copy()
    kold = r.dot(r.conj())
    if compute:
        x, d, r, kold = da.compute(x, d, r, kold)
    elif client is not None:
        x, d, r, kold = client.persist([x, d, r, kold])
    iit = 0
    for iit in range(niter):
        if compute or client is not None:
            if np.abs(kold) < tol:
                break
        Ad = A.matvec(d)
        a = kold / d.dot(Ad.conj())
        x = x + a * d
        r = r - a * Ad
        k = r.dot(r.conj())
        b = k / kold
        d = r + b * d
        if compute:
            x, d, r, Ad, a, b, k = da.compute(x, d, r, Ad, a, b, k)
        elif client is not None:
            x, d, r, Ad, a, b, k = client.persist([x, d, r, Ad, a, b, k])
        kold = k
    return x, iit


def cgls(A, y, x=None, niter=10, damp=0., tol=1e-4,
         compute=False, client=None):
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
    tol : :obj:`float`, optional
        Tolerance on residual norm
    compute : :obj:`tuple`, optional
        Compute intermediate results at the end of every iteration
    client : :obj:`dask.distributed.client.Client`, optional
        Dask client. If provided when ``compute=None`` each iteration
        is persisted. This is the preferred method to avoid repeating
        computations.

    Returns
    -------
    x : :obj:`dask.array`
        Estimated model
    iit : :obj:`int`
        Number of executed iterations

    Notes
    -----
    Minimize the following functional using conjugate gradient
    iterations:

    .. math::
        J = || \mathbf{y} -  \mathbf{Ax} ||^2 + \epsilon || \mathbf{x} ||^2

    where :math:`\epsilon` is the damping coefficient.

    Note that early stopping based on ``tol`` is activated only when
    client is provided or ``compute=True``. The formed approach is preferred
    as it avoid repeating computations along the compute tree.

    """
    if x is None:
        x = da.zeros(A.shape[1], dtype=y.dtype)
        s = y.copy()
        r = A.rmatvec(s)
    else:
        s = y - A.matvec(x)
        r = A.rmatvec(s) - damp * x
    c = r.copy()
    q = A.matvec(c)
    kold = r.dot(r.conj())
    if compute:
        x, s, r, c, q, kold = da.compute(x, s, r, c, q, kold)
    elif client is not None:
        x, s, r, c, q, kold = client.persist([x, s, r, c, q, kold])
    iit = 0
    for iit in range(niter):
        if compute or client is not None:
            if np.abs(kold) < tol:
                break
        a = kold / (q.dot(q.conj()) + damp * c.dot(c.conj()))
        x = x + a * c
        s = s - a * q
        r = A.rmatvec(s) - damp * x
        k = r.dot(r.conj())
        b = k / kold
        c = r + b * c
        q = A.matvec(c)
        if compute:
            x, s, r, a, b, c, q, k = da.compute(x, s, r, a, b, c, q, k)
        elif client is not None:
            x, s, r, a, b, c, q, k = client.persist([x, s, r, a, b, c, q, k])
        kold = k
    return x, iit