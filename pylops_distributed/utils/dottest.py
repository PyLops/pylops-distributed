import numpy as np
import dask.array as da


def dottest(Op, nr, nc, chunks, tol=1e-6, complexflag=0,
            raiseerror=True, verb=False):
    r"""Dot test.

    Generate random vectors :math:`\mathbf{u}` and :math:`\mathbf{v}`
    and perform dot-test to verify the validity of forward and adjoint operators.
    This test can help to detect errors in the operator implementation.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear operator to test.
    nr : :obj:`int`
        Number of rows of operator (i.e., elements in data)
    nc : :obj:`int`
        Number of columns of operator (i.e., elements in model)
    chunks : :obj:`tuple`, optional
        Chunks for data and model
    tol : :obj:`float`, optional
        Dottest tolerance
    complexflag : :obj:`bool`, optional
        generate random vectors with real (0) or complex numbers
        (1: only model, 2: only data, 3:both)
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity

    Raises
    ------
    ValueError
        If dot-test is not verified within chosen tolerance.

    Notes
    -----
    A dot-test is mathematical tool used in the development of numerical linear operators.

    More specifically, a correct implementation of forward and adjoint for
    a linear operator should verify the following *equality*
    within a numerical tolerance:

    .. math::
        (\mathbf{Op}*\mathbf{u})^H*\mathbf{v} = \mathbf{u}^H*(\mathbf{Op}^H*\mathbf{v})

    """
    if complexflag in (0, 2):
        u = da.random.random(nc, chunks=chunks[1])
    else:
        u = da.random.random(nc, chunks=chunks[1]) + \
            1j*da.random.random(nc, chunks=chunks[1])

    if complexflag in (0, 1):
        v = da.random.random(nr, chunks=chunks[0])
    else:
        v = da.random.random(nr, chunks=chunks[0]) + \
            1j*da.random.random(nr, chunks=chunks[0])

    y = Op.matvec(u)   # Op * u
    x = Op.rmatvec(v)  # Op'* v

    # compute
    if not Op.compute[0]:
        y.compute()
    if not Op.compute[1]:
        x.compute()

    if complexflag == 0:
        yy = np.dot(y, v.compute()) # (Op  * u)' * v
        xx = np.dot(u.compute(), x) # u' * (Op' * v)
    else:
        yy = np.vdot(y, v.compute()) # (Op  * u)' * v
        xx = np.vdot(u.compute(), x) # u' * (Op' * v)

    if complexflag == 0:
        if np.abs((yy-xx)/((yy+xx+1e-15)/2)) < tol:
            if verb: print('Dot test passed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return True
        else:
            if raiseerror:
                raise ValueError('Dot test failed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                                 % (yy, xx))
            if verb: print('Dot test failed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return False
    else:
        checkreal = np.abs((np.real(yy) - np.real(xx)) /
                           ((np.real(yy) + np.real(xx)+1e-15) / 2)) < tol
        checkimag = np.abs((np.real(yy) - np.real(xx)) /
                           ((np.real(yy) + np.real(xx)+1e-15) / 2)) < tol

        if checkreal and checkimag:
            if verb: print('Dot test passed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return True
        else:
            if raiseerror:
                raise ValueError('Dot test failed, v^H(Opu)=%f - u^H(Op^Hv)=%f'
                                 % (yy, xx))
            if verb: print('Dot test failed, v^H(Opu)=%f - u^H(Op^Hv)=%f'
                           % (yy, xx))
            return False
