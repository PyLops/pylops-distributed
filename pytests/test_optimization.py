import pytest
import numpy as np
import dask.array as da

from numpy.testing import assert_array_almost_equal
from pylops_distributed.basicoperators import MatrixMult
from pylops_distributed.optimization.cg import cg, cgls

par1 = {'ny': 11, 'nx': 11, 'imag': 0,
        'dtype':'float32'}  # square real
par2 = {'ny': 21, 'nx': 11, 'imag': 0,
        'dtype':'float32'}  # overdetermined real
par1j = {'ny': 11, 'nx': 11, 'imag': 1j,
         'dtype':'complex64'} # square complex
par2j = {'ny': 21, 'nx': 11, 'imag': 1j,
         'dtype':'complex64'} # overdetermined complex


@pytest.mark.parametrize("par", [(par1), (par1j)])
def test_cg(par):
    """Cg solver
    """
    np.random.seed(10)
    x = da.arange(par['nx']) + par['imag'] * np.arange(par['nx'])

    A = np.random.randn(par['ny'], par['nx'])
    A = np.dot(A.T, A)
    Ada = da.from_array(A, chunks=(par['ny'] // 2, par['nx']))
    Aop = MatrixMult(Ada, compute=(False, False))
    y = Aop * x

    # no starting guess
    xinv = cg(Aop, y, niter=par['nx'])[0]
    assert_array_almost_equal(x.compute(), xinv.compute(), decimal=5)

    # with starting guess
    xinv = cg(Aop, y, da.zeros(par['nx']), niter=par['nx'])[0]
    assert_array_almost_equal(x.compute(), xinv.compute(), decimal=5)


@pytest.mark.parametrize("par", [(par2), (par2j)])
def test_cgls(par):
    """Cgls solver
    """
    np.random.seed(10)
    x = da.arange(par['nx']) + par['imag'] * np.arange(par['nx'])

    A = np.random.randn(par['ny'], par['nx']) + \
        par['imag'] * np.random.randn(par['ny'], par['nx'])
    Ada = da.from_array(A, chunks=(par['ny'] // 2, par['nx']))
    Aop = MatrixMult(Ada, compute=(False, False))
    y = Aop * x

    # no starting guess
    xinv = cgls(Aop, y, niter=par['nx'])[0]
    assert_array_almost_equal(x.compute(), xinv.compute(), decimal=5)

    # with starting guess
    xinv = cgls(Aop, y, da.zeros(par['nx']), niter=par['nx'])[0]
    assert_array_almost_equal(x.compute(), xinv.compute(), decimal=5)