import pytest
import numpy as np
import dask.array as da

from numpy.testing import assert_array_equal
from pylops.basicoperators import Roll
from pylops_distributed.utils import dottest
from pylops_distributed.basicoperators import Roll as dRoll

par1 = {'ny': 11, 'nx': 11, 'imag': 0,
        'dtype':'float32'}  # square real
par2 = {'ny': 21, 'nx': 11, 'imag': 0,
        'dtype':'float32'}  # overdetermined real
par1j = {'ny': 11, 'nx': 11, 'imag': 1j,
         'dtype':'complex64'} # square complex
par2j = {'ny': 21, 'nx': 11, 'imag': 1j,
         'dtype':'complex64'} # overdetermined complex
par3 = {'ny': 11, 'nx': 21, 'imag': 0,
        'dtype':'float32'}  # underdetermined real


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Roll1D(par):
    """Dot-test and comparison with PyLops for Roll operator on 1d signal
    """
    np.random.seed(10)
    x = da.arange(par['ny']) + par['imag'] * np.arange(par['ny'])

    dRop = dRoll(par['ny'], shift=2, dtype=par['dtype'])
    Rop = Roll(par['ny'], shift=2, dtype=par['dtype'])
    assert dottest(dRop, par['ny'], par['ny'], chunks=(par['ny'], par['ny']))

    dy = dRop * x
    y = Rop * x.compute()
    assert_array_equal(dy, y)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Roll2D(par):
    """Dot-test and comparison with PyLops for Roll operator on 2d signal
    """
    np.random.seed(10)
    x = {}
    x['0'] = da.outer(np.arange(par['ny']), da.ones(par['nx'])) + \
             par['imag'] * np.outer(da.arange(par['ny']),
                                    da.ones(par['nx']))
    x['1'] = da.outer(da.ones(par['ny']), da.arange(par['nx'])) + \
             par['imag'] * np.outer(da.ones(par['ny']),
                                    da.arange(par['nx']))

    for dir in [0, 1]:
        dRop = dRoll(par['ny'] * par['nx'],
                     dims=(par['ny'], par['nx']),
                     dir=dir, shift=-2, dtype=par['dtype'])
        Rop = Roll(par['ny'] * par['nx'],
                   dims=(par['ny'], par['nx']),
                   dir=dir, shift=-2, dtype=par['dtype'])
        assert dottest(dRop, par['ny'] * par['nx'], par['ny'] * par['nx'],
                       chunks=(par['ny'] * par['nx'], par['ny'] * par['nx']))
        dy = dRop * x[str(dir)].ravel()
        y = Rop * x[str(dir)].compute().ravel()
        assert_array_equal(dy, y)\


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Roll3D(par):
    """Dot-test and comparison with PyLops for Roll operator on 3d signal
    """
    np.random.seed(10)
    x = {}
    x['0'] = np.outer(np.arange(par['ny']),
                      np.ones(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx']) + \
             par['imag'] * np.outer(np.arange(par['ny']),
                                    np.ones(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx'])

    x['1'] = np.outer(np.ones(par['ny']),
                      np.arange(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx']) + \
             par['imag'] * np.outer(np.ones(par['ny']),
                                    np.arange(par['nx']))[:, :, np.newaxis] * \
             np.ones(par['nx'])
    x['2'] = np.outer(np.ones(par['ny']),
                      np.ones(par['nx']))[:, :, np.newaxis] * \
             np.arange(par['nx']) + \
             par['imag'] * np.outer(np.ones(par['ny']),
                                    np.ones(par['nx']))[:, :, np.newaxis] * \
             np.arange(par['nx'])

    for dir in [0, 1, 2]:
        dRop = dRoll(par['ny'] * par['nx'] * par['nx'],
                     dims=(par['ny'], par['nx'], par['nx']),
                     dir=dir, shift=3, dtype=par['dtype'])
        Rop = Roll(par['ny'] * par['nx'] * par['nx'],
                   dims=(par['ny'], par['nx'], par['nx']),
                   dir=dir, shift=3, dtype=par['dtype'])
        assert dottest(dRop, par['ny'] * par['nx'] * par['nx'],
                       par['ny'] * par['nx'] * par['nx'],
                       chunks=(par['ny'] * par['nx'] * par['nx'],
                               par['ny'] * par['nx'] * par['nx']))
        dx = da.from_array(x[str(dir)])
        dy = dRop * dx.ravel()
        y = Rop * x[str(dir)].ravel()
        assert_array_equal(dy, y)
