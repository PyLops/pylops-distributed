import pytest
import numpy as np
import dask.array as da

from numpy.testing import assert_array_almost_equal
from pylops.basicoperators import Diagonal
from pylops_distributed.utils import dottest
from pylops_distributed.basicoperators import Diagonal as dDiagonal

par1 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 0,
        'dtype': 'float64'}  # real
par2 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 1j,
        'dtype': 'complex64'}  # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Diagonal_1dsignal(par):
    """Dot-test and comparision with Pylops for Diagonal operator for 1d signal
    """
    for ddim in (par['nx'], par['nt']):
        d = da.arange(ddim, chunks=ddim//2) + 1. +\
            par['imag'] * (da.arange(ddim, chunks=ddim//2) + 1.)
        dDop = dDiagonal(d, compute=(True, True), dtype=par['dtype'])
        assert dottest(dDop, ddim, ddim, chunks=(ddim//2, ddim//2),
                       complexflag=0 if par['imag'] == 0 else 3)

        x = da.ones(ddim, chunks=ddim//2) + \
            par['imag']*da.ones(ddim, chunks=ddim//2)
        Dop = Diagonal(d.compute(), dtype=par['dtype'])
        dy = dDop * x
        y = Dop * x.compute()
        assert_array_almost_equal(dy, y, decimal=5)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Diagonal_2dsignal(par):
    """Dot-test and inversion for Diagonal operator for 2d signal
    """
    for idim, ddim in enumerate((par['nx'], par['nt'])):
        d = da.arange(ddim, chunks=ddim//2) + 1. +\
            par['imag'] * (da.arange(ddim, chunks=ddim//2) + 1.)

        dDop = dDiagonal(d, dims=(par['nx'], par['nt']),
                         dir=idim, compute=(True, True), dtype=par['dtype'])
        assert dottest(dDop, par['nx']*par['nt'], par['nx']*par['nt'],
                       chunks=(par['nx'] * par['nt'] // 4,
                               par['nx'] * par['nt'] // 4),
                       complexflag=0 if par['imag'] == 0 else 3)

        x = da.ones((par['nx'], par['nt']), chunks=par['nx'] * par['nt'] // 4) + \
            par['imag']*da.ones((par['nx'], par['nt']), chunks=par['nx'] * par['nt'] // 4)
        Dop = Diagonal(d.compute(), dims=(par['nx'], par['nt']),
                       dir=idim, dtype=par['dtype'])
        dy = dDop * x.flatten()
        y = Dop * x.compute().flatten()
        assert_array_almost_equal(dy, y, decimal=5)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Diagonal_3dsignal(par):
    """Dot-test and inversion for Diagonal operator for 3d signal
    """
    for idim, ddim in enumerate((par['ny'], par['nx'], par['nt'])):
        d = da.arange(ddim, chunks=ddim // 2) + 1. +\
            par['imag'] * (da.arange(ddim, chunks=ddim // 2) + 1.)

        dDop = dDiagonal(d, dims=(par['ny'], par['nx'], par['nt']),
                         dir=idim, compute=(True, True), dtype=par['dtype'])
        assert dottest(dDop, par['ny']*par['nx']*par['nt'],
                       par['ny']*par['nx']*par['nt'],
                       chunks=(par['ny'] * par['nx'] * par['nt'] // 4,
                               par['ny'] * par['nx'] * par['nt'] // 4),
                       complexflag=0 if par['imag'] == 0 else 3)

        x = da.ones((par['ny'], par['nx'], par['nt']),
                    chunks=(par['ny'] * par['nx'] * par['nt'] // 4)) + \
            par['imag']*da.ones((par['ny'], par['nx'], par['nt']),
                                chunks=(par['ny'] * par['nx'] * par['nt'] // 4))
        Dop = Diagonal(d.compute(), dims=(par['ny'], par['nx'], par['nt']),
                       dir=idim, dtype=par['dtype'])
        dy = dDop * x.flatten()
        y = Dop * x.compute().flatten()
        assert_array_almost_equal(dy, y, decimal=5)