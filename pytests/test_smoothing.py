import pytest

import numpy as np
import dask.array as da
from numpy.testing import assert_array_almost_equal

from pylops_distributed.utils import dottest
from pylops.basicoperators import Smoothing1D
from pylops_distributed.basicoperators import Smoothing1D as dSmoothing1D

par1 = {'nz': 10, 'ny': 30, 'nx': 20, 'dir':0} # even, first direction
par2 = {'nz': 11, 'ny': 51, 'nx': 31, 'dir':0} # odd, first direction
par3 = {'nz': 10, 'ny': 30, 'nx': 20, 'dir':1} # even, second direction
par4 = {'nz': 11, 'ny': 51, 'nx': 31, 'dir':1} # odd, second direction


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Smoothing1D(par):
    """Dot-test and comparison with Pylops for smoothing
    """
    # 1d kernel on 1d signal
    D1op = Smoothing1D(nsmooth=5, dims=par['nx'], dtype='float32')
    dD1op = dSmoothing1D(nsmooth=5, dims=par['nx'],
                         compute=(True, True), dtype='float32')
    assert dottest(dD1op, par['nx'], par['nx'], chunks=(par['nx'], par['nx']))

    x = da.from_array(np.random.normal(0, 1, par['nx']),
                      chunks=(par['nx'] // 2 + 1))
    dy = D1op * x
    y = D1op * x.compute()
    assert_array_almost_equal(dy, y, decimal=3)

    # 1d kernel on 2d signal
    D1op = Smoothing1D(nsmooth=5, dims=(par['ny'], par['nx']),
                       dir=par['dir'], dtype='float32')
    dD1op = dSmoothing1D(nsmooth=5, dims=(par['ny'], par['nx']),
                         dir=par['dir'], compute=(True, True), dtype='float32')
    assert dottest(dD1op, par['ny']*par['nx'], par['ny']*par['nx'],
                   chunks=((par['ny']*par['nx']) // 2 + 1,
                           (par['ny'] * par['nx']) // 2 + 1))

    x = da.from_array(np.random.normal(0, 1, (par['ny'], par['nx'])),
                      chunks=(par['ny'] // 2 + 1, par['nx'] // 2 + 1))
    dy = D1op * x.ravel()
    y = D1op * x.ravel().compute()
    assert_array_almost_equal(y, dy, decimal=3)

    # 1d kernel on 3d signal
    D1op = Smoothing1D(nsmooth=5, dims=(par['nz'], par['ny'], par['nx']),
                       dir=par['dir'], dtype='float32')
    dD1op = dSmoothing1D(nsmooth=5, dims=(par['nz'], par['ny'], par['nx']),
                         dir=par['dir'], compute=(True, True), dtype='float32')
    assert dottest(dD1op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'],
                   chunks=((par['nz']*par['ny']*par['nx']) // 2 + 1,
                           (par['nz'] * par['ny'] * par['nx']) // 2 + 1))

    x = da.from_array(np.random.normal(0, 1, (par['nz'], par['ny'], par['nx'])),
                      chunks=(par['nz'] // 2 + 1, par['ny'] // 2 + 1,
                              par['nx'] // 2 + 1))
    dy = D1op * x.ravel()
    y = D1op * x.ravel().compute()
    assert_array_almost_equal(y, dy, decimal=3)
