import pytest

import numpy as np
import dask.array as da

from numpy.testing import assert_array_equal, assert_array_almost_equal
from pylops_distributed.utils import dottest
from pylops.basicoperators import FirstDerivative, SecondDerivative
from pylops_distributed.basicoperators import FirstDerivative as dFirstDerivative
from pylops_distributed.basicoperators import SecondDerivative as dSecondDerivative


par1 = {'nz': 10, 'ny': 30, 'nx': 40,
        'dz': 1., 'dy': 1., 'dx': 1.} # even with unitary sampling
par2 = {'nz': 10, 'ny': 30, 'nx': 40,
        'dz': 0.4, 'dy': 2., 'dx': 0.5} # even with non-unitary sampling
par3 = {'nz': 11, "ny": 51, 'nx': 61,
        'dz': 1., 'dy': 1., 'dx': 1.} # odd with unitary sampling
par4 = {'nz': 11, "ny": 51, 'nx': 61,
        'dz': 0.4, 'dy': 2., 'dx': 0.5} # odd with non-unitary sampling
par1e = {'nz': 10, 'ny': 30, 'nx': 40,
         'dz': 1., 'dy': 1., 'dx': 1.}  # even with unitary sampling
par2e = {'nz': 10, 'ny': 30, 'nx': 40,
         'dz': 0.4, 'dy': 2., 'dx': 0.5}  # even with non-unitary sampling
par3e = {'nz': 11, "ny": 51, 'nx': 61,
         'dz': 1., 'dy': 1., 'dx': 1.}  # odd with unitary sampling
par4e = {'nz': 11, "ny": 51, 'nx': 61,
         'dz': 0.4, 'dy': 2., 'dx': 0.5}  # odd with non-unitary sampling


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1e), (par2e), (par3e), (par4e)])
def test_FirstDerivative(par):
    """Dot-test and comparison with Pylops for FirstDerivative operator
    """
    np.random.seed(0)

    # 1d
    dD1op = dFirstDerivative(par['nx'], sampling=par['dx'],
                             compute=(True, True), dtype='float32')
    D1op = FirstDerivative(par['nx'], sampling=par['dx'],
                           edge=False, dtype='float32')

    assert dottest(dD1op, par['nx'], par['nx'],
                   chunks=(par['nx'], par['nx']), tol=1e-3)

    x = (par['dx']*np.arange(par['nx'])) ** 2
    x = da.from_array(x, chunks=par['nx'] // 2 + 1)
    dy = dD1op * x
    y = D1op * x.compute()
    assert_array_almost_equal(y[1:-1], dy[1:-1], decimal=1)

    # 2d - derivative on 1st direction
    dD1op = dFirstDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                             dir=0, sampling=par['dy'], compute=(True, True),
                             dtype='float32')
    D1op = FirstDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                           dir=0, sampling=par['dy'], edge=False,
                           dtype='float32')
    assert dottest(dD1op, par['ny']*par['nx'], par['ny']*par['nx'],
                   chunks=(par['ny']*par['nx'], par['ny']*par['nx']),
                   tol=1e-3)

    x = np.outer((par['dy'] * np.arange(par['ny'])) ** 2, np.ones(par['nx']))
    x = da.from_array(x, chunks=(par['ny'] // 2 + 1, par['nx'] // 2 + 1))
    dy = dD1op * x.ravel()
    y = D1op * x.compute().ravel()
    assert_array_almost_equal(y.reshape(par['ny'], par['nx'])[1:-1, 1:-1],
                              dy.reshape(par['ny'], par['nx'])[1:-1, 1:-1],
                              decimal=1)

    # 2d - derivative on 2nd direction
    dD1op = dFirstDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                             dir=1, sampling=par['dx'], compute=(True, True),
                             dtype='float32')

    D1op = FirstDerivative(par['ny'] * par['nx'], dims=(par['ny'], par['nx']),
                           dir=1, sampling=par['dx'], edge=False,
                           dtype='float32')
    assert dottest(dD1op, par['ny'] * par['nx'], par['ny'] * par['nx'],
                   chunks=(par['ny'] * par['nx'], par['ny'] * par['nx']),
                   tol=1e-3)

    x = np.outer((par['dy'] * np.arange(par['ny'])) ** 2, np.ones(par['nx']))
    x = da.from_array(x, chunks=(par['ny'] // 2 + 1, par['nx'] // 2 + 1))
    dy = dD1op * x.ravel()
    y = D1op * x.compute().ravel()
    assert_array_almost_equal(y.reshape(par['ny'], par['nx'])[1:-1, 1:-1],
                              dy.reshape(par['ny'], par['nx'])[1:-1, 1:-1],
                              decimal=1)

    # 3d - derivative on 1st direction
    dD1op = dFirstDerivative(par['nz'] * par['ny'] * par['nx'],
                             dims=(par['nz'], par['ny'], par['nx']),
                             dir=0, sampling=par['dz'], compute=(True, True),
                             dtype='float32')
    D1op = FirstDerivative(par['nz'] * par['ny'] * par['nx'],
                           dims=(par['nz'], par['ny'], par['nx']),
                           dir=0, sampling=par['dz'], edge=False,
                           dtype='float32')
    assert dottest(dD1op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'],
                   chunks=((par['nz'] // 2 + 1) *
                           (par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1),
                           (par['nz'] // 2 + 1) *
                           (par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1)), tol=1e-3)

    x = np.outer((par['dz']*np.arange(par['nz']))**2,
                 np.ones((par['ny'], par['nx']))).reshape(par['nz'],
                                                          par['ny'],
                                                          par['nx'])
    x = da.from_array(x, chunks=(par['nz'] // 2 + 1,
                                 par['ny'] // 2 + 1,
                                 par['nx'] // 2 + 1))
    dy = dD1op * x.ravel()
    y = D1op * x.compute().ravel()
    assert_array_almost_equal(y.reshape(par['nz'], par['ny'], par['nx'])[1:-1, :, :],
                              dy.reshape(par['nz'], par['ny'], par['nx'])[1:-1, :, :],
                              decimal=1)

    # 3d - derivative on 2nd direction
    dD1op = dFirstDerivative(par['nz'] * par['ny'] * par['nx'],
                             dims=(par['nz'], par['ny'], par['nx']),
                             dir=1, sampling=par['dy'], compute=(True, True),
                             dtype='float32')
    D1op = FirstDerivative(par['nz'] * par['ny'] * par['nx'],
                           dims=(par['nz'], par['ny'], par['nx']),
                           dir=1, sampling=par['dy'], edge=False,
                           dtype='float32')
    assert dottest(dD1op, par['nz']*par['ny']*par['nx'],
                   par['nz'] * par['ny'] * par['nx'],
                   chunks=((par['nz'] // 2 + 1) *
                           (par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1),
                           (par['nz'] // 2 + 1) *
                           (par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1)), tol=1e-3)

    x = np.outer(np.outer(np.ones(par['nz']),
                          par['dy'] * np.arange(par['ny']) ** 2),
                 np.ones(par['nx'])).reshape(par['nz'], par['ny'], par['nx'])
    x = da.from_array(x, chunks=(par['nz'] // 2 + 1,
                                 par['ny'] // 2 + 1,
                                 par['nx'] // 2 + 1))
    dy = dD1op * x.ravel()
    y = D1op * x.compute().ravel()
    assert_array_almost_equal(y.reshape(par['nz'], par['ny'], par['nx'])[:, 1:-1, :],
                              dy.reshape(par['nz'], par['ny'], par['nx'])[:, 1:-1, :],
                              decimal=1)

    # 3d - derivative on 3rd direction
    dD1op = dFirstDerivative(par['nz'] * par['ny'] * par['nx'],
                             dims=(par['nz'], par['ny'], par['nx']),
                             dir=2, sampling=par['dx'], compute=(True, True),
                             dtype='float32')
    D1op = FirstDerivative(par['nz'] * par['ny'] * par['nx'],
                           dims=(par['nz'], par['ny'], par['nx']),
                           dir=2, sampling=par['dx'], edge=False,
                           dtype='float32')
    assert dottest(dD1op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'],
                   chunks=((par['nz'] // 2 + 1) *
                           (par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1),
                           (par['nz'] // 2 + 1) *
                           (par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1)), tol=1e-3)

    x = np.outer(np.ones((par['nz'], par['ny'])),
                 (par['dx'] * np.arange(par['nx'])) ** 2).reshape(par['nz'],
                                                                  par['ny'],
                                                                  par['nx'])
    x = da.from_array(x, chunks=(par['nz'] // 2 + 1,
                                 par['ny'] // 2 + 1,
                                 par['nx'] // 2 + 1))
    dy = dD1op * x.ravel()
    y = D1op * x.compute().ravel()
    assert_array_almost_equal(
        y.reshape(par['nz'], par['ny'], par['nx'])[:, :, 1:-1],
        dy.reshape(par['nz'], par['ny'], par['nx'])[:, :, 1:-1],
        decimal=1)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par1e), (par2e), (par3e), (par4e)])
def test_SecondDerivative(par):
    """Dot-test and comparison with Pylops for SecondDerivative operator
    """
    np.random.seed(0)

    x = par['dx'] * np.arange(par['nx'])
    y = par['dy'] * np.arange(par['ny'])
    z = par['dz'] * np.arange(par['nz'])

    xx,yy = np.meshgrid(x, y) # produces arrays of size (ny,nx)
    xxx, yyy, zzz = np.meshgrid(x, y, z) # produces arrays of size (ny,nx,nz)

    # 1d
    dD2op = dSecondDerivative(par['nx'], sampling=par['dx'],
                              compute=(True, True), dtype='float32')
    D2op = SecondDerivative(par['nx'], sampling=par['dx'],
                            edge=False, dtype='float32')
    assert dottest(dD2op, par['nx'], par['nx'],
                   chunks=(par['nx']//2 + 1, par['nx']//2 + 1), tol=1e-3)

    x = da.from_array(x, chunks=par['nx'] // 2 + 1)
    dy = dD2op * x
    y = D2op * x.compute()
    assert_array_almost_equal(y[1:-1], dy[1:-1], decimal=1)

    # 2d - derivative on 1st direction
    dD2op = dSecondDerivative(par['ny'] * par['nx'],
                              dims=(par['ny'], par['nx']),
                              dir=0, sampling=par['dy'],
                              compute=(False, False), dtype='float32')
    D2op = SecondDerivative(par['ny']*par['nx'],
                            dims=(par['ny'], par['nx']),
                            dir=0, sampling=par['dy'],
                            edge=False, dtype='float32')

    assert dottest(dD2op, par['ny']*par['nx'], par['ny']*par['nx'],
                   chunks=((par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1),
                           (par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1)),
                   tol=1e-3)

    xx = da.from_array(xx, chunks=(par['ny'] // 2 + 1, par['nx'] // 2 + 1))
    dy = dD2op * xx.ravel()
    y = D2op * xx.compute().ravel()
    assert_array_almost_equal(y.reshape(par['ny'], par['nx'])[1:-1, 1:-1],
                              dy.reshape(par['ny'], par['nx'])[1:-1, 1:-1],
                              decimal=1)

    # 2d - derivative on 2nd direction
    dD2op = dSecondDerivative(par['ny'] * par['nx'],
                              dims=(par['ny'], par['nx']),
                              dir=1, sampling=par['dy'],
                              compute=(False, False), dtype='float32')
    D2op = SecondDerivative(par['ny']*par['nx'],
                            dims=(par['ny'], par['nx']),
                            dir=1, sampling=par['dx'],
                            edge=False, dtype='float32')

    assert dottest(dD2op, par['ny'] * par['nx'], par['ny'] * par['nx'],
                   chunks=((par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1),
                           (par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1)), tol=1e-3)

    yy = da.from_array(yy, chunks=(par['ny'] // 2 + 1, par['nx'] // 2 + 1))
    dy = dD2op * yy.ravel()
    y = D2op * yy.compute().ravel()
    assert_array_almost_equal(y.reshape(par['ny'], par['nx'])[1:-1, 1:-1],
                              dy.reshape(par['ny'], par['nx'])[1:-1, 1:-1],
                              decimal=1)

    # 3d - derivative on 1st direction
    dD2op = dSecondDerivative(par['nz'] * par['ny'] * par['nx'],
                              dims=(par['ny'], par['nx'], par['nz']),
                              dir=0, sampling=par['dy'],
                              compute=(False, False), dtype='float32')
    D2op = SecondDerivative(par['nz'] * par['ny'] * par['nx'],
                            dims=(par['ny'], par['nx'], par['nz']),
                            dir=0, sampling=par['dy'],
                            edge=False, dtype='float32')
    assert dottest(dD2op, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'],
                   chunks=((par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1),
                           (par['ny'] // 2 + 1) *
                           (par['nx'] // 2 + 1)), tol=1e-3)
    xxx = da.from_array(xxx, chunks=(par['nz'] // 2 + 1,
                                     par['ny'] // 2 + 1,
                                     par['nx'] // 2 + 1))
    dy = dD2op * xxx.ravel()
    y = D2op * xxx.compute().ravel()
    assert_array_almost_equal(y.reshape(par['nz'], par['ny'], par['nx'])[1:-1],
                              dy.reshape(par['nz'], par['ny'], par['nx'])[1:-1],
                              decimal=1)
