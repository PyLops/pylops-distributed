import pytest

import numpy as np
import dask.array as da

from numpy.testing import assert_array_almost_equal
from scipy.signal import triang

from pylops.signalprocessing import Convolve1D
from pylops_distributed.utils import dottest
from pylops_distributed.signalprocessing import Convolve1D as dConvolve1D

# filters
nfilt = (5, 6, 5)
h1 = triang(nfilt[0], sym=True)
h2 = np.outer(triang(nfilt[0], sym=True), triang(nfilt[1], sym=True))
h3 = np.outer(np.outer(triang(nfilt[0], sym=True), triang(nfilt[1], sym=True)),
              triang(nfilt[2], sym=True)).reshape(nfilt)

par1_1d = {'nz': 21, 'ny': 51, 'nx': 31, 'offset': nfilt[0] // 2,
           'dir': 0}  # zero phase, first direction
par2_1d = {'nz': 21, 'ny': 61, 'nx': 31, 'offset': 0,
           'dir': 0}  # non-zero phase, first direction
par3_1d = {'nz': 21, 'ny': 51, 'nx': 31,
           'offset': nfilt[0] // 2,
           'dir': 1}  # zero phase, second direction
par4_1d = {'nz': 21, 'ny': 61, 'nx': 31, 'offset': nfilt[0] // 2 - 1,
           'dir': 1}  # non-zero phase, second direction
par5_1d = {'nz': 21, 'ny': 51, 'nx': 31, 'offset': nfilt[0] // 2,
           'dir': 2}  # zero phase, third direction
par6_1d = {'nz': 21, 'ny': 61, 'nx': 31, 'offset': nfilt[0] // 2 - 1,
           'dir': 2}  # non-zero phase, third direction

par1_2d = {'nz': 21, 'ny': 51, 'nx': 31,
           'offset': (nfilt[0] // 2, nfilt[1] // 2),
           'dir': 0}  # zero phase, first direction
par2_2d = {'nz': 21, 'ny': 61, 'nx': 31,
           'offset': (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1),
           'dir': 0}  # non-zero phase, first direction
par3_2d = {'nz': 21, 'ny': 51, 'nx': 31,
           'offset': (nfilt[0] // 2, nfilt[1] // 2),
           'dir': 1}  # zero phase, second direction
par4_2d = {'nz': 21, 'ny': 61, 'nx': 31,
           'offset': (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1),
           'dir': 1}  # non-zero phase, second direction
par5_2d = {'nz': 21, 'ny': 51, 'nx': 31,
           'offset': (nfilt[0] // 2, nfilt[1] // 2),
           'dir': 2}  # zero phase, third direction
par6_2d = {'nz': 21, 'ny': 61, 'nx': 31,
           'offset': (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1),
           'dir': 2}  # non-zero phase, third direction

par1_3d = {'nz': 21, 'ny': 51, 'nx': 31, 'nt': 5,
           'offset': (nfilt[0] // 2, nfilt[1] // 2, nfilt[2] // 2),
           'dir': 0}  # zero phase, all directions
par2_3d = {'nz': 21, 'ny': 61, 'nx': 31, 'nt': 5,
           'offset': (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1, nfilt[2] // 2 + 1),
           'dir': 0}  # non-zero phase, first direction


@pytest.mark.parametrize("par", [(par1_1d), (par2_1d),
                                 (par3_1d), (par4_1d),
                                 (par5_1d), (par6_1d)])
def test_Convolve1D(par):
    """Dot-test and comparison with Pylops for Convolve1D operator
    """
    np.random.seed(10)
    # 1D
    if par['dir'] == 0:
        Cop = Convolve1D(par['nx'], h=h1, offset=par['offset'],
                         dtype='float32')
        dCop = dConvolve1D(par['nx'], h=h1, offset=par['offset'],
                           compute=(True, True), dtype='float32')
        assert dottest(dCop, par['nx'], par['nx'],
                       chunks=(par['nx']//2 + 1, par['nx']//2 + 1))

        x = np.random.normal(0., 1., par['nx'])
        x = da.from_array(x, chunks=par['nx']//2 + 1)
        dy = dCop * x
        y = Cop * x.compute()
        assert_array_almost_equal(y, dy, decimal=1)

    # 1D on 2D
    if par['dir'] < 2:
        Cop = Convolve1D(par['ny'] * par['nx'], h=h1, offset=par['offset'],
                         dims=(par['ny'], par['nx']), dir=par['dir'],
                         dtype='float32')
        dCop = dConvolve1D(par['ny'] * par['nx'], h=h1, offset=par['offset'],
                           dims=(par['ny'], par['nx']), dir=par['dir'],
                           compute=(True, True),
                           chunks=((par['ny'] // 2 + 1, par['nx'] // 2 + 1),
                                   (par['ny'] // 2 + 1, par['nx'] // 2 + 1)),
                           dtype='float32')
        assert dottest(dCop, par['ny'] * par['nx'], par['ny'] * par['nx'],
                       chunks=(par['ny'] * par['nx'], par['ny'] * par['nx']))

        x = np.random.normal(0., 1., (par['ny'], par['nx']))
        x = da.from_array(x, chunks=(par['ny'] // 2 + 1, par['nx'] // 2 + 1)).flatten()
        dy = dCop * x
        y = Cop * x.compute()
        assert_array_almost_equal(y, dy, decimal=1)

    # 1D on 3D
    Cop = Convolve1D(par['nz'] * par['ny'] * par['nx'], h=h1,
                     offset=par['offset'],
                     dims=(par['nz'], par['ny'], par['nx']), dir=par['dir'],
                     dtype='float32')
    dCop = dConvolve1D(par['nz'] * par['ny'] * par['nx'], h=h1,
                       offset=par['offset'],
                       dims=(par['nz'], par['ny'], par['nx']), dir=par['dir'],
                       compute=(True, True),
                       chunks=((par['nz'] // 2 + 1,
                               par['ny'] // 2 + 1,
                               par['nx'] // 2 + 1),
                               (par['nz'] // 2 + 1,
                                par['ny'] // 2 + 1,
                                par['nx'] // 2 + 1)), dtype='float32')
    assert dottest(dCop, par['nz'] * par['ny'] * par['nx'],
                   par['nz'] * par['ny'] * par['nx'],
                   chunks=(par['nz'] * par['ny'] * par['nx'],
                           par['nz'] * par['ny'] * par['nx']))

    x = np.random.normal(0., 1., (par['nz'], par['ny'], par['nx']))
    x = da.from_array(x, chunks=(par['nz'] // 2 + 1,
                                 par['ny'] // 2 + 1,
                                 par['nx'] // 2 + 1))
    dy = dCop * x.flatten()
    y = Cop * x.compute().flatten()
    assert_array_almost_equal(y, dy, decimal=1)
