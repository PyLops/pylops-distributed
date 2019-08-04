import pytest
import numpy as np
import dask.array as da

from numpy.testing import assert_array_almost_equal
from pylops.signalprocessing import Fredholm1
from pylops_distributed.utils import dottest
from pylops_distributed.signalprocessing import Fredholm1 as dFredholm1

par1 = {'nsl': 4, 'ny': 6, 'nx': 4, 'nz': 5,
        'saveGt':True, 'imag': 0, 'dtype': 'float32'}  # real, saved Gt
par2 = {'nsl': 4, 'ny': 6, 'nx': 4, 'nz': 5,
        'saveGt': False, 'imag': 0, 'dtype': 'float32'}  # real, unsaved Gt
par3 = {'nsl': 4, 'ny': 6, 'nx': 4, 'nz': 5,
        'saveGt':True, 'imag': 1j, 'dtype': 'complex64'}  # complex, saved Gt
par4 = {'nsl': 4, 'ny': 6, 'nx': 4, 'nz': 5, 'saveGt': False,
        'imag': 1j, 'dtype': 'complex64'}  # complex, unsaved Gt
#par5 = {'nsl': 4, 'ny': 6, 'nx': 4, 'nz': 1,
#        'saveGt': True, 'imag': 0, 'dtype': 'float32'}  # real, saved Gt, nz=1
#par6 = {'nsl': 4, 'ny': 6, 'nx': 4, 'nz': 1,
#        'saveGt': False, 'imag': 0, 'dtype': 'float32'}  # real, unsaved Gt, nz=1

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2),
                                 (par3), (par4)])
def test_Fredholm1(par):
    """Dot-test and comparison with PyLops for Fredholm1 operator
    """
    _F = \
        da.arange(par['nsl'] * par['nx'] *
                  par['ny']).reshape(par['nsl'],
                                     par['nx'],
                                     par['ny']).rechunk((par['nsl']//2,
                                                         par['nx'],
                                                         par['ny']))
    F = _F - par['imag'] * _F
    dFop = dFredholm1(F, nz=par['nz'], saveGt=par['saveGt'],
                      compute=(True, True), dtype=par['dtype'])
    assert dottest(dFop, par['nsl'] * par['nx'] * par['nz'],
                   par['nsl'] * par['ny'] * par['nz'],
                   chunks=(((par['nsl'] * par['nx'] * par['ny']) // 2),
                           ((par['nsl'] * par['nx'] * par['ny']) // 2)),
                   complexflag=0 if par['imag'] == 0 else 3)

    x = da.ones((par['nsl'], par['ny'], par['nz']),
                chunks=(par['nsl'] // 2, par['ny'], par['nz'])) + \
        par['imag'] * da.ones((par['nsl'], par['ny'], par['nz']),
                              chunks=(par['nsl'] // 2, par['ny'], par['nz']))
    Fop = Fredholm1(F.compute(), nz=par['nz'], saveGt=par['saveGt'],
                    usematmul=True, dtype=par['dtype'])
    dy = dFop * x.ravel()
    y = Fop * x.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=5)
