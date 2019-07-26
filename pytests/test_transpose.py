import pytest

import numpy as np
import dask.array as da
from numpy.testing import assert_equal

from pylops.basicoperators import Transpose
from pylops_distributed.utils import dottest
from pylops_distributed.basicoperators  import Transpose as dTranspose

par1 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 0,
        'dtype': 'float32'}  # real
par2 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 1j,
        'dtype': 'complex64'}  # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Transpose_2dsignal(par):
    """Dot-test and comparison with pylops for Transpose operator
    for 2d signals
    """
    dims = (par['ny'], par['nx'])
    x = np.arange(par['ny']*par['nx']).reshape(dims) + \
        par['imag'] * np.arange(par['ny']*par['nx']).reshape(dims)
    x = da.from_array(x)

    dTop = dTranspose(dims=dims, axes=(1, 0), dtype=par['dtype'])
    Top = Transpose(dims=dims, axes=(1, 0), dtype=par['dtype'])
    assert dottest(dTop, np.prod(dims), np.prod(dims),
                   chunks=(np.prod(dims) // 2, np.prod(dims) // 2),
                   complexflag=0 if par['imag'] == 0 else 3)
    dy = dTop * x.ravel()
    y = Top * x.ravel().compute()
    assert_equal(dy, y)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Transpose_3dsignal(par):
    """Dot-test and comparison with pylops for Transpose operator
    for 3d signals
    """
    dims = (par['ny'], par['nx'], par['nt'])
    x = np.arange(par['ny']*par['nx']*par['nt']).reshape(dims) + \
        par['imag'] * np.arange(par['ny']*par['nx']*par['nt']).reshape(dims)
    x = da.from_array(x)

    dTop = dTranspose(dims=dims, axes=(2, 1, 0))
    Top = Transpose(dims=dims, axes=(2, 1, 0))
    assert dottest(dTop, np.prod(dims), np.prod(dims),
                   chunks=(np.prod(dims) // 2, np.prod(dims) // 2),
                   complexflag = 0 if par['imag'] == 0 else 3)
    dy = dTop * x.ravel()
    y = Top * x.ravel().compute()
    assert_equal(dy, y)
