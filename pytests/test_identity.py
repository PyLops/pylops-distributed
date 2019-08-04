import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal
from pylops_distributed.utils import dottest
from pylops_distributed.basicoperators import Identity as dIdentity

par1 = {'ny': 11, 'nx': 11, 'imag': 0,
        'dtype': 'float32'}  # square real
par2 = {'ny': 21, 'nx': 11, 'imag': 0,
        'dtype': 'float32'}  # overdetermined real
par1j = {'ny': 11, 'nx': 11, 'imag': 1j,
         'dtype': 'complex64'}  # square complex
par2j = {'ny': 21, 'nx': 11, 'imag': 1j,
         'dtype': 'complex64'}  # overdetermined complex
par3 = {'ny': 11, 'nx': 21, 'imag': 0,
        'dtype': 'float32'}  # underdetermined real

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j), (par3)])
def test_Identity_noinplace(par):
    """Dot-test, forward and adjoint for Identity operator (not in place)
    """
    np.random.seed(10)
    Iop = dIdentity(par['ny'], par['nx'], inplace=False,
                    dtype=par['dtype'])
    assert dottest(Iop, par['ny'], par['nx'], chunks=(par['ny'] , par['nx']),
                   complexflag=0 if par['imag'] == 0 else 3)

    x = np.ones(par['nx']) + par['imag'] * np.ones(par['nx'])
    y = Iop*x
    x1 = Iop.H*y

    assert_array_almost_equal(x[:min(par['ny'], par['nx'])],
                              y[:min(par['ny'], par['nx'])], decimal=4)
    assert_array_almost_equal(x[:min(par['ny'], par['nx'])],
                              x1[:min(par['ny'], par['nx'])], decimal=4)

    # change value in x and check it doesn't change in y
    x[0] = 10
    assert x[0] != y[0]
