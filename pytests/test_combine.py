import pytest
import numpy as np
import dask.array as da

from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import MatrixMult, VStack, \
    HStack, Block, BlockDiag
from pylops_distributed.utils import dottest
from pylops_distributed.basicoperators import MatrixMult as dMatrixMult
from pylops_distributed.basicoperators import VStack as dVStack
from pylops_distributed.basicoperators import HStack as dHStack
from pylops_distributed.basicoperators import Block as dBlock
from pylops_distributed.basicoperators import BlockDiag as dBlockDiag


par1 = {'ny': 101, 'nx': 101,
        'imag': 0, 'dtype':'float32'}  # square real
par2 = {'ny': 301, 'nx': 101,
        'imag': 0, 'dtype':'float32'}  # overdetermined real
par1j = {'ny': 101, 'nx': 101,
         'imag': 1j, 'dtype':'complex64'} # square imag
par2j = {'ny': 301, 'nx': 101,
         'imag': 1j, 'dtype':'complex64'} # overdetermined imag


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_VStack(par):
    """Dot-test and comparison with pylops for VStack operator
    """
    np.random.seed(10)
    G1 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    x = da.ones(par['nx']) + par['imag']*da.ones(par['nx'])
    dops = [dMatrixMult(G1, dtype=par['dtype']),
            dMatrixMult(G2, dtype=par['dtype'])]
    ops = [MatrixMult(G1.compute(), dtype=par['dtype']),
           MatrixMult(G2.compute(), dtype=par['dtype'])]
    dVop = dVStack(dops, compute=(True, True), dtype=par['dtype'])
    Vop = VStack(ops, dtype=par['dtype'])
    assert dottest(dVop, 2*par['ny'], par['nx'],
                   chunks=(2*par['ny'], par['nx']),
                   complexflag=0 if par['imag'] == 0 else 3)

    dy = dVop * x.ravel()
    y = Vop * x.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=4)


@pytest.mark.parametrize("par", [(par2), (par2j)])
def test_HStack(par):
    """Dot-test and inversion for HStack operator
    """
    np.random.seed(10)
    G1 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    G2 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype('float32')
    x = da.ones(2 * par['nx']) + par['imag'] * da.ones(2 * par['nx'])
    dops = [dMatrixMult(G1, dtype=par['dtype']),
            dMatrixMult(G2, dtype=par['dtype'])]
    ops = [MatrixMult(G1.compute(), dtype=par['dtype']),
           MatrixMult(G2.compute(), dtype=par['dtype'])]
    dHop = dHStack(dops, compute=(True, True), dtype=par['dtype'])
    Hop = HStack(ops, dtype=par['dtype'])
    assert dottest(dHop, par['ny'], 2*par['nx'],
                   chunks=(par['ny'], 2*par['nx']),
                   complexflag=0 if par['imag'] == 0 else 3)

    dy = dHop * x.ravel()
    y = Hop * x.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_Block(par):
    """Dot-test and comparison with pylops for Block operator
    """
    np.random.seed(10)
    G11 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G12 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G21 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G22 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    x = da.ones(2*par['nx']) + par['imag']*da.ones(2*par['nx'])

    dops = [[dMatrixMult(G11, dtype=par['dtype']),
             dMatrixMult(G12, dtype=par['dtype'])],
            [dMatrixMult(G21, dtype=par['dtype']),
             dMatrixMult(G22, dtype=par['dtype'])]]
    ops = [[dMatrixMult(G11.compute(), dtype=par['dtype']),
            dMatrixMult(G12.compute(), dtype=par['dtype'])],
           [dMatrixMult(G21.compute(), dtype=par['dtype']),
            dMatrixMult(G22.compute(), dtype=par['dtype'])]]
    dBop = dBlock(dops, compute=(True, True), dtype=par['dtype'])
    Bop = Block(ops, dtype=par['dtype'])
    assert dottest(dBop, 2*par['ny'], 2*par['nx'],
                   chunks=(2 * par['ny'], 2 * par['nx']),
                   complexflag=0 if par['imag'] == 0 else 3)

    dy = dBop * x.ravel()
    y = Bop * x.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_BlockDiag(par):
    """Dot-test and comparison with pylops for BlockDiag operator
    """
    np.random.seed(10)
    G1 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    G2 = da.random.normal(0, 10, (par['ny'], par['nx'])).astype(par['dtype'])
    x = da.ones(2*par['nx']) + par['imag']*np.ones(2*par['nx'])
    dops = [dMatrixMult(G1, dtype=par['dtype']),
            dMatrixMult(G2, dtype=par['dtype'])]
    ops = [MatrixMult(G1.compute(), dtype=par['dtype']),
           MatrixMult(G2.compute(), dtype=par['dtype'])]
    dBDop = dBlockDiag(dops, compute=(True, True), dtype=par['dtype'])
    BDop = BlockDiag(ops, dtype=par['dtype'])
    assert dottest(dBDop, 2 * par['ny'], 2 * par['nx'],
                   chunks=(2 * par['ny'], 2 * par['nx']),
                   complexflag=0 if par['imag'] == 0 else 3)

    dy = dBDop * x.ravel()
    y = BDop * x.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=4)
