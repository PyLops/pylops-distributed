import pytest
import numpy as np
import dask.array as da

from numpy.testing import assert_array_almost_equal
from scipy.sparse.linalg import lsqr
from pylops.signalprocessing import FFT
from pylops_distributed.utils import dottest
from pylops_distributed.signalprocessing import FFT as dFFT

par1 = {'nt': 101, 'nx': 31, 'ny': 10,
        'nfft': None, 'real': False,
        'ffthshift': False} # nfft=nt, complex input
par2 = {'nt': 101, 'nx': 31, 'ny': 10,
        'nfft': 256, 'real': False,
        'ffthshift': False} # nfft>nt, complex input
par3 = {'nt': 101, 'nx': 31, 'ny': 10,
        'nfft': None, 'real': True,
        'ffthshift': False}  # nfft=nt, real input
par4 = {'nt': 101, 'nx': 31, 'ny': 10,
        'nfft': 256, 'real': True,
        'ffthshift': False}  # nfft>nt, real input
par5 = {'nt': 101, 'nx': 31, 'ny': 10,
        'nfft': 256, 'real': True,
        'ffthshift': True}  # nfft>nt, real input and fftshift

@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5)])
def test_FFT_1dsignal(par):
    """Dot-test and comparision with pylops FFT operator for 1d signal
    """
    dt, f0 = 0.005, 10
    t = np.arange(par['nt']) * dt
    x = da.from_array(np.sin(2 * np.pi * f0 * t))
    nfft = par['nt'] if par['nfft'] is None else par['nfft']
    dFFTop = dFFT(dims=[par['nt']], nfft=nfft, sampling=dt, real=par['real'],
                  chunks = (par['nt'], nfft))
    FFTop = FFT(dims=[par['nt']], nfft=nfft, sampling=dt, real=par['real'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        dFFTop = dFFTop.H * dFFTop
        FFTop = FFTop.H * FFTop
        assert dottest(dFFTop, par['nt'], par['nt'],
                       chunks=(par['nt'], nfft),
                       complexflag=0)
    else:
        assert dottest(dFFTop, nfft, par['nt'],
                       chunks=(par['nt'], nfft),
                       complexflag=2)
        assert dottest(dFFTop, nfft, par['nt'],
                       chunks=(par['nt'], nfft),
                       complexflag=3)
    dy = dFFTop * x
    y = FFTop * x.compute()
    assert_array_almost_equal(dy, y, decimal=5)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5)])
def test_FFT_2dsignal(par):
    """Dot-test and inversion for fft operator for 2d signal
    (fft on single dimension)
    """
    dt, f0 = 0.005, 10
    nt, nx = par['nt'], par['nx']
    t = np.arange(nt) * dt
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
    d = da.from_array(d)

    # 1st dimension
    nfft = par['nt'] if par['nfft'] is None else par['nfft']
    dFFTop = dFFT(dims=(nt, nx), dir=0, nfft=nfft,
                  sampling=dt, real=par['real'],
                  chunks=((nt, nx), (nfft, nx)))
    FFTop = FFT(dims=(nt, nx), dir=0, nfft=nfft, sampling=dt)

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        dFFTop = dFFTop.H * dFFTop
        FFTop = FFTop.H * FFTop
        assert dottest(dFFTop, nt * nx, nt * nx,
                       chunks=(nt * nx, nt * nx),
                       complexflag=0)
    else:
        assert dottest(dFFTop, nfft * nx, nt * nx,
                       chunks=(nt * nx, nfft * nx),
                       complexflag=2)
        assert dottest(dFFTop, nfft * nx, nt * nx,
                       chunks=(nt * nx, nfft * nx),
                       complexflag=3)
    dy = dFFTop * d.ravel()
    y = FFTop * d.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=5)

    # 2nd dimension
    nfft = par['nx'] if par['nfft'] is None else par['nfft']
    dFFTop = dFFT(dims=(nt, nx), dir=1, nfft=nfft, sampling=dt,
                  real=par['real'], chunks=((nt, nx), (nt, nfft)))
    FFTop = FFT(dims=(nt, nx), dir=1, nfft=nfft, sampling=dt)

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        dFFTop = dFFTop.H * dFFTop
        FFTop = FFTop.H * FFTop
        assert dottest(dFFTop, nt * nx, nt * nx,
                       chunks=(nt * nx, nt * nx),
                       complexflag=0)
    else:
        assert dottest(dFFTop, nt * nfft, nt * nx,
                       chunks=(nt * nx, nt * nfft),
                       complexflag=2)
        assert dottest(dFFTop, nt * nfft, nt * nx,
                       chunks=(nt * nx, nt * nfft),
                       complexflag=3)
    dy = dFFTop * d.ravel()
    y = FFTop * d.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=5)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5)])
def test_FFT_3dsignal(par):
    """Dot-test and inversion for fft operator for 3d signal
    (fft on single dimension)
    """
    dt, f0 = 0.005, 10
    nt, nx, ny = par['nt'], par['nx'], par['ny']
    t = np.arange(nt) * dt
    d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
    d = np.tile(d[:, :, np.newaxis], [1, 1, ny])
    d = da.from_array(d)

    # 1st dimension
    nfft = par['nt'] if par['nfft'] is None else par['nfft']
    dFFTop = dFFT(dims=(nt, nx, ny), dir=0, nfft=nfft, sampling=dt,
                  real=par['real'], chunks=((nt, nx, ny), (nfft, nx, ny)))
    FFTop = FFT(dims=(nt, nx, ny), dir=0, nfft=nfft, sampling=dt,
                real=par['real'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        dFFTop = dFFTop.H * dFFTop
        FFTop = FFTop.H * FFTop
        assert dottest(dFFTop, nt * nx * ny, nt * nx * ny,
                       chunks=(nt * nx * ny, nt * nx * ny),
                       complexflag=0)
    else:
        assert dottest(dFFTop, nfft * nx * ny, nt * nx * ny,
                       chunks=(nt * nx * ny, nfft * nx * ny),
                       complexflag=2)
        assert dottest(dFFTop, nfft * nx * ny, nt * nx * ny,
                       chunks=(nt * nx * ny, nfft * nx * ny),
                       complexflag=3)
    dy = dFFTop * d.ravel()
    y = FFTop * d.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=5)

    # 2nd dimension
    nfft = par['nx'] if par['nfft'] is None else par['nfft']
    dFFTop = dFFT(dims=(nt, nx, ny), dir=1, nfft=nfft, sampling=dt,
                  real=par['real'], chunks=((nt, nx, ny), (nt, nfft, ny)))
    FFTop = FFT(dims=(nt, nx, ny), dir=1, nfft=nfft, sampling=dt,
                real=par['real'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        dFFTop = dFFTop.H * dFFTop
        FFTop = FFTop.H * FFTop
        assert dottest(dFFTop, nt * nx * ny, nt * nx * ny,
                       chunks=(nt * nx * ny, nt * nx * ny),
                       complexflag=0)
    else:
        assert dottest(dFFTop, nt * nfft * ny, nt * nx * ny,
                       chunks=(nt * nx * ny, nt * nfft * ny),
                       complexflag=2)
        assert dottest(dFFTop, nt * nfft * ny, nt * nx * ny,
                       chunks=(nt * nx * ny, nt * nfft * ny),
                       complexflag=3)
    dy = dFFTop * d.ravel()
    y = FFTop * d.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=5)

    # 3rd dimension
    nfft = par['ny'] if par['nfft'] is None else par['nfft']
    dFFTop = dFFT(dims=(nt, nx, ny), dir=2, nfft=nfft, sampling=dt,
                  real=par['real'], chunks=((nt, nx, ny), (nt, ny, nfft)))
    FFTop = FFT(dims=(nt, nx, ny), dir=2, nfft=nfft, sampling=dt,
                real=par['real'])

    # FFT with real=True cannot pass dot-test neither be inverted correctly,
    # see FFT documentation for a detailed explanation. We thus test FFT.H*FFT
    if par['real']:
        dFFTop = dFFTop.H * dFFTop
        FFTop = FFTop.H * FFTop
        assert dottest(dFFTop, nt * nx * ny, nt * nx * ny,
                       chunks=(nt * nx * ny, nt * nx * ny),
                       complexflag=0)
    else:
        assert dottest(dFFTop, nt * nx * nfft, nt * nx * ny,
                       chunks=(nt * nx * ny, nt * nx * nfft),
                       complexflag=2)
        assert dottest(dFFTop, nt * nx * nfft, nt * nx * ny,
                       chunks=(nt * nx * ny, nt * nx * nfft),
                       complexflag=3)
    dy = dFFTop * d.ravel()
    y = FFTop * d.ravel().compute()
    assert_array_almost_equal(dy, y, decimal=5)
