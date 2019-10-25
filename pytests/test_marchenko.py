import pytest
import numpy as np
import dask.array as da

from numpy.testing import assert_array_almost_equal
from scipy.signal import convolve
from pylops.waveeqprocessing.marchenko import Marchenko
from pylops_distributed.waveeqprocessing.marchenko import Marchenko as dMarchenko

# Test data
inputfile = 'testdata/marchenko/input.npz'
inputzarr = 'testdata/marchenko/input.zarr'

# Parameters
vel = 2400.0  # velocity
toff = 0.045  # direct arrival time shift
nsmooth = 10  # time window smoothing
nfmax = 1000  # max frequency for MDC (#samples)

# Input data
inputdata = np.load(inputfile)

# Receivers
r = inputdata['r']
nr = r.shape[1]
dr = r[0, 1] - r[0, 0]

# Sources
s = inputdata['s']
ns = s.shape[1]

# Virtual points
vs = inputdata['vs']

# Multiple virtual points
vs_multi = [np.arange(-1, 2)*100 + vs[0],
            np.ones(3)*vs[1]]

# Density model
rho = inputdata['rho']
z, x = inputdata['z'], inputdata['x']

# Reflection data and subsurface fields
R = inputdata['R']
R = np.swapaxes(R, 0, 1)

gsub = inputdata['Gsub']
g0sub = inputdata['G0sub']
wav = inputdata['wav']
wav_c = np.argmax(wav)

t = inputdata['t']
ot, dt, nt = t[0], t[1] - t[0], len(t)

gsub = np.apply_along_axis(convolve, 0, gsub, wav, mode='full')
gsub = gsub[wav_c:][:nt]
g0sub = np.apply_along_axis(convolve, 0, g0sub, wav, mode='full')
g0sub = g0sub[wav_c:][:nt]

# Direct arrival window
trav = np.sqrt((vs[0] - r[0]) ** 2 + (vs[1] - r[1]) ** 2) / vel
trav_multi = np.sqrt((vs_multi[0]-r[0][:, np.newaxis])**2 +
                     (vs_multi[1]-r[1][:, np.newaxis])**2)/vel


# Create Rs in frequency domain
Rtwosided = np.concatenate((np.zeros((nr, ns, nt-1)), R), axis=-1)
R1twosided = np.concatenate((np.flip(R, axis=-1),
                             np.zeros((nr, ns, nt - 1))), axis=-1)

Rtwosided_fft = np.fft.rfft(Rtwosided, 2*nt-1, axis=-1) / np.sqrt(2*nt-1)
Rtwosided_fft = Rtwosided_fft[..., :nfmax]

# Load distributed Rs in frequency domain
dRtwosided_fft = da.from_zarr(inputzarr)


def test_Marchenko():
    """Dot-test and comparison with pylops for Marchenko.apply_onepoint
    """
    dMarchenkoWM = dMarchenko(dRtwosided_fft, nt=nt, dt=dt, dr=dr,
                              wav=wav, toff=toff, nsmooth=nsmooth)


    MarchenkoWM = Marchenko(Rtwosided_fft, nt=nt, dt=dt, dr=dr,
                            wav=wav, toff=toff, nsmooth=nsmooth)

    _, _, dp0_minus, dg_inv_minus, dg_inv_plus = \
        dMarchenkoWM.apply_onepoint(trav, nfft=2 ** 11, rtm=True, greens=True,
                                    dottest=True, **dict(niter=10,
                                                         compute=False))

    _, _, p0_minus = \
        MarchenkoWM.apply_onepoint(trav, nfft=2 ** 11, rtm=True, greens=False,
                                   dottest=False, **dict(iter_lim=0, show=0))
    assert_array_almost_equal(dp0_minus, p0_minus, decimal=5)

    dginvsub = (dg_inv_minus + dg_inv_plus)[:, nt - 1:].T
    dginvsub_norm = dginvsub / dginvsub.max()
    gsub_norm = gsub / gsub.max()
    assert np.linalg.norm(gsub_norm - dginvsub_norm) / \
           np.linalg.norm(gsub_norm) < 1e-1


def test_Marchenko__multi():
    """Dot-test and comparison with pylops for Marchenko.apply_multiplepoints
    """
    dMarchenkoWM = dMarchenko(dRtwosided_fft, nt=nt, dt=dt, dr=dr,
                              wav=wav, toff=toff, nsmooth=nsmooth)

    MarchenkoWM = Marchenko(Rtwosided_fft, nt=nt, dt=dt, dr=dr,
                            wav=wav, toff=toff, nsmooth=nsmooth)

    _, _, dp0_minus, dg_inv_minus, dg_inv_plus = \
        dMarchenkoWM.apply_multiplepoints(trav_multi, nfft=2 ** 11, rtm=True,
                                          greens=True, dottest=True,
                                          **dict(niter=10, compute=False))

    _, _, p0_minus = \
        MarchenkoWM.apply_multiplepoints(trav_multi, nfft=2 ** 11, rtm=True,
                                         greens=False, dottest=False,
                                         **dict(iter_lim=0, show=0))
    assert_array_almost_equal(dp0_minus, p0_minus, decimal=5)

    dginvsub = (dg_inv_minus + dg_inv_plus)[:, 1, nt - 1:].T
    dginvsub_norm = dginvsub / dginvsub.max()
    gsub_norm = gsub / gsub.max()
    assert np.linalg.norm(gsub_norm - dginvsub_norm) / \
           np.linalg.norm(gsub_norm) < 1e-1
