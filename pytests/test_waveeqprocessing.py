import pytest
import numpy as np
import dask.array as da

from numpy.testing import assert_array_almost_equal
from pylops.utils.wavelets import ricker
from pylops.utils.seismicevents import makeaxis, linear2d, linear3d
from pylops.waveeqprocessing.mdd import MDC

from pylops_distributed.waveeqprocessing.mdd import MDC as dMDC
from pylops_distributed.utils import dottest

PAR = {'ox': 0, 'dx': 2, 'nx': 10,
       'oy': 0, 'dy': 2, 'ny': 20,
       'ot': 0, 'dt': 0.004, 'nt': 401,
       'f0': 20}

# nt odd, single-sided, full fft
par1 = PAR.copy()
par1['twosided'] = False
par1['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))

# nt odd, double-sided, full fft
par2 = PAR.copy()
par2['twosided'] = True
par2['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))

# nt odd, single-sided, truncated fft
par3 = PAR.copy()
par3['twosided'] = False
par3['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))-30

# nt odd, double-sided, truncated fft
par4 = PAR.copy()
par4['twosided'] = True
par4['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))-30

# nt even, single-sided, full fft
par5 = PAR.copy()
par5['nt'] -= 1
par5['twosided'] = False
par5['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))

# nt even, double-sided, full fft
par6 = PAR.copy()
par6['nt'] -= 1
par6['twosided'] = True
par6['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))

# nt even, single-sided, truncated fft
par7 = PAR.copy()
par7['nt'] -= 1
par7['twosided'] = False
par7['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))-30

# nt even, double-sided, truncated fft
par8 = PAR.copy()
par8['nt'] -= 1
par8['twosided'] = True
par8['nfmax'] = int(np.ceil((PAR['nt']+1.)/2))-30


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par5), (par6), (par7), (par8)])
def test_MDC_1virtualsource(par):
    """Dot-test and comparison with pylops for MDC operator of 1 virtual source
    """
    if par['twosided']:
        par['nt2'] = 2*par['nt'] - 1
    else:
        par['nt2'] = par['nt']
    v = 1500
    it0_m = 25
    t0_m = it0_m*par['dt']
    theta_m = 0
    amp_m = 1.

    it0_G = np.array([25, 50, 75])
    t0_G = it0_G*par['dt']
    theta_G = (0, 0, 0)
    phi_G = (0, 0, 0)
    amp_G = (1., 0.6, 2.)

    # Create axis
    t, _, x, y = makeaxis(par)

    # Create wavelet
    wav = ricker(t[:41], f0=par['f0'])[0]

    # Generate model
    _, mwav = linear2d(x, t, v, t0_m, theta_m, amp_m, wav)
    # Generate operator
    _, Gwav = linear3d(x, y, t, v, t0_G, theta_G, phi_G, amp_G, wav)

    # Add negative part to data and model
    if par['twosided']:
        mwav = np.concatenate((np.zeros((par['nx'], par['nt'] - 1)), mwav),
                              axis=-1)
        Gwav = np.concatenate((np.zeros((par['ny'], par['nx'], par['nt'] - 1)),
                               Gwav), axis=-1)

    # Define MDC linear operator
    Gwav_fft = np.fft.fft(Gwav, par['nt2'], axis=-1)
    Gwav_fft = Gwav_fft[..., :par['nfmax']]

    dMDCop = dMDC(da.from_array(Gwav_fft.transpose(2, 0, 1)),
                  nt=par['nt2'], nv=1, dt=par['dt'], dr=par['dx'],
                  twosided=par['twosided'])
    MDCop = MDC(Gwav_fft.transpose(2, 0, 1), nt=par['nt2'], nv=1,
                dt=par['dt'], dr=par['dx'], twosided=par['twosided'],
                transpose=False, dtype='float32')
    dottest(dMDCop, par['nt2'] * par['ny'], par['nt2'] * par['nx'],
            chunks=((par['nt2'] * par['ny'], par['nt2'] * par['nx'])))

    mwav = mwav.T
    dy = (dMDCop * da.from_array(mwav.flatten())).compute()
    y = MDCop * mwav.flatten()
    assert_array_almost_equal(dy, y, decimal=5)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4),
                                 (par5), (par6), (par7), (par8)])
def test_MDC_Nvirtualsources(par):
    """Dot-test and comparison with pylops for MDC operator of N virtual source
    """
    if par['twosided']:
        par['nt2'] = 2*par['nt'] - 1
    else:
        par['nt2'] = par['nt']
    v = 1500
    it0_m = 25
    t0_m = it0_m * par['dt']
    theta_m = 0
    phi_m = 0
    amp_m = 1.

    it0_G = np.array([25, 50, 75])
    t0_G = it0_G * par['dt']
    theta_G = (0, 0, 0)
    phi_G = (0, 0, 0)
    amp_G = (1., 0.6, 2.)

    # Create axis
    t, _, x, y = makeaxis(par)

    # Create wavelet
    wav = ricker(t[:41], f0=par['f0'])[0]

    # Generate model
    _, mwav = linear3d(x, x, t, v, t0_m, theta_m, phi_m, amp_m, wav)

    # Generate operator
    _, Gwav = linear3d(x, y, t, v, t0_G, theta_G, phi_G, amp_G, wav)

    # Add negative part to data and model
    if par['twosided']:
        mwav = np.concatenate((np.zeros((par['nx'], par['nx'], par['nt'] - 1)),
                               mwav), axis=-1)
        Gwav = np.concatenate((np.zeros((par['ny'], par['nx'], par['nt'] - 1)),
                               Gwav), axis=-1)

    # Define MDC linear operator
    Gwav_fft = np.fft.fft(Gwav, par['nt2'], axis=-1)
    Gwav_fft = Gwav_fft[..., :par['nfmax']]

    dMDCop = dMDC(da.from_array(Gwav_fft.transpose(2, 0, 1)), nt=par['nt2'],
                  nv=par['nx'], dt=par['dt'], dr=par['dx'],
                  twosided=par['twosided'])
    MDCop = MDC(Gwav_fft.transpose(2, 0, 1), nt=par['nt2'], nv=par['nx'],
                dt=par['dt'], dr=par['dx'], twosided=par['twosided'],
                transpose=False, dtype='float32')

    dottest(dMDCop, par['nt2'] * par['ny'] * par['nx'],
            par['nt2'] * par['nx'] * par['nx'],
            chunks=((par['nt2'] * par['ny'] * par['nx'],
                     par['nt2'] * par['nx'] * par['nx'])))

    mwav = mwav.T
    dy = (dMDCop * da.from_array(mwav.flatten())).compute()
    y = MDCop * mwav.flatten()
    assert_array_almost_equal(dy, y, decimal=5)