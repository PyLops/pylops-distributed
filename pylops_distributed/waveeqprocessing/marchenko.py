import logging
import warnings
import numpy as np
import dask.array as da

from scipy.signal import filtfilt
from pylops.waveeqprocessing.marchenko import directwave
from pylops_distributed.utils import dottest as Dottest
from pylops_distributed import Diagonal, Identity, Block, BlockDiag, Roll
from pylops_distributed.waveeqprocessing.mdd import MDC
from pylops_distributed.optimization.cg import cgls

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class Marchenko():
    r"""Marchenko redatuming

    Solve multi-dimensional Marchenko redatuming problem using
    :py:func:`scipy.sparse.linalg.lsqr` iterative solver.

    Parameters
    ----------
    R : :obj:`dask.array`
        Multi-dimensional reflection response in frequency
        domain of size :math:`[n_{fmax} \times n_s \times n_r]`. Note that the
        reflection response should have already been multiplied by 2.
    nt : :obj:`float`, optional
        Number of samples in time
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    wav : :obj:`numpy.ndarray`, optional
        Wavelet to apply to direct arrival when created using ``trav``
    toff : :obj:`float`, optional
        Time-offset to apply to traveltime
    nsmooth : :obj:`int`, optional
        Number of samples of smoothing operator to apply to window
    saveRt : :obj:`bool`, optional
        Save ``R`` and ``R^H`` to speed up the computation of the adjoint of
        :class:`pylops_distributed.signalprocessing.Fredholm1` (``True``) or
        create ``R^H`` on-the-fly (``False``) Note that ``saveRt=True`` will be
        faster but double the amount of required memory
    prescaled : :obj:`bool`, optional
        Apply scaling to ``R`` (``False``) or not (``False``)
        when performing spatial and temporal summations within the
        :class:`pylops.waveeqprocessing.MDC` operator. In case
        ``prescaled=True``, the ``R`` is assumed to have been pre-scaled by
        the user.
    dtype : :obj:`bool`, optional
        Type of elements in input array.

    Attributes
    ----------
    ns : :obj:`int`
        Number of samples along source axis
    nr : :obj:`int`
        Number of samples along receiver axis
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (True) or not (False)

    Raises
    ------
    TypeError
        If ``t`` is not :obj:`numpy.ndarray`.

    See Also
    --------
    MDC : Multi-dimensional convolution

    Notes
    -----
    Refer to :class:`pylops.waveeqprocessing.Marchenko` for implementation
    details.

    """
    def __init__(self, R, nt, dt=0.004, dr=1., wav=None, toff=0.0,
                 nsmooth=10, saveRt=True, prescaled=False, dtype='float32'):
        # Save inputs into class
        self.nt = nt
        self.dt = dt
        self.dr = dr
        self.wav = wav
        self.toff = toff
        self.nsmooth = nsmooth
        self.saveRt = saveRt
        self.prescaled = prescaled
        self.dtype = dtype
        self.explicit = False

        # Infer dimensions of R
        self.nfmax, self.ns, self.nr = R.shape
        self.nt2 = int(2*self.nt-1)
        self.t = np.arange(self.nt)*self.dt

        # Fix nfmax to be at maximum equal to half of the size of fft samples
        if self.nfmax is None or self.nfmax > np.ceil((self.nt2 + 1) / 2):
            self.nfmax = int(np.ceil((self.nt2+1)/2))
            logging.warning('nfmax set equal to (nt+1)/2=%d', self.nfmax)

        # Save R
        self.Rtwosided_fft = R


    def apply_onepoint(self, trav, dist=None, G0=None, nfft=None,
                       rtm=False, greens=False, dottest=False,
                       **kwargs_cgls):
        r"""Marchenko redatuming for one point

        Solve the Marchenko redatuming inverse problem for a single point
        given its direct arrival traveltime curve (``trav``)
        and waveform (``G0``).

        Parameters
        ----------
        trav : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface point to
            surface receivers of size :math:`[n_r \times 1]`
        dist: :obj:`numpy.ndarray`, optional
            Distance between subsurface point to
            surface receivers of size :math:`\lbrack nr \times 1 \rbrack`
            (if provided the analytical direct arrival will be computed using
            a 3d formulation)
        G0 : :obj:`numpy.ndarray`, optional
            Direct arrival in time domain of size :math:`[n_r \times n_t]`
            (if None, create arrival using ``trav``)
        nfft : :obj:`int`, optional
            Number of samples in fft when creating the analytical direct wave
        rtm : :obj:`bool`, optional
            Compute and return rtm redatuming
        greens : :obj:`bool`, optional
            Compute and return Green's functions
        dottest : :obj:`bool`, optional
            Apply dot-test
        **kwargs_cgls
            Arbitrary keyword arguments for
            :py:func:`pylops_distributed.optimization.cg.cgls` solver

        Returns
        -------
        f1_inv_minus : :obj:`dask.array`
            Inverted upgoing focusing function of size :math:`[n_r \times n_t]`
        f1_inv_plus : :obj:`dask.array`
            Inverted downgoing focusing function
            of size :math:`[n_r \times n_t]`
        p0_minus : :obj:`dask.array`
            Single-scattering standard redatuming upgoing Green's function of
            size :math:`[n_r \times n_t]`
        g_inv_minus : :obj:`dask.array`
            Inverted upgoing Green's function of size :math:`[n_r \times n_t]`
        g_inv_plus : :obj:`dask.array`
            Inverted downgoing Green's function
            of size :math:`[n_r \times n_t]`

        """
        # Create window
        trav_off = trav - self.toff
        trav_off = np.round(trav_off / self.dt).astype(np.int)
        w = np.zeros((self.nr, self.nt), dtype=self.dtype)
        for ir in range(self.nr):
            w[ir, :trav_off[ir]] = 1
        w = np.hstack((np.fliplr(w), w[:, 1:]))
        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth) / self.nsmooth
            w = filtfilt(smooth, 1, w)
        w = w.astype(self.dtype)

        # Create operators
        Rop = MDC(self.Rtwosided_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                  twosided=True, conj=False, saveGt=self.saveRt,
                  prescaled=self.prescaled)
        R1op = MDC(self.Rtwosided_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                   twosided=True, conj=True, saveGt=self.saveRt,
                   prescaled=self.prescaled)
        Rollop = Roll(self.nt2 * self.ns,
                      dims=(self.nt2, self.ns),
                      dir=0, shift=-1, dtype=self.dtype)
        Wop = Diagonal(da.from_array(w.T.flatten()), dtype=self.dtype)
        Iop = Identity(self.nr * self.nt2, dtype=self.dtype)
        Mop = Block([[Iop, -1 * Wop * Rop],
                     [-1 * Wop * Rollop * R1op, Iop]]) * BlockDiag([Wop, Wop])
        Gop = Block([[Iop, -1 * Rop],
                     [-1 * Rollop * R1op, Iop]])

        if dottest:
            Dottest(Gop, 2 * self.ns * self.nt2,
                    2 * self.nr * self.nt2,
                    chunks=(2 * self.ns * self.nt2,
                            2 * self.nr * self.nt2),
                    raiseerror=True, verb=True)
        if dottest:
            Dottest(Mop, 2 * self.ns * self.nt2,
                    2 * self.nr * self.nt2,
                    chunks=(2 * self.ns * self.nt2,
                            2 * self.nr * self.nt2),
                    raiseerror=True, verb=True)

        # Create input focusing function
        if G0 is None:
            if self.wav is not None and nfft is not None:
                G0 = (directwave(self.wav, trav, self.nt,
                                 self.dt, nfft=nfft,
                                 derivative=True, dist=dist,
                                 kind='2d' if dist is None else '3d')).T
            else:
                logging.error('wav and/or nfft are not provided. '
                              'Provide either G0 or wav and nfft...')
                raise ValueError('wav and/or nfft are not provided. '
                                 'Provide either G0 or wav and nfft...')
            G0 = G0.astype(self.dtype)

        fd_plus = np.concatenate((np.fliplr(G0).T,
                                  np.zeros((self.nt - 1, self.nr),
                                           dtype=self.dtype)))
        fd_plus = da.from_array(fd_plus)

        # Run standard redatuming as benchmark
        if rtm:
            p0_minus = Rop * fd_plus.flatten()
            p0_minus = p0_minus.reshape(self.nt2, self.ns).T

        # Create data and inverse focusing functions
        d = Wop * Rop * fd_plus.flatten()
        d = da.concatenate((d.reshape(self.nt2, self.ns),
                            da.zeros((self.nt2, self.ns),
                            dtype = self.dtype)))

        # Invert for focusing functions
        f1_inv = cgls(Mop, d.flatten(), **kwargs_cgls)[0]
        f1_inv = f1_inv.reshape(2 * self.nt2, self.nr)
        f1_inv_tot = f1_inv + da.concatenate((da.zeros((self.nt2, self.nr),
                                                       dtype=self.dtype), fd_plus))
        # Create Green's functions
        if greens:
            g_inv = Gop * f1_inv_tot.flatten()
            g_inv = g_inv.reshape(2 * self.nt2, self.ns)

        # Compute
        if rtm and greens:
            d, p0_minus, f1_inv_tot, g_inv = \
                da.compute(d, p0_minus, f1_inv_tot, g_inv)
        elif rtm:
            d, p0_minus, f1_inv_tot = \
                da.compute(d, p0_minus, f1_inv_tot)
        elif greens:
            d, f1_inv_tot, g_inv = \
                da.compute(d, f1_inv_tot, g_inv)
        else:
            d, f1_inv_tot = \
                da.compute(d, f1_inv_tot)

        # Separate focusing and Green's functions
        f1_inv_minus = f1_inv_tot[:self.nt2].T
        f1_inv_plus = f1_inv_tot[self.nt2:].T
        if greens:
            g_inv = np.real(g_inv) # cast to real as Gop is a complex operator
            g_inv_minus, g_inv_plus = -g_inv[:self.nt2].T, \
                                      np.fliplr(g_inv[self.nt2:].T)

        if rtm and greens:
            return f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus
        elif rtm:
            return f1_inv_minus, f1_inv_plus, p0_minus
        elif greens:
            return f1_inv_minus, f1_inv_plus, g_inv_minus, g_inv_plus
        else:
            return f1_inv_minus, f1_inv_plus

    def apply_multiplepoints(self, trav, dist=None, G0=None, nfft=None,
                             rtm=False, greens=False,
                             dottest=False, **kwargs_cgls):
        r"""Marchenko redatuming for multiple points

        Solve the Marchenko redatuming inverse problem for multiple
        points given their direct arrival traveltime curves (``trav``)
        and waveforms (``G0``).

        Parameters
        ----------
        trav : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface points to
            surface receivers of size :math:`[n_r \times n_{vs}]`
        dist: :obj:`numpy.ndarray`, optional
            Distance between subsurface point to
            surface receivers of size :math:`[n_r \times n_{vs}]`
            (if provided the analytical direct arrival will be computed using
            a 3d formulation)
        G0 : :obj:`numpy.ndarray`, optional
            Direct arrival in time domain of size
            :math:`[n_r \times n_{vs} \times n_t]` (if None, create arrival
            using ``trav``)
        nfft : :obj:`int`, optional
            Number of samples in fft when creating the analytical direct wave
        rtm : :obj:`bool`, optional
            Compute and return rtm redatuming
        greens : :obj:`bool`, optional
            Compute and return Green's functions
        dottest : :obj:`bool`, optional
            Apply dot-test
        **kwargs_cgls
            Arbitrary keyword arguments for
            :py:func:`pylops_distributed.optimization.cg.cgls` solver

        Returns
        -------
        f1_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing focusing function of size
            :math:`[n_r \times n_{vs} \times n_t]`
        f1_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing focusing functionof size
            :math:`[n_r \times n_{vs} \times n_t]`
        p0_minus : :obj:`numpy.ndarray`
            Single-scattering standard redatuming upgoing Green's function
            of size :math:`[n_r \times n_{vs} \times n_t]`
        g_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing Green's function of size
            :math:`[n_r \times n_{vs} \times n_t]`
        g_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing Green's function of size
            :math:`[n_r \times n_{vs} \times n_t]`

        """
        nvs = trav.shape[1]

        # Create window
        trav_off = trav - self.toff
        trav_off = np.round(trav_off / self.dt).astype(np.int)

        w = np.zeros((self.nr, nvs, self.nt), dtype=self.dtype)
        for ir in range(self.nr):
            for ivs in range(nvs):
                w[ir, ivs, :trav_off[ir, ivs]] = 1
        w = np.concatenate((np.flip(w, axis=-1), w[:, :, 1:]), axis=-1)
        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth) / self.nsmooth
            w = filtfilt(smooth, 1, w)
        w = w.astype(self.dtype)

        # Create operators
        Rop = MDC(self.Rtwosided_fft, self.nt2, nv=nvs, dt=self.dt,
                  dr=self.dr, twosided=True, conj=False, saveGt=self.saveRt,
                  prescaled=self.prescaled)
        R1op = MDC(self.Rtwosided_fft, self.nt2, nv=nvs, dt=self.dt,
                   dr=self.dr, twosided=True, conj=True, saveGt=self.saveRt,
                   prescaled=self.prescaled)
        Rollop = Roll(self.ns * nvs * self.nt2,
                      dims=(self.nt2, self.ns, nvs),
                      dir=0, shift=-1, dtype=self.dtype)
        Wop = Diagonal(da.from_array(w.transpose(2, 0, 1).flatten()),
                       dtype=self.dtype)
        Iop = Identity(self.nr * nvs * self.nt2, dtype=self.dtype)
        Mop = Block([[Iop, -1 * Wop * Rop],
                     [-1 * Wop * Rollop * R1op, Iop]]) * BlockDiag([Wop, Wop])
        Gop = Block([[Iop, -1 * Rop],
                     [-1 * Rollop * R1op, Iop]])

        if dottest:
            Dottest(Gop, 2 * self.nr * nvs * self.nt2,
                    2 * self.nr * nvs * self.nt2,
                    chunks=(2 * self.ns * nvs * self.nt2,
                            2 * self.nr * nvs * self.nt2),
                    raiseerror=True, verb=True)
        if dottest:
            Dottest(Mop, 2 * self.ns * nvs * self.nt2,
                    2 * self.nr * nvs * self.nt2,
                    chunks=(2 * self.ns * nvs * self.nt2,
                            2 * self.nr * nvs * self.nt2),
                    raiseerror=True, verb=True)

        # Create input focusing function
        if G0 is None:
            if self.wav is not None and nfft is not None:
                G0 = np.zeros((self.nr, nvs, self.nt), dtype=self.dtype)
                for ivs in range(nvs):
                    G0[:, ivs] = (directwave(self.wav, trav[:, ivs],
                                             self.nt, self.dt, nfft=nfft,
                                             derivative=True,  dist=dist,
                                             kind='2d' if dist is None else '3d')).T
            else:
                logging.error('wav and/or nfft are not provided. '
                              'Provide either G0 or wav and nfft...')
                raise ValueError('wav and/or nfft are not provided. '
                                 'Provide either G0 or wav and nfft...')
            G0 = G0.astype(self.dtype)

        fd_plus = np.concatenate((np.flip(G0, axis=-1).transpose(2, 0, 1),
                                  np.zeros((self.nt - 1, self.nr, nvs),
                                           dtype=self.dtype)))
        fd_plus = da.from_array(fd_plus).rechunk(fd_plus.shape)

        # Run standard redatuming as benchmark
        if rtm:
            p0_minus = Rop * fd_plus.flatten()
            p0_minus = p0_minus.reshape(self.nt2, self.ns,
                                        nvs).transpose(1, 2, 0)

        # Create data and inverse focusing functions
        d = Wop * Rop * fd_plus.flatten()
        d = da.concatenate((d.reshape(self.nt2, self.ns, nvs),
                            da.zeros((self.nt2, self.ns, nvs),
                                     dtype=self.dtype)))

        # Invert for focusing functions
        f1_inv = cgls(Mop, d.flatten(), **kwargs_cgls)[0]
        f1_inv = f1_inv.reshape(2 * self.nt2, self.nr, nvs)
        f1_inv_tot = \
            f1_inv + da.concatenate((da.zeros((self.nt2, self.nr, nvs),
                                              dtype=self.dtype), fd_plus))
        if greens:
            # Create Green's functions
            g_inv = Gop * f1_inv_tot.flatten()
            g_inv = g_inv.reshape(2 * self.nt2, self.ns, nvs)

        # Compute
        if rtm and greens:
            d, p0_minus, f1_inv_tot, g_inv = \
                da.compute(d, p0_minus, f1_inv_tot, g_inv)
        elif rtm:
            d, p0_minus, f1_inv_tot = \
                da.compute(d, p0_minus, f1_inv_tot)
        elif greens:
            d, f1_inv_tot, g_inv = \
                da.compute(d, f1_inv_tot, g_inv)
        else:
            d, f1_inv_tot = \
                da.compute(d, f1_inv_tot)

        # Separate focusing and Green's functions
        f1_inv_minus = f1_inv_tot[:self.nt2].transpose(1, 2, 0)
        f1_inv_plus = f1_inv_tot[self.nt2:].transpose(1, 2, 0)
        if greens:
            g_inv = np.real(g_inv)  # cast to real as Gop is a complex operator
            g_inv_minus = -g_inv[:self.nt2].transpose(1, 2, 0)
            g_inv_plus = np.flip(g_inv[self.nt2:], axis=0).transpose(1, 2, 0)

        if rtm and greens:
            return f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus
        elif rtm:
            return f1_inv_minus, f1_inv_plus, p0_minus
        elif greens:
            return f1_inv_minus, f1_inv_plus, g_inv_minus, g_inv_plus
        else:
            return f1_inv_minus, f1_inv_plus
