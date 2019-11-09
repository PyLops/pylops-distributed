import logging
import numpy as np
import dask.array as da

from scipy.sparse.linalg import lsqr
from pylops.waveeqprocessing.lsm import _traveltime_table \
    as _traveltime_table_single_core

from pylops_distributed.utils import dottest as Dottest
from pylops_distributed import Spread
from pylops_distributed.signalprocessing import Convolve1D

try:
    import skfmm
except ModuleNotFoundError:
    skfmm = None


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _identify_geometry(z, x, srcs, recs, y=None):
    """Identify geometry and acquisition size and sampling
    """
    ns, nr = srcs.shape[1], recs.shape[1]
    nz, nx = len(z), len(x)
    dz = np.abs(z[1] - z[0])
    dx = np.abs(x[1] - x[0])
    if y is None:
        ndims = 2
        shiftdim = 0
        ny = 1
        dy = None
        dims = np.array([nx, nz])
        dsamp = np.array([dx, dz])
        origin = np.array([x[0], z[0]])
    else:
        ndims = 3
        shiftdim = 1
        ny = len(y)
        dy = np.abs(y[1] - y[0])
        dims = np.array([ny, nx, nz])
        dsamp = np.array([dy, dx, dz])
        origin = np.array([y[0], x[0], z[0]])
    return ndims, shiftdim, dims, ny, nx, nz, ns, nr, dy, dx, dz, \
           dsamp, origin

def _traveltime_oneway(trav, sources, vel, dsamp):
    """Auxiliary routine to compute traveltime for a subset of sources
    """
    trav = np.zeros_like(trav)
    for isrc, src in enumerate(sources.T):
        phi = np.ones_like(vel)
        if len(dsamp) == 2:
            src = np.round([src[0] / dsamp[0],
                            src[1] / dsamp[1]]).astype(np.int32)
            phi[src[0], src[1]] = -1

        else:
            src = np.round([src[0] / dsamp[0],
                            src[1] / dsamp[1],
                            src[2] / dsamp[2]]).astype(np.int32)
            phi[src[0], src[1], src[2]] = -1
        trav[:, isrc] = (skfmm.travel_time(phi=phi, speed=vel, dx=dsamp)).ravel()
    return trav

def _traveltime_twoway(trav_template, trav_src, trav_recs, npoints):
    """Auxiliary routine to combine source and receiver traveltimes
    """
    trav_srcrec = np.zeros((npoints, trav_template.shape[1]))
    for isrc in range(trav_src.shape[1]):
        trav_srcrec[:, isrc * trav_recs.shape[1]:(isrc + 1) * trav_recs.shape[1]] = \
            trav_src[:, isrc][:, np.newaxis] + trav_recs
    return trav_srcrec

def _traveltime_table(z, x, srcs, recs, vel, y=None,
                      mode='eikonal', nprocesses=None):
    r"""Traveltime table

    Compute traveltimes along the source-subsurface-receivers triplet
    in 2- or 3-dimensional media given a constant or depth- and space variable
    velocity.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    srcs : :obj:`numpy.ndarray`
        Sources in array of size :math:`\lbrack 2/3 \times n_s \rbrack`
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2/3 \times n_r \rbrack`
    vel : :obj:`numpy.ndarray` or :obj:`float`
        Velocity model of size :math:`\lbrack (n_y \times) n_x
        \times n_z \rbrack` (or constant)
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    mode : :obj:`numpy.ndarray`, optional
        Computation mode (``eikonal``, ``analytic`` - only for constant velocity)
    nprocesses : :obj:`str`, optional
        Number of processes to split computations

    Returns
    -------
    trav : :obj:`numpy.ndarray`
        Total traveltime table of size :math:`\lbrack (n_y*) n_x*n_z
        \times n_s*n_r \rbrack`
    trav_srcs : :obj:`numpy.ndarray`
        Source-to-subsurface traveltime table of size
        :math:`\lbrack (n_y*) n_x*n_z \times n_s \rbrack` (or constant)
    trav_recs : :obj:`numpy.ndarray`
        Receiver-to-subsurface traveltime table of size
        :math:`\lbrack (n_y*) n_x*n_z \times n_r \rbrack`

    """
    if mode == 'analytic':
        trav, trav_srcs, trav_recs = \
            _traveltime_table_single_core(z, x, srcs.compute(),
                                          recs.compute(), vel, y=y,
                                          mode=mode)

    elif mode == 'eikonal':
        ndims, shiftdim, _, ny, nx, nz, ns, nr, _, _, _, dsamp, origin = \
            _identify_geometry(z, x, srcs, recs, y=y)

        if skfmm is not None:
            chunkdim_src = ns//(nprocesses-1) if nprocesses > 1 else ns
            chunkdim_rec = nr//(nprocesses-1) if nprocesses > 1 else nr

            srcs = da.from_array(srcs, chunks=(2, chunkdim_src), name='srcs')
            recs = da.from_array(recs, chunks=(2, chunkdim_rec), name='recs')

            trav_srcs = da.zeros((ny * nx * nz, ns),
                                 chunks=(ny * nx * nz, chunkdim_src),
                                 name='trav-src')
            trav_srcs = da.map_blocks(_traveltime_oneway,
                                      trav_srcs, srcs, vel, dsamp,
                                      dtype='float',
                                      name='traveltime-src')

            trav_recs = da.zeros((ny * nx * nz, nr),
                                 chunks=(ny * nx * nz, chunkdim_rec),
                                 name='trav-rec')
            trav_recs = da.map_blocks(_traveltime_oneway,
                                      trav_recs, recs, vel, dsamp,
                                      dtype='float',
                                      name='traveltime-rec')
            trav_recs = trav_recs.compute()

            trav = da.empty((ny * nx * nz, ns * nr),
                            chunks=(ny * nx * nz, nr * chunkdim_src),
                            name='trav')
            trav = da.map_blocks(_traveltime_twoway, trav,
                                 trav_srcs, trav_recs,
                                 ny * nx * nz, dtype='float',
                                 name='traveltime-sum')
        else:
            raise NotImplementedError('cannot compute traveltime with '
                                      'method=eikonal as skfmm is not '
                                      'installed... choose analytical'
                                      'if using constant velocity model, '
                                      'or install scikit-fmm library')
    else:
        raise NotImplementedError('method must be analytic or eikonal')

    return trav, trav_srcs, trav_recs


def Demigration(z, x, t, srcs, recs, vel, wav, wavcenter,
                y=None, mode='eikonal', trav=None, nprocesses=None,
                client=None):
    r"""Demigration operator.

    Seismic demigration/migration operator.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    t : :obj:`numpy.ndarray`
        Time axis for data
    srcs : :obj:`numpy.ndarray`
        Sources in array of size :math:`\lbrack 2/3 \times n_s \rbrack`
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2/3 \times n_r \rbrack`
    vel : :obj:`numpy.ndarray` or :obj:`float`
        Velocity model of size :math:`\lbrack (n_y \times) n_x
        \times n_z \rbrack` (or constant)
    wav : :obj:`numpy.ndarray`
        Wavelet
    wavcenter : :obj:`int`
        Index of wavelet center
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    mode : :obj:`str`, optional
        Computation mode (``analytic``, ``eikonal`` or ``byot``, see Notes for
        more details)
    trav : :obj:`numpy.ndarray` or :obj:`dask.array.core.Array`, optional
        Traveltime table of size
        :math:`\lbrack n_r*n_s \times (n_y*) n_x*n_z \rbrack`
        To be provided only when ``mode='byot'``
    nprocesses : :obj:`str`, optional
        Number of processes to split computations
    client : :obj:`dask.distributed.client.Client`, optional
        Dask client. If provided, the traveltime computation will be persisted.

    Returns
    -------
    demop : :obj:`pylops.LinearOperator`
        Demigration/Migration operator

    Raises
    ------
    NotImplementedError
        If ``mode`` is neither ``analytic``, ``eikonal``, or ``byot``

    Notes
    -----
    The demigration operator synthetizes seismic data given from a propagation
    velocity model :math:`v` and a reflectivity model :math:`m`. In forward
    mode:

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        w(t) * \int_V G(\mathbf{x}, \mathbf{x_s}, t)
        G(\mathbf{x_r}, \mathbf{x}, t) m(\mathbf{x}) d\mathbf{x}

    where :math:`m(\mathbf{x})` is the model and it represents the reflectivity
    at every location in the subsurface, :math:`G(\mathbf{x}, \mathbf{x_s}, t)`
    and :math:`G(\mathbf{x_r}, \mathbf{x}, t)` are the Green's functions
    from source-to-subsurface-to-receiver and finally  :math:`w(t)` is the
    wavelet. Depending on the choice of ``mode`` the Green's function will be
    computed and applied differently:

    * ``mode=analytic`` or ``mode=eikonal``: traveltime curves between
      source to receiver pairs are computed for every subsurface point and
      Green's functions are implemented from traveltime look-up tables, placing
      the reflectivity values at corresponding source-to-receiver time in the
      data.
    * ``byot``: bring your own table. Traveltime table provided
      directly by user using ``trav`` input parameter. Green's functions are
      then implemented in the same way as previous options.

    The adjoint of the demigration operator is a *migration* operator which
    projects data in the model domain creating an image of the subsurface
    reflectivity.

    """
    ndim, _, dims, ny, nx, nz, ns, nr, _, _, _, _, _ = \
        _identify_geometry(z, x, srcs, recs, y=y)
    dt = t[1] - t[0]
    nt = len(t)

    if mode in ['analytic', 'eikonal', 'byot']:
        # traveltime table
        if mode in ['analytic', 'eikonal']:
            # compute traveltime table
            trav, trav_src, trac_rec = _traveltime_table(z, x, srcs, recs, vel, y=y,
                                                         mode=mode, nprocesses=nprocesses)
        trav = trav.reshape(ny * nx, nz, ns * nr).transpose(2, 0, 1)
        itrav = (trav / dt).astype('int32')
        travd = (trav / dt - itrav)

        # persist traveltime arrays to avoid repeating computations
        if client is not None:
            itrav, travd = client.persist([itrav, travd])

        # define dimensions
        if ndim == 2:
            dims = tuple(dims)
        else:
            dims = (dims[0]*dims[1], dims[2])

        # create operators (todask is temporary until inversion works even
        # without forcing compute)
        sop = Spread(dims=dims, dimsd=(ns * nr, nt),
                     table=itrav, dtable=travd, todask=(True, False))

        cop = Convolve1D(ns * nr * nt, h=wav, offset=wavcenter,
                         dims=(ns * nr, nt), dir=1, todask=(False, True))
        demop = cop * sop
    else:
        raise NotImplementedError('method must be analytic, eikonal, or byot')
    return demop


class LSM():
    r"""Least-squares Migration (LSM).

    Solve seismic migration as inverse problem given smooth velocity model
    ``vel`` and an acquisition setup identified by sources (``src``) and
    receivers (``recs``)

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    t : :obj:`numpy.ndarray`
        Time axis for data
    srcs : :obj:`numpy.ndarray`
        Sources in array of size :math:`\lbrack 2/3 \times n_s \rbrack`
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2/3 \times n_r \rbrack`
    vel : :obj:`numpy.ndarray` or :obj:`float`
        Velocity model of size :math:`\lbrack (n_y \times) n_x
        \times n_z \rbrack` (or constant)
    wav : :obj:`numpy.ndarray`
        Wavelet
    wavcenter : :obj:`int`
        Index of wavelet center
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    mode : :obj:`numpy.ndarray`, optional
        Computation mode (``eikonal``, ``analytic`` - only for
        constant velocity)
    trav : :obj:`numpy.ndarray`, optional
        Traveltime table of size
        :math:`\lbrack (n_y*) n_x*n_z \times n_r*n_s \rbrack`
        (to be provided if ``mode='byot'``)
    dottest : :obj:`bool`, optional
        Apply dot-test
    enginetrav : :obj:`str`, optional
        Engine used for traveltime computation when ``mode='eikonal'``
        (``numpy`` and ``dask`` supported)
    engine : :obj:`str`, optional
        Engine used for :class:`pylops.basicoperators.Spread` computation in
        forward and adjoint modelling operations (``numpy``, ``numba``,
        or ``dask``)
    nprocesses : :obj:`str`, optional
        Number of processes to split computations on (if ``engine=dask``)

    Attributes
    ----------
    Demop : :class:`pylops.LinearOperator`
        Demigration operator

    See Also
    --------
    pylops.waveeqprocessing.Demigration : Demigration operator

    Notes
    -----
    Inverting a demigration operator is generally referred in the literature
    as least-squares migration (LSM) as historically a least-squares cost
    function has been used for this purpose. In practice any other cost
    function could be used, for examples if
    ``solver='pylops.optimization.sparsity.FISTA'`` a sparse representation of
    reflectivity is produced as result of the inversion.

    Finally, it is worth noting that in the first iteration of an iterative
    scheme aimed at inverting the demigration operator, a projection of the
    recorded data in the model domain is performed and an approximate
    (band-limited)  image of the subsurface is created. This process is
    referred to in the literature as *migration*.

    """
    def __init__(self, z, x, t, srcs, recs, vel, wav, wavcenter, y=None,
                 mode='eikonal', trav=None, dottest=False,
                 nprocesses=None, client=None):
        self.y, self.x, self.z = y, x, z
        self.Demop = Demigration(z, x, t, srcs, recs, vel, wav, wavcenter,
                                 y=y, mode=mode, trav=trav,
                                 nprocesses=nprocesses, client=client)
        if dottest:
            Dottest(self.Demop, self.Demop.shape[0], self.Demop.shape[1],
                    chunks=(self.Demop.shape[0]//nprocesses,
                            self.Demop.shape[1]), raiseerror=True, verb=True)

    def solve(self, d, solver=lsqr, **kwargs_solver):
        r"""Solve least-squares migration equations with chosen ``solver``

        Parameters
        ----------
        d : :obj:`numpy.ndarray`
            Input data of size :math:`\lbrack n_s \times n_r
            \times n_t \rbrack`
        solver : :obj:`func`, optional
            Solver to be used for inversion
        **kwargs_solver
            Arbitrary keyword arguments for chosen ``solver``

        Returns
        -------
        minv : :obj:`np.ndarray`
            Inverted reflectivity model of size :math:`\lbrack (n_y \times)
            n_x \times n_z \rbrack`

        """
        minv = solver(self.Demop, d.ravel(), **kwargs_solver)
        if isinstance(minv, (list, tuple)):
            minv = minv[0]

        if self.y is None:
            minv = minv.reshape(len(self.x), len(self.z))
        else:
            minv = minv.reshape(len(self.y), len(self.x), len(self.z))

        return minv
