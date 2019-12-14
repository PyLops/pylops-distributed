import logging
import numpy as np
import dask.array as da

from pylops_distributed import LinearOperator

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class Fredholm1(LinearOperator):
    r"""Fredholm integral of first kind.

    Implement a multi-dimensional Fredholm integral of first kind. Note that if
    the integral is two dimensional, this can be directly implemented using
    :class:`pylops.basicoperators.MatrixMult`. A multi-dimensional
    Fredholm integral can be performed as a :class:`pylops.basicoperators.BlockDiag`
    operator of a series of :class:`pylops.basicoperators.MatrixMult`. However,
    here we take advantage of the structure of the kernel and perform it in a
    more efficient manner.

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel of size
        :math:`[n_{slice} \times n_x \times n_y]`
    nz : :obj:`numpy.ndarray`, optional
        Additional dimension of model
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint
        (``True``) or create ``G^H`` on-the-fly (``False``)
        Note that ``saveGt=True`` will double the amount of required memory
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array`
    chunks : :obj:`tuple`, optional
        Chunk size for model and data. If provided it will rechunk the model
        before applying the forward pass and the data before applying the
        adjoint pass
    todask : :obj:`tuple`, optional
        Apply :func:`dask.array.from_array` to model and data before applying
        forward and adjoint respectively
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Notes
    -----
    Refer to :class:`pylops.signalprocessing.Identity` for implementation
    details.

    """
    def __init__(self, G, nz=1, saveGt=True, compute=(False, False),
                 chunks=(None, None), todask=(None, None), dtype='float64'):
        self.nz = nz
        self.nsl, self.nx, self.ny = G.shape
        self.G = G
        self.shape = (self.nsl * self.nx * self.nz,
                      self.nsl * self.ny * self.nz)
        if saveGt:
            self.GT = (G.transpose((0, 2, 1)).conj()).persist()
        else:
            # choose what to transpose if Gt is not saved
            self.transposeG = True if self.G.size < self.shape[0] else False
        self.dtype = np.dtype(dtype)
        self.compute = compute
        self.chunks = chunks
        self.todask = todask
        self.Op = None
        self.explicit = False

    def _matvec(self, x):
        x = da.squeeze(x.reshape(self.nsl, self.ny, self.nz))
        if self.chunks[0] is not None:
            x = x.rechunk(self.chunks[0])
        if self.nz == 1:
            x = x[..., np.newaxis]
        y = da.matmul(self.G, x)
        return y.ravel()

    def _rmatvec(self, x):
        x = np.squeeze(x.reshape(self.nsl, self.nx, self.nz))
        if self.chunks[0] is not None:
            x = x.rechunk(self.chunks[0])
        if self.nz == 1:
            x = x[..., np.newaxis]
        if hasattr(self, 'GT'):
            y = da.matmul(self.GT, x)
        else:
            if self.transposeG:
                y = da.matmul(self.G.transpose((0, 2, 1)).conj(), x)
            else:
                y = da.matmul(x.transpose(0, 2, 1).conj(),
                              self.G).transpose(0, 2, 1).conj()
        return y.ravel()
