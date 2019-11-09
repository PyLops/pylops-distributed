import logging
import numpy as np
import dask.array as da

from pylops_distributed import LinearOperator

try:
    from numba import jit
    from pylops.basicoperators._Spread_numba import _matvec_numba_table, \
        _rmatvec_numba_table, _matvec_numba_onthefly, _rmatvec_numba_onthefly
except ModuleNotFoundError:
    jit = None


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class Spread(LinearOperator):
    r"""Spread operator.

    Spread values from the input model vector arranged as a 2-dimensional
    array of size :math:`[n_{x0} \times n_{t0}]` into the data vector of size
    :math:`[n_x \times n_t]`. Spreading is performed along parametric curves
    provided as look-up table of pre-computed indices (``table``)
    or computed on-the-fly using a function handle (``fh``).

    In adjont mode, values from the data vector are instead stacked
    along the same parametric curves.

    Parameters
    ----------
    dims : :obj:`tuple`
        Dimensions of model vector (vector will be reshaped internally into
        a two-dimensional array of size :math:`[n_{x0} \times n_{t0}]`,
        where the first dimension is the spreading/stacking direction)
    dimsd : :obj:`tuple`
        Dimensions of model vector (vector will be reshaped internal into
        a two-dimensional array of size :math:`[n_x \times n_t]`)
    table : :obj:`np.ndarray` or :obj:`dask.array.core.Array`, optional
        Look-up table of indeces of size
        :math:`[n_x \times n_{x0} \times n_{t0}]`
    dtable : :obj:`np.ndarray` or :obj:`dask.array.core.Array`, optional
        Look-up table of decimals remainders for linear interpolation of same
        size as ``dtable``
    fh : :obj:`np.ndarray`, optional
        Function handle that returns an index (and a fractional value in case
        of ``interp=True``) to be used for spreading/stacking given indices
        in :math:`x0` and :math:`t` axes (if ``None`` use look-up table
        ``table``)
    interp : :obj:`bool`, optional
        Apply linear interpolation (``True``) or nearest interpolation
        (``False``) during stacking/spreading along parametric curve. To be
        used only if ``engine='numba'``, inferred directly from the number of
        outputs of ``fh`` for ``engine='numpy'``
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array`
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
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Raises
    ------
    KeyError
        If ``engine`` is neither ``numpy`` nor ``numba``
    NotImplementedError
        If both ``table`` and ``fh`` are not provided
    ValueError
        If ``table`` has shape different from
        :math:`[n_{x0} \times n_t0 \times n_x]`

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Spread` for implementation
    details.

    """
    def __init__(self, dims, dimsd, table=None, dtable=None,
                 compute=(False, False), todask=(False, False),
                 dtype='float64'):

        # input parameters
        self.dims, self.dimsd = dims, dimsd
        self.nx0, self.nt0 = self.dims[0], self.dims[1]
        self.nx, self.nt = self.dimsd[0], self.dimsd[1]
        self.table = table
        self.dtable = dtable
        self.compute =  compute
        self.todask =  todask

        # find out if table has correct size
        if self.table.shape != (self.nx, self.nx0, self.nt0):
            raise ValueError('table must have shape [nx x nx0 x nt0] '
                             'for dask engine')
        above_nt = (self.table > self.nt).any().compute()
        if above_nt:
            raise ValueError('values in table must be smaller than nt')
        self.table = self.table.reshape(self.nx, self.nx0 * self.nt0)

        # find out if linear interpolation has to be carried out
        self.interp = False
        if dtable is not None:
            if self.dtable.shape != (self.nx, self.nx0, self.nt0):
                raise ValueError('dtable must have shape [nx x nx0 x nt0] '
                                 'for dask engine')
            self.interp = True
            self.dtable = \
                self.dtable.reshape(self.nx, self.nx0 * self.nt0)

        self.shape = (int(np.prod(self.dimsd)), int(np.prod(self.dims)))
        self.dtype = np.dtype(dtype)
        self.explicit = False
        self.Op = None

    def _matvec(self, x):
        def _matvec_chunk(y, x, table, dtable, dims, nt0, nx0):
            """apply matvec for a chunk of the data vector
            """
            chunks = table.shape[0]
            table = table.reshape(chunks, nx0, nt0).transpose(1, 2, 0)
            if dtable is not None:
                dtable = dtable.reshape(chunks, nx0, nt0).transpose(1, 2, 0)
            for ix0 in range(dims[0]):
                for it in range(dims[1]):
                    indices = table[ix0, it]
                    if dtable is not None:
                        dindices = dtable[ix0, it]
                    mask = np.argwhere(~np.isnan(indices))
                    if mask.size > 0:
                        indices = (indices[mask]).astype(np.int)
                        if dtable is None:
                            y[mask, indices] += x[ix0, it]
                        else:
                            y[mask, indices] += \
                                (1 - dindices[mask]) * x[ix0, it]
                            y[mask, indices + 1] += \
                                dindices[mask] * x[ix0, it]
            return y

        x = x.reshape(self.dims)
        y = da.zeros(self.dimsd, chunks=(self.table.chunksize[0],
                                         self.dimsd[1]))
        y = da.map_blocks(_matvec_chunk, y, x, self.table, self.dtable,
                          self.dims, self.nt0, self.nx0, dtype=self.dtype,
                          name='spread_forward_dask')
        y = y.ravel()
        return y

    def _rmatvec(self, x):
        def _rmatvec_chunk(y, x, table, dtable, dims):
            """apply matvec for a chunk of the data vector
            """
            y, x = y.squeeze(), x.squeeze()
            table = table.transpose(1, 2, 0)
            if dtable is not None:
                dtable = dtable.transpose(1, 2, 0)
            for ix0 in range(dims[0]):
                for it in range(dims[1]):
                    indices = table[ix0, it]
                    if dtable is not None:
                        dindices = dtable[ix0, it]
                    mask = np.argwhere(~np.isnan(indices))
                    if mask.size > 0:
                        indices = (indices[mask]).astype(np.int)
                        if dtable is None:
                            y[ix0, it] = np.sum(x[mask, indices])
                        else:
                            y[ix0, it] = \
                                np.sum(x[mask, indices] * (1 - dindices[mask])) \
                                + np.sum(x[mask, indices + 1] * dindices[mask])
            y = y[np.newaxis, :, :]
            return y

        x = x.reshape(self.dimsd[0], self.dimsd[1], 1)
        y = da.zeros((self.table.npartitions, self.nx0, self.nt0),
                     chunks=(1, self.nx0, self.nt0))
        y = da.map_blocks(_rmatvec_chunk, y, x,
                          self.table.reshape(self.nx, self.nx0, self.nt0),
                          self.dtable.reshape(self.nx, self.nx0, self.nt0),
                          self.dims, dtype=np.float, name='spread_adj_dask')
        y = y.sum(axis=0).ravel()
        return y
