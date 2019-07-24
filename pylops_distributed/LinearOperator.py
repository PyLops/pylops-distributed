from pylops import LinearOperator as pLinearOperator

class LinearOperator(pLinearOperator):
    """Common interface for performing matrix-vector products.

    This class is an overload of the
    :py:class:`pylops.LinearOperator` class. It adds
    functionalities for distributed operators; specifically, it allows users
    choosing whether to compute forward and adjoint or simply define their
    graphs.

    In order to avoid the input vectors to be converted to ``numpy`` array
    by ``matvec`` and ``rmatvec`` of the parent class, those methods are
    overwritten here to simply call their private methods
    ``_matvec`` and ``_rmatvec`` without any prior check on the input vectors.

    .. note:: End users of PyLops should not use this class directly but simply
      use operators that are already implemented. This class is meant for
      developers and it has to be used as the parent class of any new operator
      developed within PyLops-distibuted. Find more details regarding
      implementation of new operators at :ref:`addingoperator`.

    Parameters
    ----------
    shape : :obj:`tuple`
        Operator shape
    dtype : :obj:`torch.dtype`, optional
        Type of elements in input array.
    Op : :obj:`pylops.LinearOperator`
        Operator to wrap in ``LinearOperator``
        (if ``None``, self must implement ``_matvec_`` and  ``_rmatvec_``)
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array`

    """
    def __init__(self, shape, dtype, Op=None, explicit=False,
                 compute=(False, False)):
        super().__init__(Op=Op, explicit=explicit)
        self.shape = shape
        self.dtype = dtype
        if Op is None:
            self.Op = None
        self.compute = compute

    def matvec(self, x):
        if self.Op is None:
            y = self._matvec(x)
        else:
            y = self.Op._matvec(x)
        if self.compute[0]:
            y = y.compute()
        return y

    def rmatvec(self, x):
        if self.Op is None:
            y = self._rmatvec(x)
        else:
            y = self.Op._rmatvec(x)
        if self.compute[1]:
            y = y.compute()
        return y
