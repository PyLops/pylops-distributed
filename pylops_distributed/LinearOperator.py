from pylops import LinearOperator as pLinearOperator

class LinearOperator(pLinearOperator):
    """Common interface for performing matrix-vector products.

    This class is an overload of the
    :py:class:`pylops.LinearOperator` class. It adds
    functionalities to choose whether to compute forward and adjoint or simply
    define their graph.

    In order to avoid the input vectors to be converted to ``numpy`` array
    by the original ``matvec`` and ``rmatvec``, those methods are overwritten
    here to simply call their private methods ``_matvec`` and ``_rmatvec``
    without any prior check on the input vectors.

    .. note:: End users of PyLops should not use this class directly but simply
      use operators that are already implemented. This class is meant for
      developers and it has to be used as the parent class of any new operator
      developed within PyLops-distibuted. Find more details regarding implementation of
      new operators at :ref:`addingoperator`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array`

    """
    def __init__(self, Op=None, explicit=False, compute=(False, False)):
        print(Op)
        super().__init__(Op=Op, explicit=explicit)
        self.compute = compute

    def matvec(self, x):
        y = self.Op._matvec(x)
        if self.compute[0]:
            y = y.compute()
        return y

    def rmatvec(self, x):
        y = self.Op._rmatvec(x)
        if self.compute[1]:
            y = y.compute()
        return y
