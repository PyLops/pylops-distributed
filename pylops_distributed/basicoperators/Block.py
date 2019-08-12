from pylops.basicoperators.Block import _Block
from pylops_distributed.basicoperators import HStack, VStack


def Block(ops, chunks=None, compute=(False, False),
          todask=(False, False), dtype=None):
    r"""Block operator.

    Create a block operator from N lists of M linear operators each.

    Parameters
    ----------
    ops : :obj:`list`
        List of lists of operators to be combined in block fashion
    chunks : :obj:`tuple`, optional
        Chunks for model and data (an array with a single chunk is created
        if ``chunks`` is not provided)
    compute : :obj:`tuple`, optional
        Compute the outcome of forward and adjoint or simply define the graph
        and return a :obj:`dask.array.array`
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

    Notes
    -----
    Refer to :class:`pylops.basicoperators.Block` for implementation
    details.

    """
    args_stacks = {'chunks': chunks, 'compute': compute, 'todask': todask}
    return _Block(ops, dtype=dtype, _HStack=HStack, _VStack=VStack,
                  args_HStack=args_stacks, args_VStack=args_stacks)
