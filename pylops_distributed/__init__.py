from .LinearOperator import LinearOperator
from .basicoperators import MatrixMult
from .basicoperators import Diagonal
from .basicoperators import Identity
from .basicoperators import Transpose
from .basicoperators import Roll
from .basicoperators import Spread
from .basicoperators import VStack
from .basicoperators import HStack
from .basicoperators import Block
from .basicoperators import BlockDiag

from .optimization.cg import cg, cgls

from .utils.backend import dask

from . import utils
from . import basicoperators
from . import signalprocessing
from . import waveeqprocessing
from . import optimization

try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'