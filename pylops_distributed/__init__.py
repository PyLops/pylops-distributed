from .LinearOperator import LinearOperator
from .basicoperators import MatrixMult
from .basicoperators import Diagonal
from .basicoperators import Identity
from .basicoperators import Transpose
from .basicoperators import Spread

from .utils.backend import dask

from . import basicoperators
from . import signalprocessing
from . import utils
from . import waveeqprocessing


try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'