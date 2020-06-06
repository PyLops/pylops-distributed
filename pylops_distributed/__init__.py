from .LinearOperator import LinearOperator
from .basicoperators import MatrixMult
from .basicoperators import Diagonal
from .basicoperators import Identity
from .basicoperators import Transpose
from .basicoperators import Roll
from .basicoperators import Spread
from .basicoperators import Restriction
from .basicoperators import VStack
from .basicoperators import HStack
from .basicoperators import Block
from .basicoperators import BlockDiag
from .basicoperators import FirstDerivative
from .basicoperators import SecondDerivative
from .basicoperators import Laplacian
from .basicoperators import Smoothing1D

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
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. pylops should be installed
    # properly!
    from datetime import datetime
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')
