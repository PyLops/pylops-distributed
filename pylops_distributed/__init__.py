from .basicoperators import Spread

from .utils.backend import dask


try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'