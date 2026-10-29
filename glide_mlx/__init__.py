"""GLiDE (fastino/GLiDE) on Apple Silicon with MLX. Not affiliated with Fastino."""
from .prompt import RequestError

__version__ = "0.1.0"


def __getattr__(name):
    if name == "GlideMLX":
        from .model import GlideMLX
        return GlideMLX
    raise AttributeError(name)


__all__ = ["GlideMLX", "RequestError", "__version__"]
