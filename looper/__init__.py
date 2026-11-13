"""Custom looper — a minimal Hermes-architecture agent loop.

``from looper import Looper; Looper().chat("...")``
"""
from .loop import Looper
from .registry import registry
from .state import SessionStore

__all__ = ["Looper", "registry", "SessionStore"]
__version__ = "0.1.0"
