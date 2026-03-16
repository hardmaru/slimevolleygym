"""slime_volleyball.backend -- Array backend dispatch.

Provides a single array API that resolves to numpy or jax.numpy:

    from slime_volleyball.backend import xp
    arr = xp.zeros((3, 3))
"""

from slime_volleyball.backend._dispatch import get_backend, is_backend_set, set_backend

__all__ = ["xp", "set_backend", "get_backend", "is_backend_set"]


def __getattr__(name: str):
    if name == "xp":
        from slime_volleyball.backend import _dispatch

        return _dispatch.xp
    raise AttributeError(f"module 'slime_volleyball.backend' has no attribute {name!r}")
