"""Backend dispatch for slime_volleyball.

Manages the active array library (numpy or jax.numpy) used by all simulation code.
The backend is set once per process and cannot be changed after first use.

Usage:
    from slime_volleyball.backend import xp
    arr = xp.zeros((3, 3))
"""

import numpy

_backend_name: str = "numpy"
_backend_set: bool = False
xp_module = numpy


class BackendProxy:
    """Proxy that delegates attribute access to the current backend module."""

    def __getattr__(self, name):
        return getattr(xp_module, name)

    def __repr__(self):
        return f"BackendProxy(current={_backend_name})"


xp = BackendProxy()


def set_backend(name: str) -> None:
    """Set the active array backend ('numpy' or 'jax')."""
    global _backend_name, _backend_set, xp_module

    if _backend_set and name != _backend_name:
        raise RuntimeError(
            f"Backend already set to '{_backend_name}'. "
            "Cannot change backend after first environment creation."
        )

    if name == "numpy":
        xp_module = numpy
    elif name == "jax":
        try:
            import jax.numpy as jnp

            xp_module = jnp
        except ImportError:
            raise ImportError(
                "Backend 'jax' requested but JAX is not installed.\n"
                "Install JAX with: pip install jax jaxlib\n"
                "Or use the numpy backend (default)."
            )
    else:
        raise ValueError(f"Unknown backend: {name}. Use 'numpy' or 'jax'.")

    _backend_set = True
    _backend_name = name


def get_backend() -> str:
    """Return the currently active backend name ('numpy' or 'jax')."""
    return _backend_name


def is_backend_set() -> bool:
    """Return True if the backend has been explicitly set."""
    return _backend_set


def _reset_backend_for_testing() -> None:
    """Reset global backend state for test isolation. Test-only."""
    global _backend_name, _backend_set, xp_module
    _backend_name = "numpy"
    _backend_set = False
    xp_module = numpy
