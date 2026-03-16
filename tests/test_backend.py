"""Tests for backend dispatch module (Phase 1)."""

import numpy as np
import pytest

from slime_volleyball.backend._dispatch import (
    _reset_backend_for_testing,
    get_backend,
    is_backend_set,
    set_backend,
)


@pytest.fixture(autouse=True)
def reset_backend():
    _reset_backend_for_testing()
    yield
    _reset_backend_for_testing()


def test_default_backend_is_numpy():
    assert get_backend() == "numpy"


def test_set_backend_numpy():
    set_backend("numpy")
    assert get_backend() == "numpy"
    assert is_backend_set()


def test_set_backend_jax():
    set_backend("jax")
    assert get_backend() == "jax"
    assert is_backend_set()


def test_cannot_change_backend():
    set_backend("numpy")
    with pytest.raises(RuntimeError, match="already set"):
        set_backend("jax")


def test_same_backend_is_fine():
    set_backend("numpy")
    set_backend("numpy")  # no error


def test_invalid_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        set_backend("tensorflow")


def test_proxy_resolves_numpy():
    from slime_volleyball.backend import xp

    arr = xp.zeros(3)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)


def test_proxy_resolves_jax():
    set_backend("jax")
    from slime_volleyball.backend import xp

    arr = xp.zeros(3)
    import jax.numpy as jnp

    assert isinstance(arr, jnp.ndarray)
