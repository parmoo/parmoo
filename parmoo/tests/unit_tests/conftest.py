""" Shared pytest configuration for the ParMOO unit tests.

Enabling 64-bit precision in jax is process-global state that must be set
before any jax array is created.  Doing it here, at conftest import time,
guarantees that every unit test sees the same precision whether the suite is
run in full or one file at a time.

"""

from jax import config

config.update("jax_enable_x64", True)
