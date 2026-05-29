import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax

jax.config.update("jax_enable_x64", True)
