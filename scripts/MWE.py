import diffrax as dfx
import jax
from jax import numpy as jnp
from jax import random as jr


def main():
    def generate_path(shape):
        shape_dtype = jax.ShapeDtypeStruct(
            shape=shape,
            dtype=jnp.float32,
        )

        @jax.jit
        def eval_path(path, t0, t1):
            print("Compiling path evaluation...")
            return path.evaluate(t0=t0, t1=t1)

        dfx.UnsafeBrownianPath
        path = dfx.VirtualBrownianTree(
            t0=0,
            t1=1,
            tol=1e-5,
            shape=shape_dtype,
            key=jr.PRNGKey(0),
            levy_area=dfx.SpaceTimeLevyArea,
        )
        eval_path(path, 0.0, 1.0)
        print("Path evaluated successfully.")

    generate_path(shape=(jnp.int32(2),))
    generate_path(shape=(jnp.int32(2),))


def print_info():
    shape_int32 = jax.ShapeDtypeStruct(shape=(jnp.int32(2),), dtype=jnp.float32)
    shape_int = jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32)

    print(
        f"For int32: shape = {shape_int32.shape}  |  shape[0] = {shape_int32.shape[0]} |  type = {type(shape_int32.shape[0])}"
    )
    print(
        f"For int: shape = {shape_int.shape} |  shape[0] = {shape_int.shape[0]}  |  type = {type(shape_int.shape[0])}"
    )


if __name__ == "__main__":
    print_info()
    main()
