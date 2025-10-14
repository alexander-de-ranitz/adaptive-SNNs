import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr

from adaptive_SNN.models import EnvironmentModel


def test_environment():
    env = EnvironmentModel(dim=1)

    args = {"env_input": jnp.array([1.0])}
    solver = dfx.Euler()
    t0 = 0.0
    t1 = 10.0
    dt0 = 0.01
    y0 = env.initial
    key = jr.PRNGKey(0)
    terms = env.terms(key)
    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        adjoint=dfx.ForwardMode(),
    )
    ts = sol.ts
    ys = sol.ys
    assert ys.shape == (ts.shape[0], env.dim)
    assert jnp.allclose(ys[-1], jnp.array([1.0]), atol=1e-2)
