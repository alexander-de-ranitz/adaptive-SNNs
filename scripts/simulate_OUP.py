import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models import OUP


def main():
    key = jr.PRNGKey(0)
    t0 = 0
    t1 = 1
    dt0 = 0.0001

    # From Destexhe, 2001
    tau = 2.6  # 2.6 ms
    mean = 18e-9  # 18 nS
    sigma = 3.5e-9  # 3.5 nS
    D = 2 * sigma / tau

    noise_model = OUP(theta=tau, noise_scale=D, mean=mean, dim=2)
    solver = dfx.EulerHeun()
    terms = noise_model.terms(key)
    init_state = noise_model.initial
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 1000))

    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=init_state,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
        max_steps=None,
    )
    t = sol.ts
    x = sol.ys
    labels = ["Excitatory", "Inhibitory"]
    colors = ["g", "r"]
    for i in range(x.shape[1]):
        plt.plot(t, x[:, i], label=labels[i], color=colors[i])
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Conductance")
    plt.title("OU Process")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
