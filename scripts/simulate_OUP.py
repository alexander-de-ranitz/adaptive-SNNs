import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models import OUP


def main():
    key = jr.PRNGKey(1)
    t0 = 0
    t1 = 10
    dt0 = 1e-4

    def run_and_plot_OU_process(tau, D):
        noise_model = OUP(tau=tau, noise_std=D, mean=0.0, dim=2)

        expected_std = jnp.sqrt(D * tau / 2)

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
        print(f"Expected std: {expected_std}")
        for i in range(x.shape[1]):
            plt.plot(t, x[:, i], label=labels[i], color=colors[i])
            print(f"{labels[i]} mean: {jnp.mean(x[:, i])}, std: {jnp.std(x[:, i])}")

    run_and_plot_OU_process(tau=3e-3, D=1e-7)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Conductance")
    plt.title("OU Process")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
