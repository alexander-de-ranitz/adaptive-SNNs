import equinox as eqx
from jaxtyping import Array

from adaptive_SNN.models.networks import EligibilityLIFNetwork, GatedLIFNetwork


class CoupledWeightGatedLIFNetwork(GatedLIFNetwork):
    weight_coupling_indices: tuple[Array, Array]

    def __init__(self, *args, **kwargs):
        weight_coupling_indices = kwargs.pop("weight_coupling_indices")
        super().__init__(*args, **kwargs)
        self.weight_coupling_indices = weight_coupling_indices
        super().__post_init__()

    def update(self, t, x, args):
        x = super().update(t, x, args)
        x = eqx.tree_at(
            lambda x: x.W,
            x,
            x.W.at[self.weight_coupling_indices[0], self.N_neurons :].set(
                x.W[self.weight_coupling_indices[1], self.N_neurons :]
            ),
        )
        return x


class CoupledWeightEligibilityLIFNetwork(EligibilityLIFNetwork):
    weight_coupling_indices: tuple[Array, Array]

    def __init__(self, *args, **kwargs):
        weight_coupling_indices = kwargs.pop("weight_coupling_indices")
        super().__init__(*args, **kwargs)
        self.weight_coupling_indices = weight_coupling_indices
        super().__post_init__()

    def update(self, t, x, args):
        x = super().update(t, x, args)
        x = eqx.tree_at(
            lambda x: x.W,
            x,
            x.W.at[self.weight_coupling_indices[0], self.N_neurons :].set(
                x.W[self.weight_coupling_indices[1], self.N_neurons :]
            ),
        )
        return x


class CoupledNoiseGatedLIFNetwork(CoupledWeightGatedLIFNetwork):
    noise_coupling_indices: tuple[Array, Array]

    def __init__(self, *args, **kwargs):
        noise_coupling_indices = kwargs.pop("noise_coupling_indices")
        super().__init__(*args, **kwargs)
        self.noise_coupling_indices = noise_coupling_indices
        super().__post_init__()

    def drift(self, t, x, args):
        perturbations = x.perturbations
        perturbations = perturbations.at[self.noise_coupling_indices[0]].set(
            perturbations[self.noise_coupling_indices[1]]
        )
        x = eqx.tree_at(lambda x: x.perturbations, x, perturbations)
        return super().drift(t, x, args)


class CoupledNoiseEligibilityLIFNetwork(CoupledWeightEligibilityLIFNetwork):
    noise_coupling_indices: tuple[Array, Array]

    def __init__(self, *args, **kwargs):
        noise_coupling_indices = kwargs.pop("noise_coupling_indices")
        super().__init__(*args, **kwargs)
        self.noise_coupling_indices = noise_coupling_indices
        super().__post_init__()

    def drift(self, t, x, args):
        perturbations = x.perturbations
        perturbations = perturbations.at[self.noise_coupling_indices[0]].set(
            perturbations[self.noise_coupling_indices[1]]
        )
        x = eqx.tree_at(lambda x: x.perturbations, x, perturbations)
        return super().drift(t, x, args)
