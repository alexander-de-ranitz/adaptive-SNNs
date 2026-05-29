import diffrax as dfx
import equinox as eqx

from adaptive_SNN.models.environments.base import (
    AbstractEnvironment,
    AbstractEnvironmentState,
)


class ExternalController(eqx.Module):
    """A wrapper to run an environment with an external controller.

    Control function must be provided via the `get_env_input` key in the args dict passed to the solver.
    This function should have signature `get_env_input(t, x, args) -> env_input` where `x` is the current environment state.
    The output of this function will be passed as `env_input` to the environment's drift and update functions.
    """

    environment: AbstractEnvironment

    @property
    def initial(self):
        return self.environment.initial

    def pre_step_update(self, t, x: AbstractEnvironmentState, args):
        return self.environment.pre_step_update(t, x, args)

    def drift(self, t, x: AbstractEnvironmentState, args):
        input = args.get("get_env_input")(t, x, args)
        return self.environment.drift(t, x, args, env_input=input)

    def diffusion(self, t, x: AbstractEnvironmentState, args):
        return self.environment.diffusion(t, x, args)

    def update(self, t, x: AbstractEnvironmentState, args):
        input = args.get("get_env_input")(t, x, args)
        return self.environment.update(t, x, args, env_input=input)

    @property
    def noise_shape(self):
        return self.environment.noise_shape

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.environment.noise_shape,
            key=key,
            levy_area=dfx.SpaceTimeLevyArea,
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
