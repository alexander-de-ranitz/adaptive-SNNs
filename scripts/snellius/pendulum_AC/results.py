import equinox as eqx
from jaxtyping import Array


class SavedState(eqx.Module):
    mean_V: Array
    mean_RPE: Array
    var_RPE: Array
    mean_reward: Array
    mean_balance: Array
    var_balance: Array
    agent_output: Array
    mean_filtered_spikes_L: Array
    mean_filtered_spikes_R: Array
    mean_filtered_spikes_H: Array
    var_filtered_spikes_L: Array
    var_filtered_spikes_R: Array
    var_filtered_spikes_H: Array
    mean_W_actor_input: Array
    mean_W_actor_recurrent: Array
    var_W_actor_input: Array
    var_W_actor_recurrent: Array
    mean_W_critic: Array
    var_W_critic: Array
    fraction_clipped_dW: Array
