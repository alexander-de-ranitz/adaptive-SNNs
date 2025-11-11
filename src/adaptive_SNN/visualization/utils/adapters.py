from adaptive_SNN.models import (
    Agent,
    AgentEnvSystem,
    AgentState,
    LIFNetwork,
    LIFState,
    NoisyNetwork,
    NoisyNetworkState,
    SystemState,
)


def get_LIF_state(state) -> LIFState:
    if isinstance(state, LIFState):
        return state
    elif isinstance(state, NoisyNetworkState):
        return state.network_state
    elif isinstance(state, AgentState):
        return state.noisy_network.network_state
    elif isinstance(state, SystemState):
        return state.agent_state.noisy_network.network_state
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")


def get_LIF_model(model) -> LIFNetwork:
    if isinstance(model, LIFNetwork):
        return model
    elif isinstance(model, NoisyNetwork):
        return model.base_network
    elif isinstance(model, Agent):
        return model.noisy_network.base_network
    elif isinstance(model, AgentEnvSystem):
        return model.agent.noisy_network.base_network
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def get_noisy_network_state(state) -> NoisyNetworkState:
    if isinstance(state, NoisyNetworkState):
        return state
    elif isinstance(state, AgentState):
        return state.noisy_network
    elif isinstance(state, SystemState):
        return state.agent_state.noisy_network
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")


def get_noisy_network_model(model) -> NoisyNetwork:
    if isinstance(model, NoisyNetwork):
        return model
    elif isinstance(model, Agent):
        return model.noisy_network
    elif isinstance(model, AgentEnvSystem):
        return model.agent.noisy_network
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
