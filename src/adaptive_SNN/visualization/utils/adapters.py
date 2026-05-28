from adaptive_SNN.models import (
    AgentEnvSystem,
    SystemState,
)
from adaptive_SNN.models.networks import (
    Agent,
    AgentState,
    EligibilityLIFNetwork,
    GatedLIFNetwork,
    LIFNetwork,
    LIFState,
)


def get_LIF_state(state) -> LIFState:
    if isinstance(state, LIFState):
        return state
    elif isinstance(state, AgentState):
        return state.network_state
    elif isinstance(state, SystemState):
        return state.agent_state.network_state
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")


def get_LIF_model(model) -> LIFNetwork:
    if isinstance(model, (LIFNetwork, GatedLIFNetwork, EligibilityLIFNetwork)):
        return model
    elif isinstance(model, Agent):
        return model.network
    elif isinstance(model, AgentEnvSystem):
        return model.agent.network
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
