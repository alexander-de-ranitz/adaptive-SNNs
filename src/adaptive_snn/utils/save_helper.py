import dataclasses


def save_part_of_state(state, **to_save):
    """Helper function to save only parts of a state object.

    Save part of a state object by specifying which attributes to keep.
    For each attribute in the state object, if the corresponding keyword argument is given as True,
    the attribute is kept; otherwise, it is set to None.

    This function supports nested dataclasses. to_save can specify attributes at any level of the nested dataclass.
    When an attribute is itself a dataclass and is specified to be saved, all of its attributes are always saved.
    If you want to save only specific attributes of a nested dataclass, you need to specify those attributes individually in to_save.

    Parameters
    ----------
    state : eqx.Module
        The full state object.
    **to_save : dict
        Keyword arguments where keys are the names of the attributes to save
        and values are booleans indicating whether to save them. Note that if an attribute is not specified, it will not be saved.

    Returns
    -------
    eqx.Module
        A new state object containing only the specified attributes.
    """
    saved_state = {}
    for field in dataclasses.fields(state):
        value = getattr(state, field.name)
        # Check if the attribute is to be saved, if so, add it to saved_state
        if field.name in to_save and to_save[field.name]:
            saved_state[field.name] = value
        # If the attribute is a dataclass (i.e., a nested dataclass), recursively save its parts
        elif dataclasses.is_dataclass(value):
            # Pass the nested state itself so the reconstructed object keeps its original type.
            saved_state[field.name] = save_part_of_state(value, **to_save)
        # Otherwise, set it to None
        else:
            saved_state[field.name] = None
    return type(state)(**saved_state)
