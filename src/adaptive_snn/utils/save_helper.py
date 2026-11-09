import dataclasses
from pathlib import Path

import numpy as np


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


def get_array_value(value, downcast_to_float32: bool) -> np.ndarray:
    arr = np.asarray(value)
    if downcast_to_float32 and arr.dtype.kind == "f" and arr.itemsize > 4:
        arr = arr.astype(np.float32)
    return arr


def save_named_result(
    path, result, *, downcast_to_float32: bool = True, **extra_arrays
):
    """Save a result as an .npz with named arrays.

    Optionally downcast to float32 to save space.
    Extra arrays can be passed (e.g. for the final_state) and will be saved alongside the other fields.
    """
    fields = {
        f.name: get_array_value(getattr(result, f.name), downcast_to_float32)
        for f in dataclasses.fields(result)
    }
    overlap = set(fields) & set(extra_arrays)
    if overlap:
        raise ValueError(f"extra_arrays collide with record fields: {sorted(overlap)}")
    np.savez(
        path,
        _fields=np.array(list(fields)),
        **fields,
        **{k: np.asarray(v) for k, v in extra_arrays.items()},
    )


def load_named_result(path) -> dict:
    """Load a named-array result as a plain dict"""
    with np.load(Path(path), allow_pickle=True) as data:
        if "_fields" not in data.files:
            if "ys_tree_def" in data.files or "ys" in data.files:
                raise ValueError(
                    f"{path} is a legacy pickled-treedef result file (no "
                    "'_fields' stamp). Load it with "
                    "adaptive_snn.utils.runner._load_existing_solution instead."
                )
            raise ValueError(f"{path} is not a named-result file: no '_fields' stamp.")
        names = list(data["_fields"])
        out = {name: data[name] for name in names}
        if "ts" in data.files:
            out["ts"] = data["ts"]
        return out
