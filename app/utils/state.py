###################################################################################################################################
#  Utility functions for managing and persisting session state in Streamlit app. Includes functionality for:
#  - Persisting and deleting state variables.
#  - Checking and initializing session state values.
#  - Managing persisted state through an interactive interface.
###################################################################################################################################


import streamlit as st
from streamlit import session_state as _state


_PERSIST_STATE_KEY = f"{__name__.split('.')[-2]}_PERSIST"


def persist(key: str, value=None):
    """
    Mark a key-value pair as persistent in the session state.
    If the value is provided, store it in session state.
    """
    if _PERSIST_STATE_KEY not in _state:
        _state[_PERSIST_STATE_KEY] = set()

    if value is not None:
        _state[key] = value

    _state[_PERSIST_STATE_KEY].add(key)
    return key


def check_state(*keys, check_all=False):
    """
    Checks if either all or any of the provided keys exist in the session state and are not None.
    """
    if check_all:
        return all(_state.get(key) is not None for key in keys)
    else:
        return any(_state.get(key) is not None for key in keys)


def delete_state(*keys: str, delete_all: bool = False):
    """
    Delete a persisted key from the persistent state,
    or delete all if 'delete_all' is True.
    """
    if delete_all:
        _state.clear()
        _state.pop(_PERSIST_STATE_KEY, None)
    else:
        for key in keys:
            _state.get(_PERSIST_STATE_KEY, []).remove(key)
            _state.pop(key, None)


def init_values(key, value=None):
    """
    Set a key in session state to a given value if it does not already exist.
    """
    if key not in _state:
        _state[key] = value


def display_state(key: str = None, display_all: bool = False):
    """Display session state based on the provided parameters."""
    if key is not None:
        if key in _state:
            st.write(f"**{key}:** {_state[key]}")
        else:
            st.write(f"Key '{key}' not found.")
    elif display_all:
        for key, value in _state.items():
            st.write(f"**{key}:** {value}")
    else:
        st.write("No state.")


def display_persistent_state():
    """Display only the persistent session state."""
    if _PERSIST_STATE_KEY in _state:
        keys = _state[_PERSIST_STATE_KEY]

        for key in keys:
            if key in _state:
                st.write(f"**{key}:** {_state[key]}")
    else:
        st.write("No persistent state.")


def manage_state(container):
    if _PERSIST_STATE_KEY in _state:
        with container.expander("Manage Persistent State", expanded=False):
            keys = list(_state[_PERSIST_STATE_KEY])
            if keys:

                option = st.radio(
                    "Choose an action",
                    ["Persisted Keys", "Current States", "Compare"],
                    help=f"The persisted keys are stored in a special list (set) within the session state, which is `{_PERSIST_STATE_KEY}`",
                )

                if option == "Persisted Keys":
                    key = st.selectbox(
                        f"Select a persisted key (total: `{len(keys)})`", keys
                    )
                    right, left = st.columns(2)
                    value = _state.get(key, "Not initialized")
                    if (
                        (
                            isinstance(value, (str, list, dict))
                            and len(str(value)) > 1000
                        )
                        or (key == "ot_df")
                        or (key == "ppi_df")
                    ):
                        if right.button("Values"):
                            st.write("Value too large to display.")
                    else:
                        if right.button("Values"):
                            st.write(f"{value}")
                    if left.button("Delete"):
                        delete_state(key)
                        st.success(f"Deleted {key} from persistent state.")

                elif option == "Current States":
                    current_keys = list(_state.keys())
                    current_key = st.selectbox(
                        f"Select a current state key (total: `{len(current_keys)}`)",
                        current_keys,
                    )
                    right, left = st.columns(2)
                    value = _state.get(current_key, "Not initialized")
                    if (
                        (
                            isinstance(value, (str, list, dict))
                            and len(str(value)) > 1000
                        )
                        or (current_key == "ot_df")
                        or (current_key == "ppi_df")
                    ):
                        if right.button("Values"):
                            st.write("Value too large to display.")
                    else:
                        if right.button("Values"):
                            st.write(f"{value}")
                    if left.button("Delete"):
                        delete_state(current_key)
                        st.success(f"Deleted {current_key} from session state.")

                elif option == "Compare":

                    persisted_keys = set(_state.get(_PERSIST_STATE_KEY, []))
                    current_keys = set(_state.keys())

                    missing_keys = current_keys - persisted_keys
                    if missing_keys:
                        st.write("Keys that are not persisted:")
                        st.write(missing_keys)
                    else:
                        st.success("No extra keys found in the current state.")

                    if st.button("Reset state"):
                        if missing_keys:
                            for key in missing_keys:
                                del _state[key]
                            st.success(f"deleted {len(missing_keys)} keys")
