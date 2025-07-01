###################################################################################################################################
#  Utility functions for UI management, including task/result statuses, flash messages, and model results display in Streamlit app
###################################################################################################################################


import time

import streamlit as st
from streamlit import session_state as _state
from utils.state import check_state, persist


def flash_message(*keys, message: str, col=None, **kwargs) -> None:
    keys_list = [key.strip() for key in keys]

    all_keys = kwargs.get("all_keys", True)
    if check_state(*keys_list, check_all=all_keys):
        msg_obj = col.success(message) if col else st.success(message)
        timeout = kwargs.get("timeout", 1)
        if timeout > 0:
            time.sleep(timeout)
            if hasattr(msg_obj, "empty"):
                msg_obj.empty()


def task_status(stage, status) -> None:
    if "task_status" not in _state:
        _state["task_status"] = {}
    _state["task_status"][stage] = status
    persist("task_status", _state["task_status"])
    st.toast(stage, icon="✅")


def show_task_status(container, expand=True) -> None:
    if "task_status" in _state:
        cont = container.expander("Current Status", expanded=expand)
        for stage, status in _state["task_status"].items():
            cont.write(f"{status}{stage}")


def results_status(stage, results, **kwargs) -> None:
    dict_name = kwargs.get("dict_name", None)
    dict_name = dict_name if dict_name is not None else "results_status"
    if dict_name not in _state:
        _state[dict_name] = {}
    _state[dict_name][stage] = results
    persist(dict_name, _state[dict_name])


def show_results_status(container, expand=True, **kwargs) -> None:
    dict_name = kwargs.get("dict_name", None)
    dict_name = dict_name if dict_name is not None else "results_status"

    if dict_name in _state:
        header = kwargs.get("header", None)
        header = header if header is not None else "Results Status"

        cont = container.expander(header, expanded=expand)
        output = ""
        stages_to_show = kwargs.get("stages", None)

        for stage, status in _state[dict_name].items():
            if stages_to_show is None or stage in stages_to_show:
                output += f"{stage}: {status}\n"

        if output:
            cont.code(output)


def show_model_results(container, expand=True, **kwargs) -> None:
    """Display training status, and metric results. Available from Model Training page."""
    if "model_results" in _state:
        model_cont = container.expander(f"Validation Results for `{_state['previous_classifier']}`", expanded=expand)
        model_cont.info(f"{_state['classifier']}")
        for stage, results in _state["model_results"].items():
            metrics_output = "\n".join([f"{metric}: {value:.4f}" for metric, value in results.items()])
            model_cont.code(metrics_output)
