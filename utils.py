import os
import time
from datetime import datetime

import streamlit as st

# Handle missing plotly dependency gracefully
try:
    import plotly.express as px
except ImportError:
    px = None


def set_page_config():
    """Set the page configuration with title and layout."""
    st.set_page_config(
        page_title="CodeGen Hub",
        page_icon="💻",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def display_sidebar():
    """Display sidebar with navigation options."""
    st.sidebar.title("CodeGen Hub")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Navigation")
    st.sidebar.markdown(
        """
    - [Home](/)
    - [Dataset Management](/Dataset_Management)
    - [Model Training](/Model_Training)
    - [Code Generation](/Code_Generation)
    """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Resources")
    st.sidebar.markdown(
        """
    - [Hugging Face Hub](https://huggingface.co/models)
    - [Transformers Documentation](https://huggingface.co/docs/transformers/index)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    """
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2023 CodeGen Hub")


def timestamp():
    """Return a formatted timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_log(message, log_type="INFO"):
    """Add a log message to the training logs."""
    try:
        if "training_logs" not in st.session_state:
            st.session_state.training_logs = []

        initialize_session_state()  # Initialize session state before logging.

        log_entry = f"[{timestamp()}] [{log_type}] {message}"
        st.session_state.training_logs.append(log_entry)
        return log_entry
    except Exception as e:
        st.error(f"Error adding log: {str(e)}")


def initialize_session_state():
    """Initialize session state variables."""
    try:
        defaults = {
            "datasets": {},
            "trained_models": {},
            "training_logs": [],
            "training_progress": {},
            "active_jobs_count": 0,
            "stop_events": {},
            "last_error": None,
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                add_log(f"Initialized {key} in session state")
    except Exception as e:
        st.error(f"Failed to initialize session state: {str(e)}")
