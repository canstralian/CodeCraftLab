import os
import time
from datetime import datetime

# Import streamlit first
try:
    import streamlit as st
except ImportError:
    # Create a dummy st object if streamlit not available
    class DummySt:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    st = DummySt()

# Handle other dependencies gracefully
try:
    import plotly.express as px
except ImportError:
    px = None

# Import pandas separately to avoid circular imports
try:
    import pandas as pd
except ImportError:
    pd = None


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

        # Remove circular dependency by not calling initialize_session_state
        # Only create logs array if needed
        if "training_logs" not in st.session_state:
            st.session_state.training_logs = []

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


def format_code(code_text):
    """
    Format code by removing extra whitespace and normalizing indentation.

    Args:
        code_text (str): Code text to format

    Returns:
        str: Formatted code
    """
    if not code_text:
        return ""

    # Split into lines and remove trailing whitespace
    lines = [line.rstrip() for line in code_text.split("\n")]

    # Remove leading and trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    return "\n".join(lines)


def display_logs(max_logs=20):
    """
    Display training logs in the Streamlit app.

    Args:
        max_logs (int): Maximum number of logs to display
    """
    if "training_logs" not in st.session_state or not st.session_state.training_logs:
        st.info("No training logs available.")
        return

    logs = st.session_state.training_logs[-max_logs:]

    for log in logs:
        # Determine log type and display with appropriate styling
        if "[ERROR]" in log:
            st.error(log)
        elif "[WARNING]" in log:
            st.warning(log)
        else:
            st.info(log)


def plot_training_progress(model_id):
    """
    Plot training progress for a model.

    Args:
        model_id (str): ID of the model
    """
    if px is None or pd is None:
        st.warning(
            "Plotly or pandas is not available. Install them with `pip install plotly pandas`."
        )
        return

    if (
        "training_progress" not in st.session_state
        or model_id not in st.session_state.training_progress
    ):
        st.warning(f"No training progress data available for model {model_id}.")
        return

    progress = st.session_state.training_progress[model_id]

    if not progress.get("loss_history"):
        st.info("No training loss data available yet.")
        return

    # Create a dataframe for plotting
    try:
        # Create simple data for plotting even without pandas
        if pd is None:
            epochs = list(range(1, len(progress["loss_history"]) + 1))
            losses = progress["loss_history"]
            st.line_chart(dict(zip(epochs, losses)))
            return

        # With pandas available, create a more sophisticated plot
        df = pd.DataFrame(
            {
                "epoch": list(range(1, len(progress["loss_history"]) + 1)),
                "loss": progress["loss_history"],
            }
        )

        # Create plot
        fig = px.line(
            df, x="epoch", y="loss", title=f"Training Loss for {model_id}", markers=True
        )

        # Customize layout
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting training progress: {str(e)}")
        # Fallback to basic chart
        st.line_chart(progress["loss_history"])
