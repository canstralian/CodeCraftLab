import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd

# Handle missing plotly dependency
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
        initial_sidebar_state="expanded"
    )

def display_sidebar():
    """Display sidebar with navigation options."""
    st.sidebar.title("CodeGen Hub")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Navigation")
    st.sidebar.markdown("""
    - [Home](/)
    - [Dataset Management](/Dataset_Management)
    - [Model Training](/Model_Training)
    - [Code Generation](/Code_Generation)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Resources")
    st.sidebar.markdown("""
    - [Hugging Face Hub](https://huggingface.co/models)
    - [Transformers Documentation](https://huggingface.co/docs/transformers/index)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2023 CodeGen Hub")

def timestamp():
    """Return a formatted timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_log(message, log_type="INFO"):
    """Add a log message to the training logs."""
    log_entry = f"[{timestamp()}] [{log_type}] {message}"
    if 'training_logs' not in st.session_state:
        st.session_state.training_logs = []
    st.session_state.training_logs.append(log_entry)
    return log_entry

def display_logs():
    """Display the training logs in a text area."""
    if 'training_logs' in st.session_state and st.session_state.training_logs:
        logs = '\n'.join(st.session_state.training_logs)
        st.text_area('Training Logs', logs, height=300)
    else:
        st.text_area('Training Logs', 'No logs available.', height=300)

def plot_training_progress(model_id):
    """Plot training progress for a model."""
    if 'training_progress' in st.session_state and model_id in st.session_state.training_progress:
        progress = st.session_state.training_progress[model_id]
        
        if 'loss_history' in progress and progress['loss_history']:
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Step': list(range(1, len(progress['loss_history']) + 1)),
                'Loss': progress['loss_history']
            })
            
            if px is not None:
                # Create plot with plotly if available
                fig = px.line(df, x='Step', y='Loss', title=f'Training Loss for {model_id}')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to simple display without plotly
                st.subheader(f"Training Loss for {model_id}")
                st.dataframe(df)
                st.info("Install plotly for interactive charts")
        else:
            st.info("No training progress data available yet.")
    else:
        st.info("No training progress data available for this model.")

def format_code(code):
    """Format code for display."""
    return code.strip()

def create_folder_if_not_exists(folder_path):
    """Create folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return True
    return False
def initialize_session_state():
    """Initialize all required session state variables with defaults."""
    defaults = {
        'datasets': {},
        'trained_models': {},
        'training_logs': [],
        'training_progress': {},
        'active_jobs_count': 0,
        'stop_events': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
