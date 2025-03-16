import streamlit as st
from utils import set_page_config, display_sidebar
import os

# Set page configuration
set_page_config()

# Title and description
st.title("CodeGen Hub")
st.markdown("""
    Welcome to CodeGen Hub - A platform for training and using code generation models with Hugging Face integration.
    
    ### Core Features:
    - Upload and preprocess Python code datasets for model training
    - Configure and train models with customizable parameters
    - Generate code predictions using trained models through an interactive interface
    - Monitor training progress with visualizations and detailed logs
    - Seamless integration with Hugging Face Hub for model management
    
    Navigate through the different sections using the sidebar menu.
""")

# Display sidebar
display_sidebar()

# Create the session state for storing information across app pages
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []

if 'training_progress' not in st.session_state:
    st.session_state.training_progress = {}



# Display getting started card
st.subheader("Getting Started")
col1, col2 = st.columns(2)

with col1:
    st.info("""
        1. 📊 Start by uploading or selecting a Python code dataset in the **Dataset Management** section.
        2. 🛠️ Configure and train your model in the **Model Training** section.
    """)
    
with col2:
    st.info("""
        3. 💡 Generate code predictions using your trained models in the **Code Generation** section.
        4. 🔄 Access your models on Hugging Face Hub for broader use.
    """)

# Display platform statistics if available
st.subheader("Platform Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Datasets Available", len(st.session_state.datasets))
    
with col2:
    st.metric("Trained Models", len(st.session_state.trained_models))
    
with col3:
    # Calculate active training jobs
    active_jobs = sum(1 for progress in st.session_state.training_progress.values() 
                     if progress.get('status') == 'running')
    st.metric("Active Training Jobs", active_jobs)
