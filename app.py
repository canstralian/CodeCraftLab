```python
import streamlit as st
# Import the `set_page_config` and `display_sidebar` functions from the `utils` module.
# These functions are responsible for setting up the Streamlit app's page configuration and displaying the sidebar, respectively.
from utils import set_page_config, display_sidebar
import os

# Set the page configuration for the Streamlit app.
# This function sets the app's title, icon, layout, and other parameters.
set_page_config()

# Display the main title of the Streamlit app.
st.title("CodeGen Hub")

# Display the description of the Streamlit app using Markdown formatting.
# This provides a more formatted and readable way to present the app's purpose and features.
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

# Display the sidebar for the Streamlit app.
# The sidebar will contain navigation options and other functionality.
display_sidebar()

# Create session state variables to store information across app pages.
# These variables will persist data between user interactions.
if 'datasets' not in st.session_state:
    # Initialize the 'datasets' dictionary in the session state.
    # This dictionary will store information about the uploaded datasets.
    st.session_state.datasets = {}

if 'trained_models' not in st.session_state:
    # Initialize the 'trained_models' dictionary in the session state.
    # This dictionary will store information about the trained models.
    st.session_state.trained_models = {}

if 'training_logs' not in st.session_state:
    # Initialize the 'training_logs' list in the session state.
    # This list will store the training logs for the models.
    st.session_state.training_logs = []

if 'training_progress' not in st.session_state:
    # Initialize the 'training_progress' dictionary in the session state.
    # This dictionary will store the progress information for the training jobs.
    st.session_state.training_progress = {}

# Display a "Getting Started" section with instructions for the user.
st.subheader("Getting Started")
col1, col2 = st.columns(2)

with col1:
    # Display the first set of instructions in the left column.
    st.info("""
        1. 📊 Start by uploading or selecting a Python code dataset in the **Dataset Management** section.
        2. 🛠️ Configure and train your model in the **Model Training** section.
    """)

with col2:
    # Display the second set of instructions in the right column.
    st.info("""
        3. 💡 Generate code predictions using your trained models in the **Code Generation** section.
        4. 🔄 Access your models on Hugging Face Hub for broader use.
    """)

# Display platform statistics if available.
st.subheader("Platform Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    # Display the number of datasets available.
    st.metric("Datasets Available", len(st.session_state.datasets))

with col2:
    # Display the number of trained models.
    st.metric("Trained Models", len(st.session_state.trained_models))

with col3:
    # Calculate the number of active training jobs.
    active_jobs = sum(1 for progress in st.session_state.training_progress.values() 
                     if progress.get('status') == 'running')
    # Display the number of active training jobs.
    st.metric("Active Training Jobs", active_jobs)
```