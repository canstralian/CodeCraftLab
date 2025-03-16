
import streamlit as st

from utils import set_page_config

# Set the Streamlit page configuration
set_page_config()

# Display main app title
st.title("CodeGen Hub")

# App description with markdown formatting
st.markdown("""
    Welcome to CodeGen Hub - A platform for training and using code generation models with Hugging Face integration.

    ### Core Features:
    - 📂 Upload and preprocess Python code datasets for model training
    - 🎛️ Configure and train models with customizable parameters
    - 🤖 Generate code predictions using trained models through an interactive interface
    - 📊 Monitor training progress with visualizations and detailed logs
    - 🔗 Seamless integration with Hugging Face Hub for model management

    Navigate through the different sections using the sidebar menu.
""")


# Sidebar navigation using session state
def navigate(page):
    st.session_state["current_page"] = page


# Initialize session state variables using a loop
session_defaults = {
    "datasets": {},  # Stores uploaded datasets
    "trained_models": {},  # Stores trained model details
    "training_logs": [],  # Stores training logs
    "training_progress": {},  # Tracks active training jobs
    "current_page": "home",  # Default landing page
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Display sidebar with navigation buttons
with st.sidebar:
    st.header("Navigation")
    if st.button("🏗️ Dataset Management"):
        navigate("dataset_management")
    if st.button("🎯 Model Training"):
        navigate("model_training")
    if st.button("🔮 Code Generation"):
        navigate("code_generation")

# Render content dynamically based on session state
if st.session_state["current_page"] == "dataset_management":
    st.subheader("Dataset Management")
    st.write("Upload and manage your datasets here.")
elif st.session_state["current_page"] == "model_training":
    st.subheader("Model Training")
    st.write("Configure and train your models.")
elif st.session_state["current_page"] == "code_generation":
    st.subheader("Code Generation")
    st.write("Generate predictions using your trained models.")
else:
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

# Display platform statistics dynamically
st.subheader("Platform Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📂 Datasets Available", len(st.session_state.get("datasets",
                                                               {})))

with col2:
    st.metric("📦 Trained Models",
              len(st.session_state.get("trained_models", {})))

with col3:
    active_jobs = sum(
        1 for progress in st.session_state["training_progress"].values()
        if progress.get("status") == "running")
    st.metric("🚀 Active Training Jobs", active_jobs)
