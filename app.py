import os

import streamlit as st

from utils import add_log, display_sidebar, initialize_session_state, set_page_config

try:
    # Set the page configuration for the Streamlit app
    set_page_config()

    # Display the main title of the Streamlit app
    st.title("CodeGen Hub")

    st.markdown(
        """
        Welcome to CodeGen Hub - A platform for training and using code generation models with Hugging Face integration.

        ### Core Features:
        - Upload and preprocess Python code datasets for model training
        - Configure and train models with customizable parameters
        - Generate code predictions using trained models through an interactive interface
        - Monitor training progress with visualizations and detailed logs
        - Seamless integration with Hugging Face Hub for model management

        Navigate through the different sections using the sidebar menu.
    """
    )

    # Display the sidebar
    display_sidebar()

    # Initialize session state with proper error handling
    try:
        initialize_session_state()
    except Exception as e:
        st.error(f"Failed to initialize application state: {str(e)}")
        add_log(f"State initialization error: {str(e)}")

    # Display platform statistics with error handling
    try:
        st.subheader("Platform Statistics")
        col1, col2, col3 = st.columns(3)

        with st.spinner("Loading statistics..."):
            with col1:
                datasets_count = len(st.session_state.get("datasets", {}))
                st.metric("Datasets Available", datasets_count)

            with col2:
                models_count = len(st.session_state.get("trained_models", {}))
                st.metric("Trained Models", models_count)

            with col3:
                active_jobs = st.session_state.get("active_jobs_count", 0)
                st.metric("Active Training Jobs", active_jobs)

    except Exception as e:
        st.error("Failed to load platform statistics. Please refresh the page.")
        add_log(f"Statistics loading error: {str(e)}")

    # Getting Started section
    st.subheader("Getting Started")
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            """
            1. 📊 Start by uploading or selecting a Python code dataset in the **Dataset Management** section.
            2. 🛠️ Configure and train your model in the **Model Training** section.
        """
        )

    with col2:
        st.info(
            """
            3. 💡 Generate code predictions using your trained models in the **Code Generation** section.
            4. 🔄 Access your models on Hugging Face Hub for broader use.
        """
        )

except Exception as e:
    st.error("An unexpected error occurred. Please try refreshing the page.")
    add_log(f"Critical application error: {str(e)}")
