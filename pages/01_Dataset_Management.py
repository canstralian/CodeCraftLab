import time

import streamlit as st
from data_utils import get_dataset_info, list_available_datasets, process_python_dataset
from utils import add_log, display_sidebar, set_page_config

# Import pandas after other imports to avoid circular dependency
import pandas as pd

# Set page configuration
set_page_config()

# Display sidebar
display_sidebar()

# Title
st.title("Dataset Management")
st.markdown("Upload and manage your Python code datasets for model training.")

# Create tabs for different dataset operations
tab1, tab2 = st.tabs(["Upload Dataset", "View Datasets"])

with tab1:
    st.subheader("Upload a New Dataset")

    # Dataset name input
    dataset_name = st.text_input("Dataset Name", placeholder="e.g., python_functions")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Python Code Dataset",
        type=["py", "json", "csv"],
        help="Upload Python code files (.py), JSON files containing code snippets, or CSV files with code columns",
    )

    # Dataset upload options
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dataset Format")
        st.markdown(
            """
        - **Python files (.py)**: Will be split into examples by function/class definitions
        - **JSON files (.json)**: Should contain a list of objects with a 'code' field
        - **CSV files (.csv)**: Should have a 'code' column
        """
        )

    with col2:
        st.markdown("### Processing Options")
        auto_split = st.checkbox(
            "Automatically split into train/validation sets", value=True
        )
        split_ratio = st.slider(
            "Validation Split Ratio",
            min_value=0.1,
            max_value=0.3,
            value=0.2,
            step=0.05,
            disabled=not auto_split,
        )

    # Process button
    if st.button("Process Dataset"):
        if not dataset_name:
            st.error("Please provide a dataset name")
        elif not uploaded_file:
            st.error("Please upload a file")
        elif dataset_name in list_available_datasets():
            st.error(
                f"Dataset with name '{dataset_name}' already exists. Please choose a different name."
            )
        else:
            with st.spinner("Processing dataset..."):
                success = process_python_dataset(uploaded_file, dataset_name)
                if success:
                    st.success(f"Dataset '{dataset_name}' processed successfully!")
                    add_log(f"Dataset '{dataset_name}' uploaded and processed")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to process dataset. Check logs for details.")

with tab2:
    st.subheader("Available Datasets")

    # Get available datasets
    available_datasets = list_available_datasets()

    if not available_datasets:
        st.info("No datasets available. Upload a dataset in the 'Upload Dataset' tab.")
    else:
        # Dataset selection
        selected_dataset = st.selectbox("Select a Dataset", available_datasets)

        if selected_dataset:
            # Get dataset info
            dataset_info = get_dataset_info(selected_dataset)

            if dataset_info:
                # Display dataset information
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Dataset Information")
                    st.markdown(f"**Name:** {dataset_info['name']}")
                    st.markdown(f"**Total Examples:** {dataset_info['size']}")
                    st.markdown(f"**Training Examples:** {dataset_info['train_size']}")
                    st.markdown(
                        f"**Validation Examples:** {dataset_info['validation_size']}"
                    )
                    st.markdown(f"**Created:** {dataset_info['created_at']}")

                with col2:
                    st.markdown("### Dataset Structure")
                    columns = dataset_info.get("columns", [])
                    for col in columns:
                        st.markdown(f"- {col}")

                # Display sample data
                st.markdown("### Sample Data")

                # Get the dataset
                dataset = st.session_state.datasets[selected_dataset]["data"]

                # Display first few examples
                if "train" in dataset and len(dataset["train"]) > 0:
                    sample_size = min(5, len(dataset["train"]))
                    for i in range(sample_size):
                        with st.expander(f"Example {i+1}"):
                            st.code(
                                dataset["train"][i].get("code", "# No code available"),
                                language="python",
                            )
                else:
                    st.info("No examples available to display")

                # Actions
                st.markdown("### Actions")
                if st.button("Delete Dataset", key="delete_dataset"):
                    if selected_dataset in st.session_state.datasets:
                        del st.session_state.datasets[selected_dataset]
                        add_log(f"Dataset '{selected_dataset}' deleted")
                        st.success(
                            f"Dataset '{selected_dataset}' deleted successfully!"
                        )
                        time.sleep(1)
                        st.rerun()
