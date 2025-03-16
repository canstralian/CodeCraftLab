import threading
import time

import pandas as pd
import streamlit as st

from data_utils import get_dataset_info, list_available_datasets
from model_utils import list_available_huggingface_models
from training_utils import (
    get_running_training_jobs,
    simulate_training,
    start_model_training,
    stop_model_training,
)
from utils import (
    add_log,
    display_logs,
    display_sidebar,
    plot_training_progress,
    set_page_config,
)

# Set page configuration
set_page_config()

# Display sidebar
display_sidebar()

# Title
st.title("Model Training")
st.markdown("Configure and train code generation models on your datasets.")

# Training configuration tab
tab1, tab2 = st.tabs(["Configure Training", "Monitor Jobs"])

with tab1:
    st.subheader("Train a New Model")

    # Model ID input
    model_id = st.text_input("Model ID", placeholder="e.g., my_codegen_model_v1")

    # Dataset selection
    available_datasets = list_available_datasets()
    if not available_datasets:
        st.warning(
            "No datasets available. Please upload a dataset in the Dataset Management section."
        )
        dataset_name = None
    else:
        dataset_name = st.selectbox("Select Dataset", available_datasets)

    # Model selection
    model_options = list_available_huggingface_models()
    base_model = st.selectbox("Select Base Model", model_options)

    # Training parameters
    st.markdown("### Training Parameters")
    col1, col2 = st.columns(2)

    with col1:
        learning_rate = st.number_input(
            "Learning Rate", min_value=1e-6, max_value=1e-3, value=2e-5, format="%.2e"
        )
        batch_size = st.slider("Batch Size", min_value=1, max_value=32, value=8, step=1)

    with col2:
        epochs = st.slider(
            "Number of Epochs", min_value=1, max_value=10, value=3, step=1
        )
        use_simulation = st.checkbox(
            "Use Simulation Mode (for demonstration)", value=True
        )

    # Start training button
    if st.button("Start Training", disabled=not dataset_name):
        if not model_id:
            st.error("Please provide a model ID")
        elif model_id in st.session_state.get("trained_models", {}):
            st.error(
                f"Model with ID '{model_id}' already exists. Please choose a different ID."
            )
        elif model_id in st.session_state.get("training_progress", {}):
            st.error(f"A training job for model '{model_id}' already exists.")
        else:
            # Initialize stop_events if not present
            if "stop_events" not in st.session_state:
                st.session_state.stop_events = {}

            # Start training (real or simulated)
            if use_simulation:
                st.session_state.stop_events[model_id] = simulate_training(
                    model_id, dataset_name, base_model, epochs
                )
                add_log(f"Started simulated training for model '{model_id}'")
            else:
                st.session_state.stop_events[model_id] = start_model_training(
                    model_id,
                    dataset_name,
                    base_model,
                    learning_rate,
                    batch_size,
                    epochs,
                )
                add_log(f"Started training for model '{model_id}'")

            st.success(f"Training job started for model '{model_id}'")
            time.sleep(1)
            st.rerun()

with tab2:
    st.subheader("Training Jobs")

    # Check if there are any training jobs
    if (
        "training_progress" not in st.session_state
        or not st.session_state.training_progress
    ):
        st.info(
            "No training jobs found. Start a new training job in the 'Configure Training' tab."
        )
    else:
        # List all training jobs
        all_jobs = list(st.session_state.training_progress.keys())
        selected_job = st.selectbox("Select Training Job", all_jobs)

        if selected_job:
            # Get job progress
            job_progress = st.session_state.training_progress[selected_job]

            # Display job status
            status = job_progress["status"]
            status_color = {
                "initialized": "blue",
                "running": "green",
                "completed": "green",
                "failed": "red",
                "stopped": "orange",
            }.get(status, "gray")

            st.markdown(f"### Status: :{status_color}[{status.upper()}]")

            # Display progress bar
            progress = job_progress["progress"]
            st.progress(progress / 100)

            # Display job details
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Job Details")
                st.markdown(f"**Model ID:** {selected_job}")
                st.markdown(
                    f"**Current Epoch:** {job_progress['current_epoch']}/{job_progress['total_epochs']}"
                )
                st.markdown(f"**Started At:** {job_progress['started_at']}")

                if job_progress["completed_at"]:
                    st.markdown(f"**Completed At:** {job_progress['completed_at']}")

            with col2:
                # Training controls
                st.markdown("### Controls")

                # Only show stop button for running jobs
                if status == "running" and selected_job in st.session_state.get(
                    "stop_events", {}
                ):
                    if st.button("Stop Training"):
                        stop_event = st.session_state.stop_events[selected_job]
                        stop_model_training(selected_job, stop_event)
                        st.success(f"Stopping training for model '{selected_job}'")
                        time.sleep(1)
                        st.rerun()

                # Add delete button for completed/failed/stopped jobs
                if status in ["completed", "failed", "stopped"]:
                    if st.button("Delete Job"):
                        del st.session_state.training_progress[selected_job]
                        if selected_job in st.session_state.get("stop_events", {}):
                            del st.session_state.stop_events[selected_job]
                        add_log(f"Deleted training job for model '{selected_job}'")
                        st.success(f"Training job for model '{selected_job}' deleted")
                        time.sleep(1)
                        st.rerun()

            # Display training progress plot
            st.markdown("### Training Progress")
            plot_training_progress(selected_job)

            # Display logs
            st.markdown("### Training Logs")
            display_logs()

# Display running jobs summary at the bottom
st.markdown("---")
st.subheader("Running Jobs Summary")
running_jobs = get_running_training_jobs()

if not running_jobs:
    st.info("No active training jobs")
else:
    for job in running_jobs:
        progress = st.session_state.training_progress[job]
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**{job}**")

        with col2:
            st.markdown(f"Epoch {progress['current_epoch']}/{progress['total_epochs']}")

        with col3:
            st.progress(progress["progress"] / 100)
