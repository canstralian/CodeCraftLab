import random
import threading
import time
from datetime import datetime

import streamlit as st

from utils import add_log, timestamp

# Handle missing dependencies
try:
    import pandas as pd
    import torch
    from datasets import Dataset, DatasetDict
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
    from transformers import TrainingArguments as HFTrainingArguments

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    HFTrainingArguments = None

    # For demo purposes
    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callback = type("obj", (object,), {"__init__": lambda self: None})

        def train(self):
            pass


def initialize_training_progress(model_id):
    """
    Initialize training progress tracking for a model.

    Args:
        model_id: Identifier for the model
    """
    if "training_progress" not in st.session_state:
        st.session_state.training_progress = {}

    st.session_state.training_progress[model_id] = {
        "status": "initialized",
        "current_epoch": 0,
        "total_epochs": 0,
        "loss_history": [],
        "started_at": timestamp(),
        "completed_at": None,
        "progress": 0.0,
    }


def update_training_progress(
    model_id, epoch=None, loss=None, status=None, progress=None, total_epochs=None
):
    """
    Update training progress for a model.

    Args:
        model_id: Identifier for the model
        epoch: Current epoch
        loss: Current loss value
        status: Training status
        progress: Progress percentage (0-100)
        total_epochs: Total number of epochs
    """
    if (
        "training_progress" not in st.session_state
        or model_id not in st.session_state.training_progress
    ):
        initialize_training_progress(model_id)

    progress_data = st.session_state.training_progress[model_id]

    if epoch is not None:
        progress_data["current_epoch"] = epoch

    if loss is not None:
        progress_data["loss_history"].append(loss)

    if status is not None:
        progress_data["status"] = status
        if status == "completed":
            progress_data["completed_at"] = timestamp()
            progress_data["progress"] = 100.0

    if progress is not None:
        progress_data["progress"] = progress

    if total_epochs is not None:
        progress_data["total_epochs"] = total_epochs


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """
    Tokenize a dataset for model training.

    Args:
        dataset: The dataset to tokenize
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dataset: Tokenized dataset
    """

    def tokenize_function(examples):
        return tokenizer(
            examples["code"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def train_model_thread(
    model_id, dataset_name, base_model_name, training_args, device, stop_event
):
    """
    Thread function for training a model.

    Args:
        model_id: Identifier for the model
        dataset_name: Name of the dataset to use
        base_model_name: Base model from Hugging Face
        training_args: Training arguments
        device: Device to use for training (cpu/cuda)
        stop_event: Threading event to signal stopping
    """
    try:
        # Get dataset
        dataset = st.session_state.datasets[dataset_name]["data"]

        # Initialize model and tokenizer
        add_log(f"Initializing model {base_model_name} for training")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Check if tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # Tokenize dataset
        add_log(f"Tokenizing dataset {dataset_name}")
        train_dataset = tokenize_dataset(dataset["train"], tokenizer)
        val_dataset = tokenize_dataset(dataset["validation"], tokenizer)

        # Update training progress
        update_training_progress(
            model_id, status="running", total_epochs=training_args.num_train_epochs
        )

        # Define custom callback to track progress
        class CustomCallback(Trainer.callback):
            def on_epoch_end(self, args, state, control, **kwargs):
                current_epoch = state.epoch
                epoch_loss = state.log_history[-1].get("loss", 0)
                update_training_progress(
                    model_id,
                    epoch=current_epoch,
                    loss=epoch_loss,
                    progress=(current_epoch / training_args.num_train_epochs) * 100,
                )
                add_log(
                    f"Epoch {current_epoch}/{training_args.num_train_epochs} completed. Loss: {epoch_loss:.4f}"
                )

                # Check if training should be stopped
                if stop_event.is_set():
                    add_log(f"Training for model {model_id} was manually stopped")
                    control.should_training_stop = True

        # Configure training arguments
        args = HFTrainingArguments(
            output_dir=f"./results/{model_id}",
            evaluation_strategy="epoch",
            learning_rate=training_args.learning_rate,
            per_device_train_batch_size=training_args.batch_size,
            per_device_eval_batch_size=training_args.batch_size,
            num_train_epochs=training_args.num_train_epochs,
            weight_decay=0.01,
            save_total_limit=1,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            callbacks=[CustomCallback],
        )

        # Train the model
        add_log(f"Starting training for model {model_id}")
        trainer.train()

        # Save the model
        if not stop_event.is_set():
            add_log(f"Training completed for model {model_id}")
            update_training_progress(model_id, status="completed")

            # Save to session state
            st.session_state.trained_models[model_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "info": {
                    "id": model_id,
                    "base_model": base_model_name,
                    "dataset": dataset_name,
                    "created_at": timestamp(),
                    "epochs": training_args.num_train_epochs,
                    "learning_rate": training_args.learning_rate,
                    "batch_size": training_args.batch_size,
                },
            }

    except Exception as e:
        add_log(f"Error during training model {model_id}: {str(e)}", "ERROR")
        update_training_progress(model_id, status="failed")


class TrainingArguments:
    def __init__(self, learning_rate, batch_size, num_train_epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs


def start_model_training(
    model_id, dataset_name, base_model_name, learning_rate, batch_size, epochs
):
    """
    Start model training in a separate thread.

    Args:
        model_id: Identifier for the model
        dataset_name: Name of the dataset to use
        base_model_name: Base model from Hugging Face
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        epochs: Number of training epochs

    Returns:
        threading.Event: Event to signal stopping the training
    """
    # Use simulate_training instead if transformers isn't available
    if not TRANSFORMERS_AVAILABLE:
        add_log("No transformers library available, using simulation mode")
        return simulate_training(model_id, dataset_name, base_model_name, epochs)

    # Create training arguments
    training_args = TrainingArguments(
        learning_rate=learning_rate, batch_size=batch_size, num_train_epochs=epochs
    )

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    add_log(f"Using device: {device}")

    # Initialize training progress
    initialize_training_progress(model_id)

    # Create stop event
    stop_event = threading.Event()

    # Start training thread
    training_thread = threading.Thread(
        target=train_model_thread,
        args=(
            model_id,
            dataset_name,
            base_model_name,
            training_args,
            device,
            stop_event,
        ),
    )
    training_thread.start()

    return stop_event


def stop_model_training(model_id, stop_event):
    """
    Stop model training.

    Args:
        model_id: Identifier for the model
        stop_event: Threading event to signal stopping
    """
    if stop_event.is_set():
        return

    add_log(f"Stopping training for model {model_id}")
    stop_event.set()

    # Update training progress
    if (
        "training_progress" in st.session_state
        and model_id in st.session_state.training_progress
    ):
        progress_data = st.session_state.training_progress[model_id]
        if progress_data["status"] == "running":
            progress_data["status"] = "stopped"
            progress_data["completed_at"] = timestamp()


def get_running_training_jobs():
    """
    Get list of currently running training jobs.

    Returns:
        list: List of model IDs with running training jobs
    """
    running_jobs = []

    if "training_progress" in st.session_state:
        for model_id, progress in st.session_state.training_progress.items():
            if progress["status"] == "running":
                running_jobs.append(model_id)

    return running_jobs


# For demo purposes - Simulate training progress without actual model training
def simulate_training_thread(
    model_id, dataset_name, base_model_name, epochs, stop_event
):
    """
    Simulate training progress for demonstration purposes.

    Args:
        model_id: Identifier for the model
        dataset_name: Name of the dataset to use
        base_model_name: Base model from Hugging Face
        epochs: Number of training epochs
        stop_event: Threading event to signal stopping
    """
    add_log(f"Starting simulated training for model {model_id}")
    update_training_progress(model_id, status="running", total_epochs=epochs)

    for epoch in range(1, epochs + 1):
        if stop_event.is_set():
            add_log(f"Simulated training for model {model_id} was manually stopped")
            update_training_progress(model_id, status="stopped")
            return

        # Simulate epoch time
        time.sleep(2)

        # Generate random loss that decreases over time
        loss = max(0.1, 2.0 - (epoch / epochs) * 1.5 + random.uniform(-0.1, 0.1))

        # Update progress
        update_training_progress(
            model_id, epoch=epoch, loss=loss, progress=(epoch / epochs) * 100
        )

        add_log(f"Epoch {epoch}/{epochs} completed. Loss: {loss:.4f}")

    # Training completed
    add_log(f"Simulated training completed for model {model_id}")
    update_training_progress(model_id, status="completed")

    # Create dummy model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Save to session state
    st.session_state.trained_models[model_id] = {
        "model": model,
        "tokenizer": tokenizer,
        "info": {
            "id": model_id,
            "base_model": base_model_name,
            "dataset": dataset_name,
            "created_at": timestamp(),
            "epochs": epochs,
            "simulated": True,
        },
    }


def simulate_training(model_id, dataset_name, base_model_name, epochs):
    """
    Start simulated training in a separate thread.

    Args:
        model_id: Identifier for the model
        dataset_name: Name of the dataset to use
        base_model_name: Base model from Hugging Face
        epochs: Number of training epochs

    Returns:
        threading.Event: Event to signal stopping the training
    """
    # Initialize training progress
    initialize_training_progress(model_id)

    # Create stop event
    stop_event = threading.Event()

    # Start training thread
    training_thread = threading.Thread(
        target=simulate_training_thread,
        args=(model_id, dataset_name, base_model_name, epochs, stop_event),
    )
    training_thread.start()

    return stop_event


def update_active_jobs_count():
    """Update the count of active training jobs."""
    with threading.Lock():
        active_count = sum(
            1
            for progress in st.session_state.get("training_progress", {}).values()
            if progress.get("status") == "running"
        )
        st.session_state.active_jobs_count = active_count


def start_model_training(*args, **kwargs):
    """Start model training with job counter update."""
    result = (
        simulate_training(*args, **kwargs)
        if not TRANSFORMERS_AVAILABLE
        else train_model_thread(*args, **kwargs)
    )
    update_active_jobs_count()
    return result


def stop_model_training(model_id, stop_event):
    """Stop model training."""
    if stop_event and not stop_event.is_set():
        add_log(f"Stopping training for model {model_id}")
        stop_event.set()
        update_active_jobs_count()
