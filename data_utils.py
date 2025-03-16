import io
import json
import os
import pandas as pd
import streamlit as st
from utils import add_log

# Handle missing dependencies
try:
    from datasets import Dataset, DatasetDict
except ImportError:
    # Create dummy classes for Dataset and DatasetDict
    class Dataset:
        @classmethod
        def from_list(cls, items):
            return {
                "data": items,
                "column_names": list(items[0].keys()) if items else [],
            }

        @classmethod
        def from_dict(cls, dict_obj):
            return {"data": dict_obj, "column_names": list(dict_obj.keys())}

        @classmethod
        def from_pandas(cls, df):
            return {"data": df, "column_names": df.columns.tolist()}

        def train_test_split(self, test_size=0.2):
            return {"train": self, "test": self}

    class DatasetDict(dict):
        pass


def process_python_dataset(uploaded_file, dataset_name):
    """
    Process an uploaded Python dataset file.
    Supports .py, .json, and .csv formats.

    Args:
        uploaded_file: The uploaded file object
        dataset_name: Name to identify the dataset

    Returns:
        bool: Success status
    """
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "py":
            # Process Python file
            content = uploaded_file.read().decode("utf-8")
            # Split by function or class definitions for separate examples
            examples = split_python_file(content)
            dataset = create_dataset_from_examples(examples)

        elif file_extension == "json":
            # Process JSON file
            content = json.loads(uploaded_file.read().decode("utf-8"))
            if isinstance(content, list):
                dataset = Dataset.from_list(content)
            else:
                dataset = Dataset.from_dict(content)

        elif file_extension == "csv":
            # Process CSV file
            df = pd.read_csv(uploaded_file)
            dataset = Dataset.from_pandas(df)

        else:
            add_log(f"Unsupported file format: {file_extension}", "ERROR")
            return False

        # Split into train/validation sets
        train_test_split = dataset.train_test_split(test_size=0.2)

        # Create a DatasetDict
        dataset_dict = DatasetDict(
            {"train": train_test_split["train"], "validation": train_test_split["test"]}
        )

        # Store in session state
        st.session_state.datasets[dataset_name] = {
            "data": dataset_dict,
            "info": {
                "name": dataset_name,
                "size": len(dataset),
                "train_size": len(train_test_split["train"]),
                "validation_size": len(train_test_split["test"]),
                "columns": dataset.column_names,
                "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        add_log(
            f"Dataset '{dataset_name}' processed successfully with {len(dataset)} examples"
        )
        return True

    except Exception as e:
        add_log(f"Error processing dataset: {str(e)}", "ERROR")
        return False


def split_python_file(content):
    """
    Split a Python file content into separate code examples.

    Args:
        content: String content of Python file

    Returns:
        list: List of code examples
    """
    examples = []

    # Simple splitting by function or class definitions
    lines = content.split("\n")
    current_example = []

    for line in lines:
        if (line.startswith("def ") or line.startswith("class ")) and current_example:
            # Start of a new function/class, save the previous one
            examples.append("\n".join(current_example))
            current_example = [line]
        else:
            current_example.append(line)

    # Add the last example
    if current_example:
        examples.append("\n".join(current_example))

    # If no examples were extracted, use the whole file as one example
    if not examples:
        examples = [content]

    return [{"code": example} for example in examples]


def create_dataset_from_examples(examples):
    """
    Create a dataset from code examples.

    Args:
        examples: List of code examples

    Returns:
        Dataset: Hugging Face dataset
    """
    return Dataset.from_list(examples)


def validate_dataset_structure(dataset):
    """
    Validate that the dataset has the required structure for training.

    Args:
        dataset: Hugging Face dataset

    Returns:
        bool: True if valid, False otherwise
    """
    if "code" not in dataset.column_names:
        add_log("Dataset missing 'code' column required for training", "ERROR")
        return False
    return True


def list_available_datasets():
    """
    List all available datasets in session state.

    Returns:
        list: List of dataset names
    """
    if "datasets" in st.session_state:
        return list(st.session_state.datasets.keys())
    return []


def get_dataset_info(dataset_name):
    """
    Get information about a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        dict: Dataset information
    """
    if "datasets" in st.session_state and dataset_name in st.session_state.datasets:
        return st.session_state.datasets[dataset_name]["info"]
    return None


def get_dataset(dataset_name):
    """
    Get a dataset by name.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dataset: The dataset object
    """
    if "datasets" in st.session_state and dataset_name in st.session_state.datasets:
        return st.session_state.datasets[dataset_name]["data"]
    return None
