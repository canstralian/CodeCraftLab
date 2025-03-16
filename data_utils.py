import io
import json
import os
import re
import ast
import hashlib
from datetime import datetime

# First import utilities that don't depend on pandas
from utils import add_log, format_code

# Then import pandas and streamlit
import streamlit as st
try:
    import pandas as pd
except ImportError:
    pd = None

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


def process_python_dataset(uploaded_file, dataset_name, preprocessing_options=None):
    """
    Process an uploaded Python dataset file.
    Supports .py, .json, and .csv formats.

    Args:
        uploaded_file: The uploaded file object
        dataset_name: Name to identify the dataset
        preprocessing_options: Dictionary of preprocessing options

    Returns:
        bool: Success status
    """
    try:
        # Default preprocessing options if none provided
        if preprocessing_options is None:
            preprocessing_options = {
                "clean_code": True,
                "extract_docstring": False,
                "extract_comments": False,
                "calculate_complexity": False,
                "deduplicate": False,
                "min_length": 10,
                "max_length": float('inf'),
                "filter_by_complexity": False,
                "min_complexity": 1,
                "max_complexity": float('inf')
            }
        
        file_extension = uploaded_file.name.split(".")[-1].lower()
        add_log(f"Processing {file_extension} file with name: {uploaded_file.name}")

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
            if pd is None:
                add_log("Pandas is required to read CSV files but not available", "ERROR")
                return False
            df = pd.read_csv(uploaded_file)
            dataset = Dataset.from_pandas(df)

        else:
            add_log(f"Unsupported file format: {file_extension}", "ERROR")
            return False
        
        # Validate dataset structure
        if not validate_dataset_structure(dataset):
            add_log("Dataset has invalid structure", "ERROR")
            return False

        # Split into train/validation sets
        test_size = preprocessing_options.get("validation_split", 0.2)
        train_test_split = dataset.train_test_split(test_size=test_size)

        # Create a DatasetDict
        dataset_dict = DatasetDict(
            {"train": train_test_split["train"], "validation": train_test_split["test"]}
        )
        
        # Apply preprocessing steps
        add_log(f"Applying preprocessing steps to dataset '{dataset_name}'")
        processed_dataset = preprocess_dataset(dataset_dict, preprocessing_options)
        
        # Get dataset statistics for info
        train_size = len(processed_dataset["train"]) if "train" in processed_dataset else 0
        validation_size = len(processed_dataset["validation"]) if "validation" in processed_dataset else 0
        total_size = train_size + validation_size

        # Store additional preprocessing metadata
        preprocessing_metadata = {}
        
        if preprocessing_options.get("deduplicate", False):
            preprocessing_metadata["deduplicated"] = True
        
        if preprocessing_options.get("clean_code", False):
            preprocessing_metadata["cleaned"] = True
            
        if preprocessing_options.get("extract_docstring", False):
            preprocessing_metadata["has_docstrings"] = True
            
        if preprocessing_options.get("extract_comments", False):
            preprocessing_metadata["has_comments"] = True
            
        if preprocessing_options.get("calculate_complexity", False):
            preprocessing_metadata["has_complexity"] = True
        
        # Store in session state
        st.session_state.datasets[dataset_name] = {
            "data": processed_dataset,
            "info": {
                "name": dataset_name,
                "size": total_size,
                "train_size": train_size,
                "validation_size": validation_size,
                "columns": processed_dataset["train"].column_names if train_size > 0 else [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if pd is None else pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "preprocessing": preprocessing_metadata
            },
        }

        add_log(
            f"Dataset '{dataset_name}' processed successfully with {total_size} examples"
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


# ===================== PREPROCESSING UTILITIES ===================== #

def clean_code(code_text):
    """
    Clean and normalize code by removing excessive whitespace and normalizing line endings.
    
    Args:
        code_text (str): Raw code text
        
    Returns:
        str: Cleaned code
    """
    if not code_text or not isinstance(code_text, str):
        return ""
    
    # Normalize line endings
    code_text = code_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in code_text.split('\n')]
    
    # Remove multiple blank lines
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line.strip():
            if not prev_empty:
                cleaned_lines.append(line)
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    
    return '\n'.join(cleaned_lines)


def extract_docstring(code_text):
    """
    Extract docstrings from Python code.
    
    Args:
        code_text (str): Python code text
        
    Returns:
        str: Extracted docstring or empty string if none found
    """
    try:
        # Parse the code text to an AST
        parsed = ast.parse(code_text)
        
        # Function to get docstring from AST node
        def get_docstring_from_node(node):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if ast.get_docstring(node):
                    return ast.get_docstring(node)
            
            for child in ast.iter_child_nodes(node):
                docstring = get_docstring_from_node(child)
                if docstring:
                    return docstring
            
            return ""
        
        return get_docstring_from_node(parsed)
    except SyntaxError:
        # If code has syntax errors, return empty string
        return ""


def extract_comments(code_text):
    """
    Extract comments from Python code.
    
    Args:
        code_text (str): Python code text
        
    Returns:
        list: List of extracted comments
    """
    if not code_text or not isinstance(code_text, str):
        return []
    
    comments = []
    lines = code_text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Find inline comments
        if '#' in line:
            comment_idx = line.find('#')
            # Ensure this is not within a string
            if not is_in_string(line, comment_idx):
                comments.append(line[comment_idx+1:].strip())
    
    return comments


def is_in_string(line, pos):
    """
    Check if a position in a line is inside a string.
    
    Args:
        line (str): Line of code
        pos (int): Position to check
        
    Returns:
        bool: True if position is within a string
    """
    string_delimiters = ['"', "'"]
    in_string = False
    current_delimiter = None
    
    for i, char in enumerate(line):
        if i >= pos:
            return in_string
        
        if char in string_delimiters and (i == 0 or line[i-1] != '\\'):
            if not in_string:
                in_string = True
                current_delimiter = char
            elif char == current_delimiter:
                in_string = False
                current_delimiter = None
    
    return False


def calculate_cyclomatic_complexity(code_text):
    """
    Calculate the cyclomatic complexity of Python code.
    
    Args:
        code_text (str): Python code text
        
    Returns:
        int: Cyclomatic complexity score
    """
    try:
        # Parse the code text to an AST
        parsed = ast.parse(code_text)
        
        # Count control flow statements
        complexity = 1  # Base complexity is 1
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1
            
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self.complexity += len(node.handlers) + len(node.finalbody)
                self.generic_visit(node)
            
            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_BoolOp(self, node):
                # Count boolean operators (and, or)
                self.complexity += len(node.values) - 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(parsed)
        
        return visitor.complexity
    except SyntaxError:
        # If code has syntax errors, return a high complexity value
        return 10


def get_code_hash(code_text):
    """
    Generate a hash for code text to identify duplicates.
    Normalizes the code first to avoid false negatives.
    
    Args:
        code_text (str): Code text to hash
        
    Returns:
        str: SHA256 hash of the normalized code
    """
    # Normalize code by removing comments and whitespace
    normalized_code = normalize_code_for_hashing(code_text)
    
    # Generate hash
    return hashlib.sha256(normalized_code.encode()).hexdigest()


def normalize_code_for_hashing(code_text):
    """
    Normalize code for hashing by removing comments, whitespace, and variable names.
    
    Args:
        code_text (str): Code text to normalize
        
    Returns:
        str: Normalized code
    """
    if not code_text or not isinstance(code_text, str):
        return ""
    
    try:
        # Remove comments and docstrings
        parsed = ast.parse(code_text)
        
        class NameNormalizer(ast.NodeTransformer):
            def __init__(self):
                self.var_counter = 0
                self.func_counter = 0
                self.class_counter = 0
                self.var_map = {}
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if node.id not in self.var_map:
                        self.var_map[node.id] = f"var_{self.var_counter}"
                        self.var_counter += 1
                    node.id = self.var_map[node.id]
                elif isinstance(node.ctx, ast.Load):
                    if node.id in self.var_map:
                        node.id = self.var_map[node.id]
                return node
            
            def visit_FunctionDef(self, node):
                self.func_counter += 1
                node.name = f"func_{self.func_counter}"
                self.generic_visit(node)
                return node
            
            def visit_ClassDef(self, node):
                self.class_counter += 1
                node.name = f"class_{self.class_counter}"
                self.generic_visit(node)
                return node
        
        normalized = NameNormalizer().visit(parsed)
        
        # Convert back to source code
        normalized_code = ast.unparse(normalized)
        
        # Remove all whitespace and convert to lowercase
        return re.sub(r'\s+', '', normalized_code).lower()
    except (SyntaxError, AttributeError):
        # If code has syntax errors or ast.unparse is not available, 
        # do a simpler normalization
        # Remove comments
        code_without_comments = re.sub(r'#.*$', '', code_text, flags=re.MULTILINE)
        # Remove docstrings
        code_without_docstrings = re.sub(r'""".*?"""', '', code_without_comments, flags=re.DOTALL)
        code_without_docstrings = re.sub(r"'''.*?'''", '', code_without_docstrings, flags=re.DOTALL)
        # Remove all whitespace
        return re.sub(r'\s+', '', code_without_docstrings).lower()


def preprocess_dataset(dataset, preprocessing_options):
    """
    Apply preprocessing steps to a dataset based on selected options.
    
    Args:
        dataset: The dataset to preprocess
        preprocessing_options (dict): Dictionary of preprocessing options
        
    Returns:
        Dataset: Preprocessed dataset
    """
    add_log(f"Starting dataset preprocessing with options: {preprocessing_options}")
    
    # Apply preprocessing to each split
    processed_dataset = {}
    total_examples = 0
    
    for split_name, split_data in dataset.items():
        examples = []
        hashes = set()  # For deduplication
        
        for example in split_data["data"]:
            code = example.get("code", "")
            
            # Skip if code is empty
            if not code.strip():
                continue
            
            # Apply cleaning if enabled
            if preprocessing_options.get("clean_code", False):
                code = clean_code(code)
            
            # Skip if minimum length not met
            if len(code) < preprocessing_options.get("min_length", 0):
                continue
                
            # Skip if maximum length exceeded
            if preprocessing_options.get("max_length", float('inf')) < len(code):
                continue
            
            # Skip if complexity filtering is enabled
            if preprocessing_options.get("filter_by_complexity", False):
                complexity = calculate_cyclomatic_complexity(code)
                min_complexity = preprocessing_options.get("min_complexity", 1)
                max_complexity = preprocessing_options.get("max_complexity", float('inf'))
                
                if complexity < min_complexity or complexity > max_complexity:
                    continue
            
            # Deduplicate if enabled
            if preprocessing_options.get("deduplicate", False):
                code_hash = get_code_hash(code)
                if code_hash in hashes:
                    continue
                hashes.add(code_hash)
            
            # Create processed example
            processed_example = {"code": code}
            
            # Extract docstring if enabled
            if preprocessing_options.get("extract_docstring", False):
                docstring = extract_docstring(code)
                if docstring:
                    processed_example["docstring"] = docstring
            
            # Extract comments if enabled
            if preprocessing_options.get("extract_comments", False):
                comments = extract_comments(code)
                if comments:
                    processed_example["comments"] = comments
            
            # Calculate complexity if enabled
            if preprocessing_options.get("calculate_complexity", False):
                processed_example["complexity"] = calculate_cyclomatic_complexity(code)
            
            examples.append(processed_example)
        
        # Create processed split
        processed_dataset[split_name] = Dataset.from_list(examples)
        total_examples += len(examples)
    
    # Create DatasetDict
    processed_dataset_dict = DatasetDict(processed_dataset)
    
    add_log(f"Dataset preprocessing completed. Total examples after preprocessing: {total_examples}")
    
    return processed_dataset_dict
