import io
import json
import os
import re
import ast
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

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

# Import utils after other imports to avoid circular dependency
from utils import add_log, format_code

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
                # Code quality options
                "clean_code": True,
                "deduplicate": False,
                "min_length": 10,
                "max_length": float('inf'),
                
                # Feature extraction
                "extract_docstring": False,
                "extract_comments": False,
                "calculate_complexity": False,
                "extract_names": False,
                "analyze_imports": False,
                "calculate_statistics": False,
                
                # Code style and standardization
                "analyze_style": False,
                "standardize_code": False,
                "use_standardized": False,
                
                # Complexity filtering
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
        
        # Code quality metadata
        if preprocessing_options.get("deduplicate", False):
            preprocessing_metadata["deduplicated"] = True
        
        if preprocessing_options.get("clean_code", False):
            preprocessing_metadata["cleaned"] = True
            
        # Feature extraction metadata
        if preprocessing_options.get("extract_docstring", False):
            preprocessing_metadata["has_docstrings"] = True
            
        if preprocessing_options.get("extract_comments", False):
            preprocessing_metadata["has_comments"] = True
            
        if preprocessing_options.get("calculate_complexity", False):
            preprocessing_metadata["has_complexity"] = True
            
        if preprocessing_options.get("extract_names", False):
            preprocessing_metadata["has_function_class_names"] = True
            
        if preprocessing_options.get("analyze_imports", False):
            preprocessing_metadata["has_import_analysis"] = True
            
        if preprocessing_options.get("calculate_statistics", False):
            preprocessing_metadata["has_code_statistics"] = True
            
        # Code style metadata
        if preprocessing_options.get("analyze_style", False):
            preprocessing_metadata["has_style_analysis"] = True
            
        if preprocessing_options.get("standardize_code", False):
            preprocessing_metadata["has_standardized_code"] = True
            
        if preprocessing_options.get("use_standardized", False):
            preprocessing_metadata["uses_standardized_code"] = True
        
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


def extract_function_and_class_names(code_text):
    """
    Extract function and class names from Python code.
    
    Args:
        code_text (str): Python code text
        
    Returns:
        tuple: (list of function names, list of class names)
    """
    try:
        parsed = ast.parse(code_text)
        function_names = []
        class_names = []
        
        for node in ast.walk(parsed):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                class_names.append(node.name)
                
        return function_names, class_names
    except SyntaxError:
        return [], []


def analyze_imports(code_text):
    """
    Extract and analyze import statements from Python code.
    
    Args:
        code_text (str): Python code text
        
    Returns:
        dict: Information about imports
    """
    try:
        parsed = ast.parse(code_text)
        imports = {
            "standard_lib": [],
            "third_party": [],
            "local": [],
            "total_count": 0
        }
        
        # List of standard library modules
        std_libs = [
            "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib", 
            "copy", "csv", "datetime", "enum", "functools", "glob", "hashlib", "html", 
            "http", "importlib", "inspect", "io", "itertools", "json", "logging", "math", 
            "multiprocessing", "os", "pathlib", "pickle", "random", "re", "shutil", 
            "socket", "sqlite3", "string", "subprocess", "sys", "tempfile", "threading", 
            "time", "typing", "unittest", "urllib", "uuid", "wave", "weakref", "xml", 
            "zipfile"
        ]
        
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports["total_count"] += 1
                    module_name = name.name.split('.')[0]
                    if module_name in std_libs:
                        imports["standard_lib"].append(name.name)
                    elif module_name.startswith('.'):
                        imports["local"].append(name.name)
                    else:
                        imports["third_party"].append(name.name)
                        
            elif isinstance(node, ast.ImportFrom):
                imports["total_count"] += 1
                module_name = node.module.split('.')[0] if node.module else ""
                if module_name in std_libs:
                    imports["standard_lib"].append(node.module)
                elif module_name.startswith('.') or node.level > 0:
                    imports["local"].append(node.module or f".{'.' * (node.level-1)}")
                else:
                    imports["third_party"].append(node.module)
                
        return imports
    except (SyntaxError, AttributeError):
        return {"standard_lib": [], "third_party": [], "local": [], "total_count": 0}


def calculate_code_statistics(code_text):
    """
    Calculate various code statistics and metrics.
    
    Args:
        code_text (str): Python code text
        
    Returns:
        dict: Dictionary of code statistics
    """
    if not code_text or not isinstance(code_text, str):
        return {}
    
    lines = code_text.split('\n')
    
    # Count lines by type
    blank_lines = sum(1 for line in lines if not line.strip())
    comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
    code_lines = len(lines) - blank_lines - comment_lines
    
    # Count characters and tokens
    chars = len(code_text)
    words = len(re.findall(r'\b\w+\b', code_text))
    
    # Calculate character density (characters per line of actual code)
    char_density = chars / max(code_lines, 1)
    
    return {
        "total_lines": len(lines),
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "chars": chars,
        "tokens": words,
        "char_density": round(char_density, 2),
        "comment_ratio": round(comment_lines / max(len(lines), 1) * 100, 2)
    }


def analyze_code_style(code_text):
    """
    Check for basic PEP 8 compliance and code style issues.
    
    Args:
        code_text (str): Python code text
        
    Returns:
        dict: Dictionary with style issues
    """
    if not code_text or not isinstance(code_text, str):
        return {"issues": [], "compliant": True}
    
    issues = []
    
    # Check line length
    for i, line in enumerate(code_text.split('\n')):
        if len(line) > 79:
            issues.append(f"Line {i+1} exceeds 79 characters (PEP 8)")
    
    # Check indentation (expecting 4 spaces)
    indent_pattern = re.compile(r'^(\s+)[^\s]')
    for i, line in enumerate(code_text.split('\n')):
        match = indent_pattern.match(line)
        if match:
            indent = match.group(1)
            if '\t' in indent:
                issues.append(f"Line {i+1} uses tabs instead of spaces (PEP 8)")
            elif len(indent) % 4 != 0:
                issues.append(f"Line {i+1} has inconsistent indentation (PEP 8)")
    
    # Check variable naming (snake_case for variables and functions)
    try:
        parsed = ast.parse(code_text)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.id) and not node.id.startswith('_'):
                    if not node.id.isupper():  # Allow uppercase for constants
                        issues.append(f"Variable '{node.id}' does not follow snake_case convention (PEP 8)")
            
            elif isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name) and not node.name.startswith('_'):
                    issues.append(f"Function '{node.name}' does not follow snake_case convention (PEP 8)")
            
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append(f"Class '{node.name}' does not follow PascalCase convention (PEP 8)")
    except SyntaxError:
        issues.append("Syntax error prevented full style analysis")
    
    return {
        "issues": issues[:10],  # Limit to top 10 issues
        "total_issues": len(issues),
        "compliant": len(issues) == 0
    }


def standardize_code(code_text):
    """
    Standardize code by cleaning and applying consistent formatting.
    Does not require external dependencies.
    
    Args:
        code_text (str): Python code text
        
    Returns:
        str: Standardized code
    """
    if not code_text or not isinstance(code_text, str):
        return ""
    
    # Normalize line endings
    code_text = code_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split into lines and process
    lines = code_text.split('\n')
    processed_lines = []
    
    in_multiline_string = False
    string_delimiter = None
    
    for line in lines:
        # Skip processing if we're in a multiline string
        if in_multiline_string:
            processed_lines.append(line)
            
            # Check if multiline string ends
            if line.rstrip().endswith(string_delimiter * 3):
                in_multiline_string = False
                string_delimiter = None
            continue
        
        # Check for multiline string start
        if '"""' in line or "'''" in line:
            # Find which delimiter is used
            if '"""' in line:
                string_delimiter = '"'
                # Check if it doesn't end on the same line
                if line.count('"""') % 2 != 0:
                    in_multiline_string = True
            elif "'''" in line:
                string_delimiter = "'"
                # Check if it doesn't end on the same line
                if line.count("'''") % 2 != 0:
                    in_multiline_string = True
        
        # Process the line if not in multiline string
        original_line = line
        
        # Standardize indentation (convert tabs to 4 spaces)
        if line.startswith('\t'):
            line = line.replace('\t', '    ')
        
        # Remove trailing whitespace
        line = line.rstrip()
        
        # Add space after commas if missing
        line = re.sub(r',([^\s])', r', \1', line)
        
        # Add space around operators
        for op in ['=', '+', '-', '*', '/', '%', '==', '!=', '<=', '>=', '<', '>', '+=', '-=', '*=', '/=']:
            if op in line:
                # Skip if within quotes or comments
                parts = line.split('#', 1)
                code_part = parts[0]
                
                # Simple heuristic to not adjust operators in strings
                in_string = False
                string_char = None
                result = []
                
                for char in code_part:
                    if char in ['"', "'"]:
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                            string_char = None
                    result.append(char)
                
                # If the processing gets too complex, revert to original
                if in_string:
                    line = original_line
                    break
                
                # Only replace operators in the code part, not in strings or comments
                if not in_string:
                    # Make sure we don't add spaces in string parts
                    code_part = re.sub(f'([^\\s{op[0]}])({re.escape(op)})([^\\s{op[-1]}])', r'\1 \2 \3', code_part)
                    
                    if len(parts) > 1:
                        line = code_part + '#' + parts[1]
                    else:
                        line = code_part
        
        processed_lines.append(line)
    
    # Process the entire file
    standardized_code = '\n'.join(processed_lines)
    
    # Remove multiple consecutive blank lines
    standardized_code = re.sub(r'\n{3,}', '\n\n', standardized_code)
    
    # Ensure file ends with a single newline
    if standardized_code and not standardized_code.endswith('\n'):
        standardized_code += '\n'
    
    return standardized_code


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
            
            # Extract function and class names if enabled
            if preprocessing_options.get("extract_names", False):
                function_names, class_names = extract_function_and_class_names(code)
                if function_names:
                    processed_example["function_names"] = function_names
                if class_names:
                    processed_example["class_names"] = class_names
            
            # Analyze imports if enabled
            if preprocessing_options.get("analyze_imports", False):
                imports = analyze_imports(code)
                if imports["total_count"] > 0:
                    processed_example["imports"] = imports
            
            # Calculate code statistics if enabled
            if preprocessing_options.get("calculate_statistics", False):
                stats = calculate_code_statistics(code)
                if stats:
                    processed_example["statistics"] = stats
            
            # Analyze code style if enabled
            if preprocessing_options.get("analyze_style", False):
                style = analyze_code_style(code)
                if style["issues"]:
                    processed_example["style_issues"] = style
            
            # Standardize code if enabled
            if preprocessing_options.get("standardize_code", False):
                standardized_code = standardize_code(code)
                processed_example["standardized_code"] = standardized_code
                
                # Use standardized code for further processing if requested
                if preprocessing_options.get("use_standardized", False):
                    processed_example["original_code"] = code
                    processed_example["code"] = standardized_code
            
            examples.append(processed_example)
        
        # Create processed split
        processed_dataset[split_name] = Dataset.from_list(examples)
        total_examples += len(examples)
    
    # Create DatasetDict
    processed_dataset_dict = DatasetDict(processed_dataset)
    
    add_log(f"Dataset preprocessing completed. Total examples after preprocessing: {total_examples}")
    
    return processed_dataset_dict