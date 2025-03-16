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
        
    # Preprocessing settings section with expandable container
    with st.expander("Advanced Preprocessing Options", expanded=False):
        st.markdown("### Code Quality")
        col1, col2 = st.columns(2)
        with col1:
            clean_code_option = st.checkbox("Clean and normalize code", value=True, 
                                help="Remove excessive whitespace and normalize line endings")
            deduplicate = st.checkbox("Remove duplicate code samples", value=False,
                                help="Detect and remove duplicate code snippets")
            
        with col2:
            min_length = st.number_input("Minimum code length (chars)", min_value=0, value=10,
                                help="Filter out code samples below this length")
            max_length = st.number_input("Maximum code length (chars)", min_value=0, value=10000,
                                help="Filter out code samples above this length (0 for no limit)")
    
        st.markdown("### Feature Extraction")
        col1, col2 = st.columns(2)
        with col1:
            extract_docstrings = st.checkbox("Extract docstrings", value=False,
                                    help="Extract docstrings from Python code")
            extract_comments = st.checkbox("Extract comments", value=False,
                                help="Extract comments from Python code")
            calculate_complexity = st.checkbox("Calculate code complexity", value=False,
                                    help="Compute cyclomatic complexity for each code sample")
            
        with col2:
            extract_names = st.checkbox("Extract function/class names", value=False,
                                help="Extract function and class names from code")
            analyze_imports = st.checkbox("Analyze imports", value=False,
                                help="Extract and analyze import statements")
            calculate_statistics = st.checkbox("Calculate code statistics", value=False,
                                    help="Calculate various code statistics (lines, tokens, etc)")
        
        st.markdown("### Code Style & Standardization")
        col1, col2 = st.columns(2)
        with col1:
            analyze_style = st.checkbox("Analyze code style", value=False,
                                help="Check code for PEP 8 compliance and style issues")
            filter_by_complexity = st.checkbox("Filter by complexity", value=False,
                                    help="Filter code samples based on complexity")
            
        with col2:
            standardize_code = st.checkbox("Standardize code", value=False,
                                help="Apply code standardization (spacing, indentation, etc)")
            use_standardized = st.checkbox("Use standardized code", value=False,
                                help="Use standardized code for processing (requires 'Standardize code')")
            if standardize_code and not use_standardized:
                st.info("The original code will be preserved, standardized version will be added as 'standardized_code'.")
            
        if filter_by_complexity:
            col1, col2 = st.columns(2)
            with col1:
                min_complexity = st.number_input("Minimum complexity", min_value=1, value=1,
                                        help="Filter out code below this complexity")
            with col2:
                max_complexity = st.number_input("Maximum complexity", min_value=1, value=20,
                                        help="Filter out code above this complexity")

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
            # Collect preprocessing options
            preprocessing_options = {
                # Dataset splitting
                "validation_split": split_ratio if auto_split else 0.2,
                
                # Code quality options
                "clean_code": clean_code_option,
                "deduplicate": deduplicate,
                "min_length": min_length,
                "max_length": max_length if max_length > 0 else float('inf'),
                
                # Feature extraction
                "extract_docstring": extract_docstrings,
                "extract_comments": extract_comments,
                "calculate_complexity": calculate_complexity,
                "extract_names": extract_names,
                "analyze_imports": analyze_imports,
                "calculate_statistics": calculate_statistics,
                
                # Code style and standardization
                "analyze_style": analyze_style,
                "standardize_code": standardize_code,
                "use_standardized": use_standardized if standardize_code else False,
                
                # Complexity filtering
                "filter_by_complexity": filter_by_complexity,
                "min_complexity": min_complexity if filter_by_complexity else 1,
                "max_complexity": max_complexity if filter_by_complexity else float('inf')
            }
            
            with st.spinner("Processing dataset..."):
                add_log(f"Processing dataset with options: {preprocessing_options}")
                success = process_python_dataset(uploaded_file, dataset_name, preprocessing_options)
                if success:
                    st.success(f"Dataset '{dataset_name}' processed successfully!")
                    add_log(f"Dataset '{dataset_name}' uploaded and processed with preprocessing")
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
                    
                    # Display preprocessing metadata if available
                    preprocessing = dataset_info.get("preprocessing", {})
                    if preprocessing:
                        st.markdown("### Preprocessing Info")
                        
                        # Code quality metrics
                        if preprocessing.get("cleaned"):
                            st.markdown("✅ Code cleaning applied")
                        if preprocessing.get("deduplicated"):
                            st.markdown("✅ Duplicate removal applied")
                            
                        # Feature extraction
                        if preprocessing.get("has_docstrings"):
                            st.markdown("✅ Docstrings extracted")
                        if preprocessing.get("has_comments"):
                            st.markdown("✅ Comments extracted")
                        if preprocessing.get("has_complexity"):
                            st.markdown("✅ Complexity calculated")
                        if preprocessing.get("has_function_class_names"):
                            st.markdown("✅ Function/class names extracted")
                        if preprocessing.get("has_import_analysis"):
                            st.markdown("✅ Import statements analyzed")
                        if preprocessing.get("has_code_statistics"):
                            st.markdown("✅ Code statistics calculated")
                            
                        # Code style
                        if preprocessing.get("has_style_analysis"):
                            st.markdown("✅ Code style analyzed")
                        if preprocessing.get("has_standardized_code"):
                            st.markdown("✅ Code standardization applied")
                        if preprocessing.get("uses_standardized_code"):
                            st.markdown("✅ Using standardized code for processing")

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
                            # Display code
                            st.code(
                                dataset["train"][i].get("code", "# No code available"),
                                language="python",
                            )
                            
                            # Display additional features in columns if they exist
                            example = dataset["train"][i]
                            
                            # Check if we have additional extracted features
                            has_extra = any(k in example for k in ["docstring", "comments", "complexity", 
                                                                  "function_names", "class_names", "imports", 
                                                                  "statistics", "style_issues", "standardized_code"])
                            
                            if has_extra:
                                st.markdown("### Additional Features")
                                
                                # Basic code features
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if "docstring" in example:
                                        with st.expander("Docstring"):
                                            st.markdown(f"```\n{example['docstring']}\n```")
                                    
                                    if "comments" in example:
                                        with st.expander("Comments"):
                                            for comment in example["comments"]:
                                                st.markdown(f"- `{comment}`")
                                
                                with col2:
                                    if "complexity" in example:
                                        st.metric("Cyclomatic Complexity", example["complexity"])
                                    
                                    # Structure metrics
                                    if "function_names" in example and example["function_names"]:
                                        with st.expander("Functions"):
                                            for func in example["function_names"]:
                                                st.markdown(f"- `{func}`")
                                    
                                    if "class_names" in example and example["class_names"]:
                                        with st.expander("Classes"):
                                            for cls in example["class_names"]:
                                                st.markdown(f"- `{cls}`")
                                
                                # Code analysis
                                if "imports" in example:
                                    with st.expander("Import Analysis"):
                                        imports = example["imports"]
                                        st.markdown(f"**Total imports:** {imports.get('total_count', 0)}")
                                        
                                        if imports.get("standard_lib"):
                                            st.markdown("**Standard library imports:**")
                                            for imp in imports.get("standard_lib", []):
                                                st.markdown(f"- `{imp}`")
                                        
                                        if imports.get("third_party"):
                                            st.markdown("**Third-party imports:**")
                                            for imp in imports.get("third_party", []):
                                                st.markdown(f"- `{imp}`")
                                        
                                        if imports.get("local"):
                                            st.markdown("**Local imports:**")
                                            for imp in imports.get("local", []):
                                                st.markdown(f"- `{imp}`")
                                
                                # Code statistics
                                if "statistics" in example:
                                    with st.expander("Code Statistics"):
                                        stats = example["statistics"]
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Lines", stats.get("total_lines", 0))
                                        with col2:
                                            st.metric("Tokens", stats.get("total_tokens", 0))
                                        with col3:
                                            st.metric("AST Nodes", stats.get("ast_nodes", 0))
                                
                                # Code style
                                if "style_issues" in example:
                                    with st.expander("Style Analysis"):
                                        style = example["style_issues"]
                                        st.markdown(f"**PEP 8 compliance score:** {style.get('pep8_score', 0)}/10")
                                        
                                        if style.get("issues"):
                                            st.markdown("**Style issues:**")
                                            for issue in style.get("issues", []):
                                                st.markdown(f"- {issue}")
                                
                                # Standardized code
                                if "standardized_code" in example:
                                    with st.expander("Standardized Code"):
                                        st.code(example["standardized_code"], language="python")
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
