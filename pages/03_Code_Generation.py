import streamlit as st
import time
from model_utils import list_trained_models, generate_code, get_model_info
from utils import set_page_config, display_sidebar, add_log, format_code

# Set page configuration
set_page_config()

# Display sidebar
display_sidebar()

# Title
st.title("Code Generation")
st.markdown("Generate Python code using your trained models.")

# Get available models
available_models = list_trained_models()

if not available_models:
    st.warning("No trained models available. Please train a model in the Model Training section.")
else:
    # Create main columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Code Generation Setup")
        
        # Model selection
        selected_model = st.selectbox("Select Model", available_models)
        
        # Display model info if available
        if selected_model:
            model_info = get_model_info(selected_model)
            if model_info:
                st.markdown("#### Model Information")
                
                # Create expandable section for model details
                with st.expander("Model Details", expanded=False):
                    for key, value in model_info.items():
                        if key != 'id':  # Skip ID as it's already shown in the selectbox
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Generation parameters
        st.markdown("#### Generation Parameters")
        max_length = st.slider("Maximum Length", min_value=50, max_value=500, value=200, step=10)
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1,
                             help="Higher values make output more random, lower values more deterministic")
        top_p = st.slider("Top P (Nucleus Sampling)", min_value=0.1, max_value=1.0, value=0.9, step=0.05,
                       help="Controls diversity. 0.9 means consider tokens comprising the top 90% probability mass")
        
        # Input prompt
        st.markdown("#### Input Prompt")
        prompt = st.text_area(
            "Enter your code prompt",
            height=200,
            placeholder="# Function to calculate fibonacci sequence\ndef fibonacci(n):"
        )
        
        # Generate button
        generate_button = st.button("Generate Code", disabled=not prompt)
    
    with col2:
        st.markdown("### Generated Code")
        
        # Create a placeholder for generated code
        code_placeholder = st.empty()
        
        # Initialize session state for code history if not exists
        if 'code_history' not in st.session_state:
            st.session_state.code_history = []
        
        # Generate code when button is clicked
        if generate_button and prompt and selected_model:
            with st.spinner("Generating code..."):
                generated_code = generate_code(
                    selected_model, 
                    prompt, 
                    max_length=max_length, 
                    temperature=temperature, 
                    top_p=top_p
                )
                
                # Add to history
                st.session_state.code_history.append({
                    'prompt': prompt,
                    'code': generated_code,
                    'model': selected_model,
                    'parameters': {
                        'max_length': max_length,
                        'temperature': temperature,
                        'top_p': top_p
                    },
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Display the generated code
                code_placeholder.code(format_code(generated_code), language='python')
                
                # Log the generation
                add_log(f"Generated code with model '{selected_model}' (length: {len(generated_code)})")
        
        # If there's code history but the generate button wasn't pressed, show the most recent one
        elif st.session_state.code_history:
            last_code = st.session_state.code_history[-1]['code']
            code_placeholder.code(format_code(last_code), language='python')
        else:
            # Show empty placeholder when no code has been generated
            code_placeholder.code("# Generated code will appear here", language='python')
    
    # Code history section
    st.markdown("---")
    st.markdown("### Code Generation History")
    
    if not st.session_state.code_history:
        st.info("No code has been generated yet. Use the form above to generate code.")
    else:
        # Display code history
        for i, item in enumerate(reversed(st.session_state.code_history)):
            with st.expander(f"Generation {len(st.session_state.code_history) - i}: {item['timestamp']}"):
                st.markdown(f"**Model:** {item['model']}")
                st.markdown(f"**Parameters:** Max Length: {item['parameters']['max_length']}, "
                          f"Temperature: {item['parameters']['temperature']}, "
                          f"Top P: {item['parameters']['top_p']}")
                
                st.markdown("**Prompt:**")
                st.code(format_code(item['prompt']), language='python')
                
                st.markdown("**Generated Code:**")
                st.code(format_code(item['code']), language='python')
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.code_history = []
            st.success("History cleared!")
            time.sleep(1)
            st.rerun()
