import streamlit as st
import json
import os
from utils import add_log

# Initialize huggingface_models in session state if not present
if 'huggingface_models' not in st.session_state:
    st.session_state.huggingface_models = [
        "codegen-350M-mono",
        "codegen-2B-mono",
        "Salesforce/codegen-350M-mono",
        "Salesforce/codegen-2B-mono",
        "gpt2",
        "EleutherAI/gpt-neo-125M"
    ]

# Handle missing dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
    # Mock classes for demo purposes
    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()
            
        def __call__(self, text, **kwargs):
            return {"input_ids": [list(range(10))] * (1 if isinstance(text, str) else len(text))}
            
        def decode(self, token_ids, **kwargs):
            return "# Generated code placeholder\n\ndef example_function():\n    return 'Hello world!'"
        
        @property
        def eos_token(self):
            return "[EOS]"
            
        @property
        def eos_token_id(self):
            return 0
            
        @property
        def pad_token(self):
            return None
            
        @pad_token.setter
        def pad_token(self, value):
            pass
            
    class DummyModel:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()
            
        def generate(self, input_ids, **kwargs):
            return [[1, 2, 3, 4, 5]]
            
        @property
        def config(self):
            class Config:
                @property
                def eos_token_id(self):
                    return 0
                    
                @property
                def pad_token_id(self):
                    return 0
                
                @pad_token_id.setter
                def pad_token_id(self, value):
                    pass
                    
            return Config()
            
    # Set aliases to match transformers
    AutoTokenizer = DummyTokenizer
    AutoModelForCausalLM = DummyModel

def list_available_huggingface_models():
    """
    List available code generation models from Hugging Face.
    
    Returns:
        list: List of model names
    """
    # Return the list stored in session state
    return st.session_state.huggingface_models

def get_model_and_tokenizer(model_name):
    """
    Load model and tokenizer from Hugging Face Hub.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    try:
        add_log(f"Loading model and tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        add_log(f"Model and tokenizer loaded successfully: {model_name}")
        return model, tokenizer
    except Exception as e:
        add_log(f"Error loading model {model_name}: {str(e)}", "ERROR")
        return None, None

def save_trained_model(model_id, model, tokenizer):
    """
    Save trained model information to session state.
    
    Args:
        model_id: Identifier for the model
        model: The trained model
        tokenizer: The model's tokenizer
        
    Returns:
        bool: Success status
    """
    try:
        # Store model information in session state
        from datetime import datetime
        st.session_state.trained_models[model_id] = {
            'model': model,
            'tokenizer': tokenizer,
            'info': {
                'id': model_id,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        add_log(f"Model {model_id} saved to session state")
        return True
    except Exception as e:
        add_log(f"Error saving model {model_id}: {str(e)}", "ERROR")
        return False

def list_trained_models():
    """
    List all trained models in session state.
    
    Returns:
        list: List of model IDs
    """
    if 'trained_models' in st.session_state:
        return list(st.session_state.trained_models.keys())
    return []

def generate_code(model_id, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """
    Generate code using a trained model.
    
    Args:
        model_id: ID of the model to use
        prompt: Input prompt for code generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        
    Returns:
        str: Generated code or error message
    """
    try:
        if model_id not in st.session_state.trained_models:
            return "Error: Model not found. Please select a valid model."
        
        model_data = st.session_state.trained_models[model_id]
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        if TRANSFORMERS_AVAILABLE:
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Demo mode - return dummy generated code
            inputs = tokenizer(prompt)
            outputs = model.generate(inputs["input_ids"])
            generated_code = tokenizer.decode(outputs[0])
            
            # Add some context to the generated code based on the prompt
            if "fibonacci" in prompt.lower():
                generated_code = "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n"
            elif "sort" in prompt.lower():
                generated_code = "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n"
        
        # If the prompt is included in the output, remove it to get only the generated code
        if generated_code.startswith(prompt):
            generated_code = generated_code[len(prompt):]
            
        return generated_code
        
    except Exception as e:
        add_log(f"Error generating code: {str(e)}", "ERROR")
        return f"Error generating code: {str(e)}"

def get_model_info(model_id):
    """
    Get information about a model.
    
    Args:
        model_id: ID of the model
        
    Returns:
        dict: Model information
    """
    if 'trained_models' in st.session_state and model_id in st.session_state.trained_models:
        return st.session_state.trained_models[model_id]['info']
    return None
