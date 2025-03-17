
import requests
import os
import json
from typing import List, Dict, Any
import pandas as pd

class DatasetManager:
    """Handles dataset downloading and preprocessing for code generation."""
    
    def __init__(self, cache_dir: str = "datasets"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_github_dataset(self, owner: str, repo: str, path: str) -> str:
        """Downloads a dataset from a GitHub repository."""
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
        local_path = os.path.join(self.cache_dir, f"{owner}_{repo}_{path.replace('/', '_')}")
        
        if not os.path.exists(local_path):
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
        
        return local_path

    def preprocess_code(self, code: str) -> str:
        """Preprocesses code by removing comments and normalizing whitespace."""
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            # Remove inline comments
            if '#' in line:
                line = line.split('#')[0]
            # Skip empty lines
            if line.strip():
                clean_lines.append(line.rstrip())
        return '\n'.join(clean_lines)

    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Loads and preprocesses a code dataset."""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format")
            
        if 'code' in df.columns:
            df['processed_code'] = df['code'].apply(self.preprocess_code)
        
        return df

    def deduplicate_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate code samples."""
        return df.drop_duplicates(subset=['processed_code'])
