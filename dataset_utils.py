
import requests
import os
import json
from typing import List, Dict, Any
import pandas as pd

class DatasetManager:
    """Handles dataset downloading and preprocessing for code generation."""
    
    KNOWN_REPOSITORIES = {
        "python-code-500k": {
            "owner": "Jtatman",
            "repo": "python-code-dataset-500k",
            "path": "dataset/code_samples.json"
        },
        "codeforces-python": {
            "owner": "MatrixStudio",
            "repo": "Codeforces-Python-Submissions",
            "path": "submissions.json"
        },
        "python-reasoning": {
            "owner": "sdiazlor",
            "repo": "python-reasoning-dataset",
            "path": "data/samples.json"
        },
        "github-code": {
            "owner": "angie-chen55",
            "repo": "python-github-code",
            "path": "data/code_samples.json"
        }
    }
    
    def __init__(self, cache_dir: str = "datasets"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_available_repositories(self) -> List[str]:
        """Returns list of available repository names."""
        return list(self.KNOWN_REPOSITORIES.keys())
        
    def download_repository(self, repo_name: str) -> pd.DataFrame:
        """Downloads and processes a known repository."""
        if repo_name not in self.KNOWN_REPOSITORIES:
            raise ValueError(f"Unknown repository: {repo_name}")
            
        repo_info = self.KNOWN_REPOSITORIES[repo_name]
        local_path = self.download_github_dataset(
            repo_info["owner"],
            repo_info["repo"],
            repo_info["path"]
        )
        return self.load_dataset(local_path)
        
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
