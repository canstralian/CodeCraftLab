
import unittest
from dataset_utils import DatasetManager
import pandas as pd

class TestDatasetManager(unittest.TestCase):
    def setUp(self):
        self.manager = DatasetManager(cache_dir="test_datasets")
        
    def test_preprocess_code(self):
        code = """def hello(): # Says hello
            # This is a comment
            print("Hello") # Print greeting"""
        expected = """def hello():
            print("Hello")"""
        processed = self.manager.preprocess_code(code)
        self.assertEqual(processed.strip(), expected.strip())
        
    def test_deduplicate_samples(self):
        data = {
            'code': ['print("a")', 'print("a")', 'print("b")'],
            'processed_code': ['print("a")', 'print("a")', 'print("b")']
        }
        df = pd.DataFrame(data)
        deduped = self.manager.deduplicate_samples(df)
        self.assertEqual(len(deduped), 2)

if __name__ == '__main__':
    unittest.main()
