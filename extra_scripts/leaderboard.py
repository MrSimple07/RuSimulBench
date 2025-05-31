import pandas as pd
import json
from datetime import datetime
import os
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Leaderboard:
    def __init__(self, file_path='leaderboard.json'):
        self.file_path = file_path
        self.results = self._load_results()

    def _load_results(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_results(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    def add_batch_results(self, model_name: str, results: list):
        """
        Add a batch of results to the leaderboard
        results should be a list of dictionaries containing:
        - baseline_prompt
        - variations
        - stability_score
        - creativity_scores
        - combined_score
        """
        print(f"\nAdding {len(results)} results to leaderboard for model: {model_name}")
        
        # Calculate average scores for the batch
        avg_stability = float(np.mean([r['stability_score'] for r in results]))
        avg_creativity = float(np.mean([r['creativity_score'] for r in results]))
        avg_combined = float(np.mean([r['combined_score'] for r in results]))
        
        # Create a summary result
        summary_result = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'num_tasks': len(results),
            'avg_stability_score': avg_stability,
            'avg_creativity_score': avg_creativity,
            'avg_combined_score': avg_combined,
            'detailed_results': results
        }
        
        self.results.append(summary_result)
        self._save_results()
        print("Results added to leaderboard successfully")
        return summary_result

    def get_top_results(self, n=10):
        """Get top n results sorted by combined score"""
        if not self.results:
            return []
        
        sorted_results = sorted(self.results, 
                              key=lambda x: x['avg_combined_score'], 
                              reverse=True)
        return sorted_results[:n]

    def get_results_df(self):
        """Get all results as a pandas DataFrame"""
        if not self.results:
            return pd.DataFrame()
            
        return pd.DataFrame(self.results) 