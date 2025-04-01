import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from typing import List, Dict, Tuple

class EloRater:
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
    
    def _expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected win probability"""
        diff = rating2 - rating1
        return 1 / (1 + 10 ** (diff / 400))
    
    def update_ratings(self, winner: str, loser: str):
        if winner not in self.ratings:
            self.ratings[winner] = self.initial_rating
        if loser not in self.ratings:
            self.ratings[loser] = self.initial_rating
        
        expected_win_prob = self._expected_score(
            self.ratings[winner], 
            self.ratings[loser]
        )
        
        rating_change = self.k_factor * (1 - expected_win_prob)
        
        self.ratings[winner] += rating_change
        self.ratings[loser] -= rating_change
    
    def pairwise_comparison(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        metrics_cols = [
            'Креативность', 
            'Разнообразие', 
            'Релевантность', 
            'Среднее'
        ]
        
        for metric in metrics_cols:
            # Normalize metric values
            normalized_metric = (metrics_df[metric] - metrics_df[metric].min()) / \
                                (metrics_df[metric].max() - metrics_df[metric].min())
            
            # Random sampling for comparisons
            for _ in range(len(metrics_df) * 3):  # Multiple iterations
                idx1, idx2 = random.sample(range(len(metrics_df)), 2)
                
                if normalized_metric[idx1] > normalized_metric[idx2]:
                    self.update_ratings(
                        metrics_df.iloc[idx1].name, 
                        metrics_df.iloc[idx2].name
                    )
                else:
                    self.update_ratings(
                        metrics_df.iloc[idx2].name, 
                        metrics_df.iloc[idx1].name
                    )
        
        ratings_df = pd.DataFrame.from_dict(
            self.ratings, 
            orient='index', 
            columns=['ELO Rating']
        ).reset_index()
        ratings_df.columns = ['Model', 'ELO Rating']
        ratings_df = ratings_df.sort_values('ELO Rating', ascending=False)
        
        return ratings_df
    
    def confidence_interval(self, rating: float, games: int) -> Tuple[float, float]:
        """
        Calculate confidence interval for a rating
        
        Args:
            rating: Current ELO rating
            games: Number of games/comparisons
        
        Returns:
            Tuple of lower and upper confidence bounds
        """
        std_dev = self.k_factor / np.sqrt(games)
        confidence = 0.95
        z_score = norm.ppf((1 + confidence) / 2)
        
        lower_bound = rating - z_score * std_dev
        upper_bound = rating + z_score * std_dev
        
        return lower_bound, upper_bound

def main():
    metrics_df = pd.read_csv('evaluated_responses.csv')
    elo_rater = EloRater()
    
    # Perform pairwise comparisons
    elo_ratings = elo_rater.pairwise_comparison(metrics_df)
    
    elo_ratings['Lower CI'] = elo_ratings.apply(
        lambda row: elo_rater.confidence_interval(row['ELO Rating'], 10)[0], 
        axis=1
    )
    elo_ratings['Upper CI'] = elo_ratings.apply(
        lambda row: elo_rater.confidence_interval(row['ELO Rating'], 10)[1], 
        axis=1
    )
    
    elo_ratings.to_csv('model_elo_ratings.csv', index=False)
    print(elo_ratings)

if __name__ == "__main__":
    main()