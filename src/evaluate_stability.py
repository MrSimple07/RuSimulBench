import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_stability(df: pd.DataFrame, prompt_col: str, answer_col: str,
                       model_name: str = 'paraphrase-MiniLM-L6-v2',
                       progress=None) -> Dict:
    if progress:
        progress(0, desc="Loading sentence transformer model...")
    
    model = SentenceTransformer(model_name)
    
    prompts = df[prompt_col].tolist()
    outputs = df[answer_col].tolist()
    
    if progress:
        progress(0.3, desc="Encoding prompts...")
    prompt_embeddings = model.encode(prompts)
    
    if progress:
        progress(0.6, desc="Encoding outputs...")
    output_embeddings = model.encode(outputs)
    
    if progress:
        progress(0.9, desc="Computing similarities...")
    similarities = cosine_similarity(prompt_embeddings, output_embeddings)
    stability_coefficients = np.diag(similarities)
    
    if progress:
        progress(1.0, desc="Done!")
    return {
        'stability_score': np.mean(stability_coefficients) * 100,  
        'stability_std': np.std(stability_coefficients) * 100,
        'individual_similarities': stability_coefficients
    }

def evaluate_combined_score(creativity_df: pd.DataFrame, stability_results: Dict, 
                           model_name: str) -> Dict:
    creative_score = creativity_df["Среднее"].mean()
    stability_score = stability_results['stability_score']
    combined_score = (creative_score + stability_score) / 2
    
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        'model': model_name,
        'creativity_score': creative_score,
        'stability_score': stability_score,
        'combined_score': combined_score,
        'evaluation_timestamp': timestamp,
        'creative_details': {
            'creativity': creativity_df["Креативность"].mean(),
            'diversity': creativity_df["Разнообразие"].mean(),
            'relevance': creativity_df["Релевантность"].mean(),
        },
        'stability_details': stability_results
    }

def create_radar_chart(all_results):
    os.makedirs('results', exist_ok=True)
    
    # Extract data for radar chart
    categories = ['Креативность', 'Разнообразие', 'Релевантность', 'Стабильность']
    models = list(all_results.keys())
    
    # Create figure and polar axis
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Number of variables
    N = len(categories)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Draw the polygons for each model
    for i, model in enumerate(models):
        values = [
            all_results[model]['creative_details']['creativity'],
            all_results[model]['creative_details']['diversity'],
            all_results[model]['creative_details']['relevance'],
            all_results[model]['stability_score']
        ]
        
        # Add the first value again to close the polygon
        values += values[:1]
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title('Model Performance Comparison', size=15, pad=20)
    
    # Save the chart
    radar_chart_path = 'results/radar_chart.png'
    plt.savefig(radar_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return radar_chart_path

def create_bar_chart(all_results):
    # Extract data for bar chart
    models = list(all_results.keys())
    creative_scores = [all_results[model]['creativity_score'] for model in models]
    stability_scores = [all_results[model]['stability_score'] for model in models]
    combined_scores = [all_results[model]['combined_score'] for model in models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set bar width
    bar_width = 0.25
    
    # Set bar positions
    r1 = np.arange(len(models))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    ax.bar(r1, creative_scores, width=bar_width, label='Креативность', color='skyblue')
    ax.bar(r2, stability_scores, width=bar_width, label='Стабильность', color='orange')
    ax.bar(r3, combined_scores, width=bar_width, label='Общий балл', color='green')
    
    # Add labels and title
    ax.set_xlabel('Модели')
    ax.set_ylabel('Оценка')
    ax.set_title('Сравнение моделей по креативности и стабильности')
    ax.set_xticks([r + bar_width for r in range(len(models))])
    ax.set_xticklabels(models)
    
    # Add legend
    ax.legend()
    
    # Save the chart
    bar_chart_path = 'results/bar_chart.png'
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return bar_chart_path

def get_leaderboard_data():
    benchmark_file = 'results/benchmark_results.csv'
    if not os.path.exists(benchmark_file):
        return pd.DataFrame(columns=[
            "Model", "Креативность", "Разнообразие", "Релевантность", "Стабильность", "Общий балл"
        ])
    
    try:
        df = pd.read_csv(benchmark_file)
        # Format the dataframe for display
        formatted_df = pd.DataFrame({
            "Model": df['model'],
            "Креативность": df['creativity_score'].round(2),
            "Стабильность": df['stability_score'].round(2),
            "Общий балл": df['combined_score'].round(2)
        })
        return formatted_df.sort_values(by="Общий балл", ascending=False)
    except Exception as e:
        print(f"Error loading leaderboard data: {str(e)}")
        return pd.DataFrame(columns=[
            "Model", "Креативность", "Разнообразие", "Релевантность", "Стабильность", "Общий балл"
        ])