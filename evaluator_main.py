import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import logging
from evaluate_creativity import evaluate_creativity
from evaluate_stability import (
    evaluate_stability, 
    evaluate_combined_score, 
    create_radar_chart, 
    create_bar_chart,
    get_leaderboard_data
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Evaluate models on RuSimulBench')
    parser.add_argument('--input_file', type=str, default='results/rusimulbench_responses.csv',
                        help='Path to the CSV file with model responses')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--api_key', type=str, help='API key for evaluation')
    parser.add_argument('--models', nargs='+', help='Models to evaluate (if not specified, all models will be evaluated)')
    return parser.parse_args()

def list_available_models(csv_file):
    try:
        df = pd.read_csv(csv_file)
        model_columns = [col for col in df.columns if col.endswith('_answers')]
        models = [col.replace('_answers', '') for col in model_columns]
        return models
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return []

def evaluate_models(file_path, api_key, prompt_col, selected_models=None):
    os.makedirs('results', exist_ok=True)
    
    logger.info("Loading data...")
    df = pd.read_csv(file_path)
    
    # Determine which models to evaluate
    if selected_models:
        answer_cols = [f"{model}_answers" for model in selected_models if f"{model}_answers" in df.columns]
        models = [col.replace('_answers', '') for col in answer_cols]
    else:
        answer_cols = [col for col in df.columns if col.endswith('_answers')]
        models = [col.replace('_answers', '') for col in answer_cols]
    
    model_mapping = dict(zip(models, answer_cols))
    
    logger.info(f"Found {len(model_mapping)} models to evaluate: {', '.join(models)}")
    
    all_results = {}
    all_creativity_dfs = {}
    
    benchmark_file = 'results/benchmark_results.csv'
    if os.path.exists(benchmark_file):
        try:
            benchmark_df = pd.read_csv(benchmark_file)
        except:
            benchmark_df = pd.DataFrame(columns=[
                'model', 'creativity_score', 'stability_score', 
                'combined_score', 'evaluation_timestamp'
            ])
    else:
        benchmark_df = pd.DataFrame(columns=[
            'model', 'creativity_score', 'stability_score', 
            'combined_score', 'evaluation_timestamp'
        ])
    
    for model, column in model_mapping.items():
        try:
            logger.info(f"Evaluating {model}...")
            
            # Evaluate creativity
            logger.info(f"Evaluating creativity for {model}...")
            creativity_df = evaluate_creativity(api_key, df, prompt_col, column, batch_size=5)
            
            # Evaluate stability
            logger.info(f"Evaluating stability for {model}...")
            stability_results = evaluate_stability(df, prompt_col, column)
            
            # Calculate combined score
            logger.info(f"Calculating combined score for {model}...")
            combined_results = evaluate_combined_score(creativity_df, stability_results, model)
            
            # Save detailed results
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_file = f'results/evaluated_responses_{model}_{timestamp}.csv'
            creativity_df.to_csv(output_file, index=False)
            
            # Add to benchmark DataFrame
            result_row = {
                'model': model,
                'creativity_score': combined_results['creativity_score'],
                'stability_score': combined_results['stability_score'],
                'combined_score': combined_results['combined_score'],
                'evaluation_timestamp': combined_results['evaluation_timestamp']
            }
            benchmark_df = pd.concat([benchmark_df, pd.DataFrame([result_row])], ignore_index=True)
            
            all_results[model] = combined_results
            all_creativity_dfs[model] = creativity_df
            
            logger.info(f"Finished evaluating {model}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model}: {str(e)}")
    
    # Save benchmark results
    benchmark_df.to_csv(benchmark_file, index=False)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_benchmark_path = f'results/benchmark_results_{timestamp}.csv'
    benchmark_df.to_csv(combined_benchmark_path, index=False)
    
    logger.info("Creating visualizations...")
    radar_chart_path = create_radar_chart(all_results)
    bar_chart_path = create_bar_chart(all_results)
    
    logger.info("Evaluation complete!")
    
    sorted_results = benchmark_df.sort_values(by='combined_score', ascending=False)
    
    # Print results to terminal
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Evaluated models: {', '.join(models)}")
    print("\nRESULTS SUMMARY (sorted by combined score):")
    print("-"*50)
    for i, row in sorted_results.iterrows():
        print(f"Model: {row['model']}")
        print(f"  Combined Score: {row['combined_score']:.2f}")
        print(f"  Creativity Score: {row['creativity_score']:.2f}")
        print(f"  Stability Score: {row['stability_score']:.2f}")
        print("-"*30)
    
    # Display file paths
    print("\nOUTPUT FILES:")
    print(f"Results saved to: {combined_benchmark_path}")
    print(f"Radar chart: {radar_chart_path}")
    print(f"Bar chart: {bar_chart_path}")
    print("="*50)
    
    return sorted_results, radar_chart_path, bar_chart_path, combined_benchmark_path

def main():
    args = setup_argparse()
    
    # Load API key from environment if not provided
    if not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    
    # Get all available models if not specified
    if not args.models:
        args.models = list_available_models(args.input_file)
        
    logger.info(f"Starting evaluation with models: {args.models}")
    
    # Run evaluation
    sorted_results, radar_chart_path, bar_chart_path, benchmark_path = evaluate_models(
        args.input_file, 
        args.api_key, 
        prompt_col='rus_prompt', 
        selected_models=args.models
    )
    
    logger.info("Evaluation process completed successfully!")

if __name__ == "__main__":
    main()
