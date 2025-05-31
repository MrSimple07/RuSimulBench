import gradio as gr
from evaluation import ModelEvaluator
from leaderboard import Leaderboard
import pandas as pd
import os
import json
import numpy as np

# Initialize components
print("Initializing application components...")
evaluator = ModelEvaluator()
leaderboard = Leaderboard()
print("Components initialized successfully")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_outputs(file, model_name):
    print(f"\nGenerating outputs for file: {file.name}")
    print(f"Model name: {model_name}")
    
    # Read the uploaded file
    print("Reading CSV file...")
    df = pd.read_csv(file.name)
    print(f"File loaded successfully. Found {len(df)} tasks")
    
    # Generate model outputs
    print("\nGenerating model outputs...")
    model_outputs = evaluator.generate_model_outputs(df, model_name)
    
    # Save model outputs
    output_filename = f"{model_name}_results.csv"
    print(f"\nSaving model outputs to {output_filename}...")
    model_outputs.to_csv(output_filename, index=False)
    print("Model outputs saved successfully")
    
    return output_filename, model_outputs

def evaluate_model(file, model_name, model_outputs_column):
    print(f"\nEvaluating model outputs from file: {file.name}")
    print(f"Model name: {model_name}")
    print(f"Model outputs column: {model_outputs_column}")
    
    # Read the uploaded file
    print("Reading CSV file...")
    df = pd.read_csv(file.name)
    print(f"File loaded successfully. Found {len(df)} tasks")
    
    # Calculate scores for each task
    print("\nCalculating scores for each task...")
    task_results = []
    
    # First pass: calculate all scores
    for idx, row in df.iterrows():
        print(f"\nProcessing task {idx + 1}/{len(df)}...")
        # Get model output for this task
        model_output = row[model_outputs_column]
        
        # Get variations for this task
        task_variations = df[df['act'] == row['act']][model_outputs_column].tolist()
        
        print("Calculating stability score...")
        stability_score = float(evaluator.calculate_stability(task_variations, model_output))
        
        print("Calculating creativity scores...")
        creativity_score, detailed_scores = evaluator.evaluate_creativity(model_output)
        creativity_score = float(creativity_score)
        
        print("Calculating combined score...")
        combined_score = float(evaluator.calculate_combined_score(creativity_score, stability_score))
        
        task_results.append({
            "task_id": row['act'],
            "baseline_prompt": row['rus_prompt'],
            "variations": task_variations,
            "stability_score": stability_score,
            "creativity_score": creativity_score,
            "detailed_scores": {
                "creativity": float(detailed_scores['creativity']),
                "diversity": float(detailed_scores['diversity']),
                "coherence": float(detailed_scores['coherence'])
            },
            "combined_score": combined_score
        })
    
    print("\nAll tasks evaluated successfully")
    
    # Add results to leaderboard
    print("\nAdding results to leaderboard...")
    leaderboard_result = leaderboard.add_batch_results(model_name, task_results)
    
    # Format results for display
    display_results = {
        "Model Name": model_name,
        "Number of Tasks": len(task_results),
        "Average Stability Score": f"{leaderboard_result['avg_stability_score']:.3f}",
        "Average Creativity Score": f"{leaderboard_result['avg_creativity_score']:.3f}",
        "Average Combined Score": f"{leaderboard_result['avg_combined_score']:.3f}",
        "Detailed Results": task_results
    }
    
    return display_results

def show_leaderboard():
    print("\nRetrieving leaderboard...")
    df = leaderboard.get_results_df()
    if df.empty:
        print("No results in leaderboard yet")
        return "No results yet"
    
    # Format the leaderboard display
    print("Formatting leaderboard display...")
    df_display = df[['timestamp', 'model_name', 'avg_combined_score', 'num_tasks']].copy()
    df_display['avg_combined_score'] = df_display['avg_combined_score'].round(3)
    print("Leaderboard retrieved successfully")
    return df_display.to_string(index=False)

# Create Gradio interface
print("\nCreating Gradio interface...")
with gr.Blocks(title="Model Evaluation System") as app:
    gr.Markdown("# Model Evaluation System")
    
    with gr.Row():
        with gr.Column():
            model_name = gr.Textbox(label="Model Name")
            file_input = gr.File(label="Upload CSV File")
            
            with gr.Row():
                generate_btn = gr.Button("Generate Model Outputs")
                evaluate_btn = gr.Button("Evaluate Model")
            
            model_outputs_column = gr.Textbox(
                label="Column Name for Model Outputs",
                placeholder="Enter the column name containing model outputs to evaluate"
            )
            
            output_file = gr.File(label="Generated Model Outputs File")
            results = gr.JSON(label="Evaluation Results")
    
    with gr.Row():
        leaderboard_btn = gr.Button("Show Leaderboard")
        leaderboard_display = gr.Textbox(label="Leaderboard", lines=10)
    
    # Set up event handlers
    generate_btn.click(
        fn=generate_outputs,
        inputs=[file_input, model_name],
        outputs=[output_file]
    )
    
    evaluate_btn.click(
        fn=evaluate_model,
        inputs=[file_input, model_name, model_outputs_column],
        outputs=results
    )
    
    leaderboard_btn.click(
        fn=show_leaderboard,
        outputs=leaderboard_display
    )

if __name__ == "__main__":
    print("\nStarting application...")
    app.launch() 