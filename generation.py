import os
import pandas as pd
import argparse
import time
from tqdm import tqdm
import json
import logging
from openai import OpenAI
import anthropic
import requests
import backoff

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Generate responses for RuSimulBench tasks')
    parser.add_argument('--input_file', type=str, default='tasks/rusimulbench_all_tasks.csv',
                        help='Path to the input CSV file containing tasks')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save the results')
    parser.add_argument('--models', nargs='+', default=['gpt-4o', 'claude-3-opus', 'gemini-pro'],
                        help='Models to evaluate')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--anthropic_api_key', type=str, help='Anthropic API key')
    parser.add_argument('--google_api_key', type=str, help='Google API key')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    return parser.parse_args()

def create_output_file(input_file, output_dir, models):
    df = pd.read_csv(input_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add columns for each model's answers
    for model in models:
        if f"{model}_answers" not in df.columns:
            df[f"{model}_answers"] = ""
    
    output_file = os.path.join(output_dir, 'rusimulbench_responses.csv')
    df.to_csv(output_file, index=False)
    return df, output_file

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_openai_response(prompt, model_name, api_key):
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error with OpenAI API: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_anthropic_response(prompt, model_name, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error with Anthropic API: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_gemini_response(prompt, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024
        }
    }
    
    response = requests.post(
        f"{url}?key={api_key}",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        logger.error(f"Error with Gemini API: {response.text}")
        raise Exception(f"Gemini API error: {response.status_code}")
    
    response_json = response.json()
    return response_json["candidates"][0]["content"]["parts"][0]["text"]

def generate_responses(df, models, args):
    total_rows = len(df)
    
    for model in models:
        logger.info(f"Generating responses for model: {model}")
        
        # Skip if the model's answers are already generated
        if all(df[f"{model}_answers"].notna()) and all(df[f"{model}_answers"] != ""):
            logger.info(f"Responses for {model} already exist, skipping...")
            continue
            
        for i in tqdm(range(total_rows), desc=f"Processing {model}"):
            if pd.notna(df.at[i, f"{model}_answers"]) and df.at[i, f"{model}_answers"] != "":
                continue
                
            prompt = df.at[i, 'rus_prompt']
            
            try:
                if model.startswith("gpt"):
                    response = generate_openai_response(prompt, model, args.api_key)
                elif model.startswith("claude"):
                    response = generate_anthropic_response(prompt, model, args.anthropic_api_key)
                elif model.startswith("gemini"):
                    response = generate_gemini_response(prompt, args.google_api_key)
                else:
                    logger.warning(f"Model {model} not supported, skipping...")
                    break
                    
                df.at[i, f"{model}_answers"] = response
                
                # Save after each batch or at the end
                if (i + 1) % args.batch_size == 0 or i == total_rows - 1:
                    df.to_csv(os.path.join(args.output_dir, 'rusimulbench_responses.csv'), index=False)
                    logger.info(f"Saved batch for {model}, progress: {i+1}/{total_rows}")
                    
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error generating response for {model} on row {i}: {str(e)}")
                # Save progress even on error
                df.to_csv(os.path.join(args.output_dir, 'rusimulbench_responses.csv'), index=False)
                
    return df

def main():
    args = setup_argparse()
    
    # Load environment variables if API keys not provided
    if not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY")
    if not args.anthropic_api_key:
        args.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not args.google_api_key:
        args.google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    logger.info("Starting generation process...")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Models to evaluate: {args.models}")
    
    # Create output file with columns for each model
    df, output_file = create_output_file(args.input_file, args.output_dir, args.models)
    logger.info(f"Created output file: {output_file}")
    
    # Generate responses
    df = generate_responses(df, args.models, args)
    
    logger.info("Generation completed successfully!")

if __name__ == "__main__":
    main()
