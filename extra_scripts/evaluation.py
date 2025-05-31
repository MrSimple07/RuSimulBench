import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Tuple
import json
import time
from google.api_core import retry
import random

load_dotenv()

class ModelEvaluator:
    def __init__(self):
        print("Initializing ModelEvaluator...")
        # Initialize Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini model initialized")
        # Initialize sentence transformer
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("Sentence transformer model initialized")
        
    @retry.Retry(
        initial=1.0,
        maximum=60.0,
        multiplier=2.0,
        deadline=300.0,
        predicate=retry.if_exception_type(
            genai.types.GenerateContentResponseException,
            genai.types.StopCandidateException,
            genai.types.BlockedPromptException,
            genai.types.InternalServerError,
            genai.types.ServiceUnavailable,
            genai.types.ResourceExhausted
        )
    )
    def _generate_with_retry(self, prompt: str) -> str:
        """Generate content with retry logic and exponential backoff"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                wait_time = random.uniform(30, 60)  # Random wait between 30-60 seconds
                print(f"Rate limit hit. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                raise  # Re-raise to trigger retry
            raise
        
    def calculate_stability(self, responses: List[str], baseline_response: str) -> float:
        """
        Calculate stability coefficient using sentence transformer embeddings
        """
        print(f"Calculating stability for {len(responses)} variations...")
        # Get embeddings for all texts
        all_texts = [baseline_response] + responses
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(all_texts)
        
        # Get baseline embedding
        baseline_embedding = embeddings[0]
        
        # Calculate cosine similarities
        similarities = []
        for i in range(1, len(embeddings)):
            similarity = np.dot(baseline_embedding, embeddings[i]) / (
                np.linalg.norm(baseline_embedding) * np.linalg.norm(embeddings[i])
            )
            similarities.append(float(similarity))  # Convert to Python float
        
        # Calculate stability coefficient
        stability = float(np.mean(similarities))  # Convert to Python float
        print(f"Stability score calculated: {stability:.3f}")
        return stability

    def evaluate_creativity(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate creativity using Gemini model
        """
        print("Evaluating creativity...")
        prompt = f"""
        Please evaluate the following text on three dimensions from 0 to 10:
        1. Creativity (originality and inventiveness)
        2. Diversity (lexical and stylistic variety)
        3. Coherence (logical flow and structural integrity)
        
        Text to evaluate:
        {text}
        
        Please provide scores in the format:
        Creativity: X
        Diversity: Y
        Coherence: Z
        """
        
        print("Generating evaluation from Gemini...")
        try:
            response_text = self._generate_with_retry(prompt)
            scores = self._parse_scores(response_text)
            
            # Calculate weighted creativity score
            creativity_score = float((
                0.4 * scores['creativity'] +
                0.3 * scores['diversity'] +
                0.3 * scores['coherence']
            ) / 10)  # Normalize to 0-1 and convert to Python float
            
            print(f"Creativity evaluation complete. Score: {creativity_score:.3f}")
            return creativity_score, scores
        except Exception as e:
            print(f"Error in creativity evaluation: {str(e)}")
            # Return default scores in case of error
            default_scores = {
                'creativity': 5.0,
                'diversity': 5.0,
                'coherence': 5.0
            }
            return 0.5, default_scores  # Return middle score as default

    def _parse_scores(self, response_text: str) -> Dict[str, float]:
        """
        Parse scores from Gemini response
        """
        print("Parsing evaluation scores...")
        scores = {
            'creativity': 0.0,
            'diversity': 0.0,
            'coherence': 0.0
        }
        
        for line in response_text.split('\n'):
            line = line.lower()
            for key in scores.keys():
                if key in line:
                    try:
                        score = float(line.split(':')[-1].strip())
                        scores[key] = float(score)  # Ensure Python float
                    except:
                        continue
                        
        print(f"Parsed scores: {scores}")
        return scores

    def calculate_combined_score(self, creativity_score: float, stability_coefficient: float) -> float:
        """
        Calculate Combined Evaluation Score (CES)
        """
        combined_score = float((creativity_score + stability_coefficient) / 2)
        print(f"Combined score calculated: {combined_score:.3f}")
        return combined_score

    def generate_model_outputs(self, tasks_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Generate model outputs for all tasks
        """
        print(f"Generating model outputs for {len(tasks_df)} tasks...")
        results = []
        
        for idx, row in tasks_df.iterrows():
            print(f"Processing task {idx + 1}/{len(tasks_df)}...")
            prompt = row['rus_prompt']
            try:
                print("Generating response from Gemini...")
                response_text = self._generate_with_retry(prompt)
                results.append({
                    'act': row['act'],
                    'rus_prompt': prompt,
                    'model_output': response_text,
                    'model_name': model_name
                })
                print("Response generated successfully")
            except Exception as e:
                print(f"Error generating response for prompt: {prompt}")
                print(f"Error: {str(e)}")
                # Add placeholder for failed generation
                results.append({
                    'act': row['act'],
                    'rus_prompt': prompt,
                    'model_output': "Error: Failed to generate response",
                    'model_name': model_name
                })
                continue
        
        print("All model outputs generated")
        return pd.DataFrame(results) 