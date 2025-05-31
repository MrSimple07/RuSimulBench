# Model Evaluation System

This system evaluates model responses based on stability and creativity metrics, providing a comprehensive assessment of model performance.

## Features

- Stability evaluation using sentence-transformers (paraphrase-MiniLM-L6-v2)
- Creativity assessment using Gemini 1.5 Flash
- Combined Evaluation Score (CES) calculation
- Leaderboard for tracking model performance
- Batch processing of RuSimulBench tasks
- User-friendly Gradio interface

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open the Gradio interface in your browser

3. Input the following information:
   - Model Name: Name of the model being evaluated
   - Upload CSV File: The RuSimulBench tasks file (must contain 'act' and 'rus_prompt' columns)
   - Column Name: The column containing prompt variations for stability evaluation

4. Click "Process File" to:
   - Generate model outputs for all tasks
   - Calculate stability and creativity scores
   - Save results to a CSV file
   - Update the leaderboard

5. Click "Show Leaderboard" to view the performance history

## Evaluation Metrics

### Stability Score (S)
Calculated using sentence transformer embeddings and cosine similarity:
```
S = (1/N) * Σ Cosine Similarity(Ai, Ab)
```

### Creativity Score (CS)
Weighted combination of three dimensions:
```
CS = 0.4 * Creativity + 0.3 * Diversity + 0.3 * Coherence
```

### Combined Evaluation Score (CES)
Average of normalized Creativity Score and Stability Coefficient:
```
CES = (Normalized Creativity Score + Stability Coefficient) / 2
```

## Output Format

The system provides:
- Stability Score for each task
- Creativity Score for each task
- Detailed Creativity Scores (Creativity, Diversity, Coherence)
- Combined Evaluation Score
- Model outputs saved to CSV file
- Leaderboard with historical results

## File Format

The input CSV file should contain:
- `act`: Task identifier
- `rus_prompt`: The prompt in Russian
- Additional columns for variations (specified in the interface)

The output CSV file will contain:
- `act`: Task identifier
- `rus_prompt`: Original prompt
- `model_output`: Generated response
- `model_name`: Name of the model 