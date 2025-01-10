# NIR - 4th Semester
# RuSimulBench: Russian Language Creativity Benchmark for LLMs

## Overview

RuSimulBench is designed to assess the creative capabilities of Large Language Models (LLMs) specifically for the Russian language. The benchmark incorporates culturally-aware evaluation metrics and Russian-specific linguistic features, providing a standardized way to measure model performance in creative tasks.

## Features

- Russian-specific creativity evaluation metrics
- Culturally adapted testing scenarios
- Comprehensive evaluation across multiple creative dimensions
- Support for various LLM architectures
- Transparent scoring methodology

## Supported Models

The benchmark has been tested with the following models:

- IlyaGusev/saiga_llama3_8b
- Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24
- TinyLlama
- Google-gemma-2-27b-it
- mistralai/Mistral-Nemo-Instruct-2407
- Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct

## Evaluation Metrics

The benchmark evaluates models across three key dimensions:
- Creativity Score
- Coherence Score
- Diversity Score

Evaluation is performed using the Gemini 1-5 flash API for consistent and reliable scoring.

## Demo

Try out the benchmark in our interactive demo:
[RuSimulBench Arena on Hugging Face](https://huggingface.co/spaces/MrSimple07/RuSimulBench_arena)

## Dataset

The benchmark dataset is available in the `data` directory and includes various creative tasks specifically designed for Russian language evaluation. For detailed analysis of the dataset, refer to our [EDA notebook](https://github.com/MrSimple07/RuSimulBench/blob/main/rusimulbench-4-eda.ipynb).

## Technical Objectives

1. **Benchmark Analysis**: Comprehensive study of existing creativity benchmarks, focusing on adaptation requirements for Russian language.

2. **Cultural Adaptation**: Translation and cultural adaptation of tasks, preserving evaluation integrity while incorporating Russian cultural context.

3. **Dataset Curation**: Development of Russian-specific datasets capturing:
   - Cultural nuances
   - Historical context
   - Linguistic features
   - Creative writing scenarios
   - Storytelling elements

## Project Structure

```
RuSimulBench/
├── data/                 # Benchmark datasets
├── notebooks/           
│   └── rusimulbench-4-eda.ipynb  # Exploratory Data Analysis
├── src/                 # Source code
├── examples/            # Usage examples
└── README.md           # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Contact
abdurahimov.muslimbek@gmail.com
