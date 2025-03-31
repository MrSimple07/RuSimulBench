# RuSimulBench: Russian Language Benchmark for LLMs

<p align="center">
  <picture>
    <img alt="RuSimulBench" src="pics/Rusimulbench-logo.png" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://github.com/MrSimple07/RuSimulBench/">
    <img alt="Release" src="https://img.shields.io/badge/release-v1.1.0-blue">
    </a>

</p>

<h2 align="center">
    <p> RuSimulBench is a new open benchmark for the Russian language for evaluating fundamental models.
</p>
</h2>

## Overview
In recent years, Large Language Models (LLMs) have made significant strides in natural language processing (NLP), demonstrating remarkable capabilities in text generation, reasoning, and domain-specific adaptations. These advancements have led to widespread applications in content creation, machine translation, and conversational AI.

However, while English-language models benefit from well-established evaluation frameworks, Russian LLMs lack standardized benchmarks that measure their ability to generate high-quality, contextually appropriate, and robust responses. This gap poses a challenge in improving and fine-tuning Russian LLMs for real-world applications.

## Why This Benchmark?
Existing benchmarks primarily assess general language understanding but do not account for complex aspects such as **creativity** and **stability**, which are critical for advanced AI applications, including content generation and decision-support systems.

- **Creativity** ensures that models generate diverse and original responses while maintaining contextual relevance.
- **Stability** ensures consistency in model behavior when faced with slight variations in input prompts, making AI models more reliable and trustworthy.

## Key Features
- **Standardized evaluation methodology** for Russian LLMs
- **Focus on creativity and stability** for advanced AI applications
- **Publicly available** for research and development

## Demo
Check out the **RuSimulBench Arena** on Hugging Face:
[![Demo](https://img.shields.io/badge/Demo-HuggingFace-blue)](https://huggingface.co/spaces/MrSimple01/RuSimulBench_arena)

## Tasks & Datasets
Our benchmarking framework leverages:
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts) 📝
- [MERA: Russian LLM Evaluation](https://github.com/ai-forever/MERA) 🏆

## Installation & Usage
To use our benchmarking framework, clone this repository and follow the instructions:
```bash
# Clone the repository
git clone https://github.com/rusimulbench.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run the benchmark
python benchmark.py
```

## Contributing
We welcome contributions! Feel free to open issues or submit pull requests to improve the benchmark.

## License
This project is licensed under the MIT License.

## Contact
abdurahimov.muslimbek@gmail.com
