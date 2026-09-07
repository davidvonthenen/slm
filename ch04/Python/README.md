# Building An Python Coding SLM Using Fine-Tuning (and Quantization)

This project provides an end-to-end build of a Python Coding Small Language Model (SLM) via fine-tuning and quantization.

> **IMPORTANT:** Do not actually use any of these examples models for production. There are already a **TON** of models that do this exact thing built for production. (Check out HuggingFace) Definitely, use those models over these. Some examples below...
>
> Take a look at:
> [https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
>
> CPU Quantized:
> [https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)

## Prerequisites

- Python 3.12+
- An H100 (or better)

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

## Usage

### Step 1: Benchmark Qwen 2.5 7B for Python Understanding

This step will benchmark and create test results for Qwen 2.5 7B understanding the Python language suitable as an expert in that field.

```bash
python baseline_python_benchmark.py
```

### Step 2: Data Prep + Fine-Tune A Model for Python Coding

This step will perform data prep and fine-tune your model.

> **IMPORTANT:** This step minimally requires an NVIDIA H100 and will take roughly 14-16 hours to finish. For those that don't have access to those resources nor want to take on the cost burden, a fine-tuned safetensor model will be provided for you below to continue with the rest of the tasks.

```bash
python finetune.py
```

Download these prebuild models below and skip to `Step 3`. Note these fine-tuned safetensor models are quite large and will take some time to download. Total download size is 20GB.

PRIMARY DOWNLOAD:
XXXX

BACKUP DOWNLOAD:
XXXX

### Step 3: Benchmark the Specialist Model

This step will benchmark and create test results for our fine-tuned Python Coding expert's understanding the Python language.

```bash
python specialist_python_benchmark.py
```
