

# Llama 7B Model Fine-Tuning with Hugging Face on Foundry Cloud Platform

This project demonstrates how to fine-tune the Llama 7B model using Hugging Face’s `transformers` library. The code loads a simple dataset, tokenizes it, and trains the model using the `SFTTrainer` with mixed precision and gradient accumulation for efficient training.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Testing & Troubleshooting](#troubleshooting)

## Overview

This project fine-tunes the Llama 7B model (`meta-llama/Llama-2-7b-hf`) using a basic text dataset. The script includes tokenization, dataset preparation, and model training using Hugging Face’s `Trainer` API. The training process is optimized with gradient accumulation and mixed precision (`fp16`) for better performance on compatible hardware.

## Setup

### Prerequisites

- Python 3.10+
- CUDA and compatible GPU drivers
- PyTorch and Hugging Face libraries
- FCP Licence

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/llama-finetune.git
    cd llama-finetune
    ```

2. Install the required Python packages:
    ```bash
      pip install transformers datasets torch accelerate huggingface_hub accelerate trl fastapi uvicorn torch transformers pandas
    ```

3. Log in to Hugging Face to access the model:
    ```bash
    huggingface-cli login
    ```

## Data Preparation
The script uses a small, custom dataset for demonstration. The dataset contains simple question and answer text pairs:

```
#1: Read the file
df = pd.read_csv("qna_data.csv")
df['text'] = 'Question:\n' + df['Question'] + '\n\nAnswer:\n' + df['Answer']
df.drop(columns=['Question','Answer'], axis=1, inplace=True)
```

## Model Training

1. Using bitsandbytes library for memory management to load the pre-trained model of 7B.

2. LORA config - Finetuning specific params to keep the costs low.

3. Setting up the trainign arguments to save create epoch based checkpoints, learning rate, and also tell the model where to output the finetuned model.

4. Setting up the trainer with the finetuned model and the data set. Here you can have training dataset, evaluation dataset, and validations dataset based on your requirements.
   
5. Save the model and merge the model to be used for your applicaitons or upload it to Huggingface for consumption. 


## Testing and Troubleshooting
