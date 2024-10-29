

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
    git clone https://github.com/sbrahmasamudra/cye.git
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
The csv (QNA_DATA.CSV) uses a small, custom dataset for demonstration. The dataset contains simple question and answer text pairs. Keep the file in the same directory as your Py scripts that trains the model. 

```
#1: Read the file
df = pd.read_csv("qna_data.csv")
df['text'] = 'Question:\n' + df['Question'] + '\n\nAnswer:\n' + df['Answer']
df.drop(columns=['Question','Answer'], axis=1, inplace=True)
```

## Model Training

1. Using bitsandbytes library for memory management to load the pre-trained model of 7B.

   ``` bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype)
   ```

2. Tokenizer to convert the loaded data. THis step will tokenize the sentences to words to ids and load the pre-trained model. 

```tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=bnb_config,device_map="auto")
```

3. LORA config - Finetuning specific params to keep the costs low.

```
peft_config = LoraConfig(
                          lora_alpha=16,
                          lora_dropout=0.1,
                          r=64,
                          bias="none",
                          task_type="CAUSAL_LM"
                        )
```

4. Setting up the training arguments to save create epoch based checkpoints, learning rate, and also tell the model where to output the finetuned model.

```
args = TrainingArguments(
    output_dir='llama2-7b-tuned-qna',
    num_train_epochs=10, # adjust based on the data size
    per_device_train_batch_size=2, # use 4 if you have more GPU RAM
    save_strategy="epoch", #steps
    # evaluation_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    seed=42
)

```

5. Setting up the trainer with the finetuned model and the data set. Here you can have training dataset, evaluation dataset, and validations dataset based on your requirements.

```
trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    # eval_dataset=test,
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=1042,
    tokenizer=tokenizer,
    args=args,
    packing=True,
)
trainer.train()
trainer.save_model()

```
   
6. Save the model and merge the model to be used for your applicaitons or upload it to Huggingface for consumption.

```
merged_model = new_model.merge_and_unload()

#17: Save the merged model
merged_model.save_pretrained("metallama2-7b-qa-tuned-merged", safe_serialization=True)
tokenizer.save_pretrained("metallama2-7b-qa-tuned-merged")

```

## Testing and Troubleshooting

1. The end of training the model and merged will create metallama2-7b-qa-tuned-merged as a directory with several files. It will include the tokenizer.json, the model safetencors (3 parts).
2. To test if the Prompt is geenrating text, create another Python Service with an API end-point using Flask or Fast API. The below is to test if the /predict end-point is working in the local machine which is your VM you have provisioned leveraging Foundry's Spot/Compute instance.

   ```
   In your Terminal or POSTMAN:
   
   curl -X POST "http://127.0.0.1:8080/predict" -H "Content-Type: application/json" -d '{"prompt": "What is your purpose?"}
   ```
    ```
    OUTPUT: {"response":"What is your purpose? Are you a business? Are you a brand? Or are you a person?"}
    ```
    
3. Follow this command and make sure a port like 8080 is open - https://docs.mlfoundry.com/foundry-documentation/compute-instances/managing-open-ports
4. Server :
    ```
        #app.py
        This script acts as a server and waits for /predict requests from clients and users
    ```
5. genResponse:

   ```
    #genResponse().py
    This script returns the response by knowing where the fine-tuned model is, loading the model, and running the tokenizer on the 'prompt'. This is the resposne that is returned to app.py.
   ```

7. Use the below commands to ensure an output like below exists that confirms the port is active and running and can accept external HTTP requests.

   ```
   ubuntu@unwilling-jaguar-1:/etc/systemd/system$ ss -tuln | grep 8080
   tcp   LISTEN 0      2048               0.0.0.0:8080       0.0.0.0:*      
   ```
   ```
    ubuntu@unwilling-jaguar-1:/etc/systemd/system$ cat examplefoundry.service 

                [Unit]  
                Description=Foundry Port Forwarding Service  
                After=network.target
                Wants=network-online.target
                [Service]  
                Type=simple  
                User=root  
                ExecStart=/usr/local/bin/foundrypf 8080
                Restart=always  
                RestartSec=3
                [Install]  
                WantedBy=multi-user.target
   ```
    TESTING /predict as an external API endpoint
    
   ```
    curl -X POST "http://<PUBLIC_IP_ADDRESS_OF_YOUR_VM>:8080/predict" -H "Content-Type: application/json" -d '{"prompt": "What is your purpose?"}'
     OUTPUT: {"response":"What is your purpose? Are you a business? Are you a brand? Or are you a person?"}  
   ```

8. Check logs in app.py whenever you make an HTTP request

```
ubuntu@unwilling-jaguar-1:~/foundry$ python3 app.py 
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.18it/s]
INFO:     Started server process [3543]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     127.0.0.1:33854 - "POST /predict HTTP/1.1" 200 OK
```
