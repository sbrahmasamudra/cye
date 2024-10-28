
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from huggingface_hub import login
import torch

#HuggingFace Token
hf_token =""

torch.cuda.empty_cache()

#Logging into HuggingFace to dowload the model
login (token=hf_token)

# Creating the dataset
data = {
    "source_text": ["Hi there", "How are you?", "What's your name?", "Goodbye",
                    "What time is it?", "Thank you", "I am hungry",
                    "Where is the store?", "I'm tired", "Help me"],
    "target_text": ["Hello", "I am fine", "I am ChatGPT", "See you",
                    "It's noon", "You're welcome", "Let's eat",
                    "It's on Main Street", "Take a nap", "I am here to help"]
}

# Convert to Hugging Face dataset
dataset = Dataset.from_dict(data)


# Load the Llama tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=hf_token)

# Set the padding token to an existing token or add a new one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the source and target texts
def tokenize_function(example):
    source = tokenizer(example['source_text'], padding="max_length", truncation=True, max_length=128)
    target = tokenizer(example['target_text'], padding="max_length", truncation=True, max_length=128)

    # Return input_ids and labels for the model
    return {
        "input_ids": source["input_ids"],
        "attention_mask": source["attention_mask"],
        "labels": target["input_ids"]
    }

# Apply the tokenization function
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the Llama model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=hf_token)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./llama-7b-finetuned",
    per_device_train_batch_size=1,  # Use smaller batch size
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    num_train_epochs=5,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2
    fp16=True  # Enable mixed precision training
) 
#Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./llama-7b-finetuned")
tokenizer.save_pretrained("./llama-7b-finetuned")
