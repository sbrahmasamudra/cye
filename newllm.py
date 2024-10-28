import pandas as pd
import torch
from datasets import Dataset, load_dataset
from random import randrange
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from huggingface_hub import login

#1: Read the file
df = pd.read_csv("qna_data.csv")
df['text'] = 'Question:\n' + df['Question'] + '\n\nAnswer:\n' + df['Answer']
df.drop(columns=['Question','Answer'], axis=1, inplace=True)

#2: Convert to Huggingface Datasets format
train = Dataset.from_pandas(df)

#3: Define Parameters
model_id = "meta-llama/Llama-2-7b-hf"
hf_token ="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"

#3: Login to HF
login(hf_token)

#4: BitsandBytesConfig
# Get the type
compute_dtype = getattr(torch, "float16")
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

#5:Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#6: Load the pretrained_model
model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=bnb_config,device_map="auto")

#7LoRA config based on QLoRA paper
peft_config = LoraConfig(
                          lora_alpha=16,
                          lora_dropout=0.1,
                          r=64,
                          bias="none",
                          task_type="CAUSAL_LM"
                        )
#8: Training Arguments
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

#9:Create the trainer
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

#10:Train
trainer.train()

#11 save model in local
trainer.save_model()

#12: Empty VRAM
del model
del trainer
import gc
gc.collect()
gc.collect()

#13: Empty cache
torch.cuda.empty_cache()
gc.collect()

#14: PEFT
from peft import AutoPeftModelForCausalLM

new_model = AutoPeftModelForCausalLM.from_pretrained(
    'llama2-7b-tuned-qna',
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)


prompt = "What is the output format of the tasks that the Florence-2 model can handle?"
#ground truth = "The output format of the tasks that the Florence-2 model can handle is text forms, whether it be captioning, object detection, grounding or segmentation."


#15: Tokenizer
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = new_model.generate(input_ids=input_ids,
                         max_new_tokens=200,
                        #  do_sample=True,
                        #  top_p=0.9,
                         temperature=0.6)
result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
print(result)

#16: Merge LoRA and base model
merged_model = new_model.merge_and_unload()

#17: Save the merged model
merged_model.save_pretrained("metallama2-7b-qa-tuned-merged", safe_serialization=True)
tokenizer.save_pretrained("metallama2-7b-qa-tuned-merged")


