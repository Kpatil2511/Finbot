import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --------------------------
# Ensure offload folder exists
# --------------------------
os.makedirs("offload", exist_ok=True)

# --------------------------
# Load Dataset
# --------------------------
dataset = load_dataset("json", data_files="hindi_dataset2.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1)

# --------------------------
# Load Tokenizer
# --------------------------
model_id = "sarvamai/OpenHathi-7B-Hi-v0.1-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# --------------------------
# 4-bit Quantization Config
# --------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --------------------------
# Load Model
# --------------------------
print("Loading model with quantization config...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# --------------------------
# Prepare model for k-bit training
# --------------------------
model = prepare_model_for_kbit_training(model)

# --------------------------
# Apply LoRA
# --------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("LoRA applied successfully.")

# --------------------------
# Print trainable parameters
# --------------------------
model.print_trainable_parameters()

# --------------------------
# Gradient checkpointing & use_cache=False
# --------------------------
model.gradient_checkpointing_enable()
model.config.use_cache = False

# --------------------------
# Tokenization
# --------------------------
def format_example(example):
    text = f"Q: {example['instruction']}\nA: {example['output']}{tokenizer.eos_token}"
    return {"text": text}

def tokenize_function(examples):
    # Format the text first
    texts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        text = f"Q: {instruction}\nA: {output}{tokenizer.eos_token}"
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )
    
    # Create labels - for causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply formatting and tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names  # Remove original columns
)

# --------------------------
# Data collator
# --------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# --------------------------
# Training arguments
# --------------------------
training_args = TrainingArguments(
    output_dir="./finetuned-stock-qa",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to=[],
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    remove_unused_columns=True  # Changed to True since we removed columns
)

# --------------------------
# Trainer
# --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# --------------------------
# Train
# --------------------------
print("Starting training...")
trainer.train()

# --------------------------
# Save Model
# --------------------------
model.save_pretrained("./finetuned-stock-qa")
tokenizer.save_pretrained("./finetuned-stock-qa")
print("Model saved successfully!")