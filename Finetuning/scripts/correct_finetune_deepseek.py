#!/usr/bin/env python3
"""
Correct way to fine-tune DeepSeek for CTR prediction

Key fixes:
1. Use apply_chat_template() instead of str(dict)
2. Convert CTR to string
3. Properly format the training data
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load dataset
# ============================================================
print("=" * 60)
print("Loading dataset...")
print("=" * 60)

data_files = {
    "train": "../data/train.csv",
    "validation": "../data/valid.csv",
    "test": "../data/test.csv"
}

ctr_datasets = load_dataset("csv", data_files=data_files)
print(ctr_datasets)

# ============================================================
# 2. Setup tokenizer and model
# ============================================================
print("\n" + "=" * 60)
print("Loading model and tokenizer...")
print("=" * 60)

checkpoint = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 3. CORRECT WAY: Format data using chat template
# ============================================================
print("\n" + "=" * 60)
print("Formatting data with chat template...")
print("=" * 60)

SYSTEM_PROMPT = "You are an editor tasked with choosing catchy headlines for articles. Catchy means the headline that is likely to generate more clicks. Please predict the corresponding CTR (Click Through Rate: i.e., clicks over impressions) of the headline, ranging from 0 to 1. Only give the number, nothing else."

def format_example(example):
    """Format a single example using DeepSeek's chat template"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"The headline is: {example['headline']}."},
        {"role": "assistant", "content": str(example['CTR'])}  # MUST be string!
    ]
    
    # Use the tokenizer's chat template
    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=False
    )
    
    example["text"] = formatted_text
    return example

# Apply formatting
formatted_datasets = ctr_datasets.map(format_example, num_proc=8)

# Show example
print("\n--- Example formatted text (first 500 chars) ---")
print(formatted_datasets["train"][0]["text"][:500])
print("...")
print("--- End example ---\n")

# ============================================================
# 4. Load model with quantization
# ============================================================
print("Loading model with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# ============================================================
# 5. Setup LoRA
# ============================================================
print("Setting up LoRA...")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ============================================================
# 6. Training arguments
# ============================================================
training_args = TrainingArguments(
    output_dir="/home/users/s1155227960/large-data/ctr-deepseek-correct",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,  # Lower LR for more stable training
    logging_steps=10,
    num_train_epochs=1,  # Start with 1 epoch to test
    save_strategy="steps",
    save_steps=500,
    bf16=True,
    dataloader_num_workers=4,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    warmup_steps=100,
    report_to="none",
    max_grad_norm=1.0,  # Add gradient clipping
)

# ============================================================
# 7. Train with SFTTrainer
# ============================================================
print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_datasets["train"],
    eval_dataset=formatted_datasets["validation"],
    processing_class=tokenizer,
)

trainer.train()

# ============================================================
# 8. Save the model
# ============================================================
print("\n" + "=" * 60)
print("Saving model...")
print("=" * 60)

trainer.save_model("/home/users/s1155227960/large-data/ctr-deepseek-correct/final")
tokenizer.save_pretrained("/home/users/s1155227960/large-data/ctr-deepseek-correct/final")

print("Training complete!")

# ============================================================
# 9. Quick test
# ============================================================
print("\n" + "=" * 60)
print("Testing the fine-tuned model...")
print("=" * 60)

test_headlines = [
    "This Simple Trick Can Save You Thousands",
    "You Won't Believe What Happened Next",
]

for headline in test_headlines:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"The headline is: {headline}."}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_part = response.split("assistant")[-1].strip() if "assistant" in response else response
    
    print(f"\nHeadline: {headline[:40]}...")
    print(f"Predicted CTR: {assistant_part[:50]}")
