"""
Quick test to verify DeepSeek-7B-LoRA model outputs CTR values correctly.
"""
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
import warnings
warnings.filterwarnings('ignore')

# Paths
base_model_path = "deepseek-ai/deepseek-llm-7b-chat"
adapter_path = os.path.expanduser("~/large-data/ctr-deepseek-correct/final")

def load_model():
    """Load base model and merge LoRA adapter weights."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )

    print("Loading and merging LoRA weights...")
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        adapter_config = json.load(f)

    adapter_weights = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))

    lora_alpha = adapter_config.get("lora_alpha", 32)
    lora_r = adapter_config.get("r", 16)
    scaling = lora_alpha / lora_r

    merged_count = 0
    for key, value in adapter_weights.items():
        if "lora_A" in key:
            lora_B_key = key.replace("lora_A", "lora_B")
            if lora_B_key in adapter_weights:
                lora_A = value
                lora_B = adapter_weights[lora_B_key]
                base_key = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")

                try:
                    parts = base_key.split(".")
                    target = model
                    for part in parts[:-1]:
                        if part.isdigit():
                            target = target[int(part)]
                        else:
                            target = getattr(target, part)

                    param_name = parts[-1]
                    base_weight = getattr(target, param_name)
                    delta = (lora_B.to(base_weight.dtype) @ lora_A.to(base_weight.dtype)) * scaling
                    new_weight = base_weight.data + delta.to(base_weight.device)
                    getattr(target, param_name).data.copy_(new_weight)
                    merged_count += 1
                except Exception as e:
                    pass

    print(f"Merged {merged_count} LoRA weights")
    model.eval()

    return model, tokenizer


def predict_ctr(model, tokenizer, headline):
    """Predict CTR for a single headline."""
    messages = [
        {"role": "system", "content": "You are an editor tasked with choosing catchy headlines for articles. Catchy means the headline that is likely to generate more clicks. Please predict the corresponding CTR (Click Through Rate： i.e., clicks over impressions) of the headline, ranging from 0 to 1. Only give the number, nothing else."},
        {"role": "user", "content": f"Headline: {headline}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    try:
        pred_ctr = float(response.strip().split()[0])
        pred_ctr = max(0.0, min(1.0, pred_ctr))
    except (ValueError, IndexError):
        pred_ctr = None

    return pred_ctr, response


def main():
    print("=" * 60)
    print("Testing DeepSeek-7B-LoRA CTR Prediction Output")
    print("=" * 60)

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded successfully!\n")

    # Test headlines (from Upworthy dataset)
    test_headlines = [
        "This Little Girl's Reaction To A Surprise Puppy Is The Cutest Thing You'll See Today",
        "This Is What Happens When You Let A Dog Pick Your Halloween Costume",
        "13 Kids Who Are Way Funnier Than Their Parents",
        "A Scientist Explains Why Your Cat Is A Jerk",
        "This Simple Trick Will Change The Way You Make Grilled Cheese Forever",
    ]

    print("Testing predictions on sample headlines:")
    print("-" * 60)

    for i, headline in enumerate(test_headlines, 1):
        pred_ctr, raw_response = predict_ctr(model, tokenizer, headline)
        print(f"\n{i}. Headline: {headline[:60]}...")
        print(f"   Raw response: '{raw_response}'")
        print(f"   Parsed CTR: {pred_ctr}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
