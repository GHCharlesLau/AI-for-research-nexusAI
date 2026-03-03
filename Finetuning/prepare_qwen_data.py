#!/usr/bin/env python3
"""
Prepare Training and Test Datasets for Qwen3-8B Fine-tuning

This script:
1. Converts OpenAI format training data to Qwen3-8B format
2. Creates train/test split
3. Validates data format
4. Saves to data/ folder
"""

import json
import os
import random
from pathlib import Path
from datetime import datetime

# Configuration
INPUT_TRAIN_FILE = r'D:\Doctoral_study\CourseSelection\S4 Courses\DOTE6635 AIforBusiness\Assignments\PaperReplication\LLM_News\Finetune ChatGPT\train.jsonl'
OUTPUT_DIR = r'D:\Doctoral_study\CourseSelection\S4 Courses\DOTE6635 AIforBusiness\Assignments\PaperReplication\data'
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, 'train_qwen.jsonl')
TEST_OUTPUT = os.path.join(OUTPUT_DIR, 'test_qwen.jsonl')
VAL_OUTPUT = os.path.join(OUTPUT_DIR, 'validation_qwen.jsonl')

# Train/Val/Test split ratio
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Set random seed for reproducibility
random.seed(42)

def create_output_dir():
    """Create output directory if it doesn't exist"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created output directory: {OUTPUT_DIR}")

def convert_openai_to_qwen_format(openai_message):
    """
    Convert OpenAI ChatCompletion format to Qwen3-8B format

    OpenAI format:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Qwen format (Instruction-Output):
    {"instruction": user_content, "output": assistant_content}

    Qwen format (Messages - also supported):
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """

    messages = openai_message.get('messages', [])

    # Extract system prompt and messages
    system_content = ""
    user_content = ""
    assistant_content = ""

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'system':
            system_content = content
        elif role == 'user':
            user_content = content
        elif role == 'assistant':
            assistant_content = content

    # Method 1: Instruction-Output format (simpler, recommended for Qwen)
    instruction = f"{system_content}\n\n{user_content}" if system_content else user_content

    qwen_instruction_format = {
        "instruction": instruction,
        "output": assistant_content
    }

    # Method 2: Messages format (OpenAI compatible)
    qwen_messages_format = {
        "messages": messages
    }

    # Return instruction-output format (primary for Qwen)
    return qwen_instruction_format

def load_openai_data(file_path):
    """Load OpenAI format JSONL data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def split_data(data, train_ratio, val_ratio, test_ratio):
    """Split data into train/val/test sets"""
    n = len(data)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data

def save_qwen_format(data, output_path):
    """Save data in Qwen format JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            # Convert to Qwen format
            qwen_item = convert_openai_to_qwen_format(item)
            f.write(json.dumps(qwen_item, ensure_ascii=False) + '\n')

def validate_qwen_format(file_path):
    """Validate Qwen format JSONL file"""
    errors = []
    count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)

                # Check required fields
                if 'instruction' not in item:
                    errors.append(f"Line {line_num}: Missing 'instruction' field")
                if 'output' not in item:
                    errors.append(f"Line {line_num}: Missing 'output' field")

                # Validate field types
                if not isinstance(item.get('instruction'), str):
                    errors.append(f"Line {line_num}: 'instruction' must be string")
                if not isinstance(item.get('output'), str):
                    errors.append(f"Line {line_num}: 'output' must be string")

                count += 1

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error - {e}")

    return count, errors

def main():
    print("="*70)
    print("Qwen3-8B Fine-tuning Data Preparation")
    print("="*70)

    # Create output directory
    create_output_dir()

    # Load original OpenAI format data
    print(f"\n[Loading] Reading training data from: {INPUT_TRAIN_FILE}")
    openai_data = load_openai_data(INPUT_TRAIN_FILE)
    print(f"  Loaded {len(openai_data)} examples")

    # Split into train/val/test
    print(f"\n[Splitting] Dividing data into train/val/test sets")
    print(f"  Train: {TRAIN_RATIO*100:.0f}%")
    print(f"  Validation: {VAL_RATIO*100:.0f}%")
    print(f"  Test: {TEST_RATIO*100:.0f}%")

    train_data, val_data, test_data = split_data(openai_data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")

    # Save in Qwen format
    print(f"\n[Converting] Converting to Qwen3-8B format")

    print(f"  Saving training data to: {TRAIN_OUTPUT}")
    save_qwen_format(train_data, TRAIN_OUTPUT)

    print(f"  Saving validation data to: {VAL_OUTPUT}")
    save_qwen_format(val_data, VAL_OUTPUT)

    print(f"  Saving test data to: {TEST_OUTPUT}")
    save_qwen_format(test_data, TEST_OUTPUT)

    # Validate output files
    print(f"\n[Validating] Validating output files")

    train_count, train_errors = validate_qwen_format(TRAIN_OUTPUT)
    print(f"  Train: {train_count} examples, {len(train_errors)} errors")

    val_count, val_errors = validate_qwen_format(VAL_OUTPUT)
    print(f"  Validation: {val_count} examples, {len(val_errors)} errors")

    test_count, test_errors = validate_qwen_format(TEST_OUTPUT)
    print(f"  Test: {test_count} examples, {len(test_errors)} errors")

    if train_errors or val_errors or test_errors:
        print("\n[Errors found]")
        all_errors = train_errors + val_errors + test_errors
        for error in all_errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")
    else:
        print("  [OK] All files valid!")

    # Show sample data
    print(f"\n[Sample] First training example in Qwen format:")
    with open(TRAIN_OUTPUT, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        print(f"  Instruction (first 200 chars): {sample['instruction'][:200]}...")
        print(f"  Output: {sample['output']}")

    print("\n" + "="*70)
    print("Data Preparation Complete!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  Training:   {TRAIN_OUTPUT}")
    print(f"  Validation: {VAL_OUTPUT}")
    print(f"  Test:       {TEST_OUTPUT}")
    print(f"\nTotal: {train_count + val_count + test_count} examples")

    # Create README for data folder
    readme_content = f"""# Qwen3-8B Fine-tuning Datasets

This folder contains the training, validation, and test datasets for fine-tuning Qwen3-8B on the headline selection task.

## Data Format

Qwen3-8B uses the following JSONL format:

```json
{{"instruction": "user instruction here", "output": "expected response here"}}
```

## Files

| File | Examples | Description |
|------|----------|-------------|
| `train_qwen.jsonl` | {train_count} | Training dataset |
| `validation_qwen.jsonl` | {val_count} | Validation dataset for hyperparameter tuning |
| `test_qwen.jsonl` | {test_count} | Test dataset for final evaluation |

## Data Source

Original data: Upworthy news headline A/B testing dataset
- Source: https://osf.io/jd64p/
- CTR values: Real clicks/impressions from actual A/B tests
- Task: Select the headline with highest click-through rate

## Task Description

**Instruction:** Given multiple headlines for the same news article, select the one that is most likely to generate the highest click-through rate (CTR).

**Output:** The number (1, 2, 3...) corresponding to the selected headline.

## Usage with Aliyun PAI

1. Upload these files to OSS or NAS storage
2. In PAI Model Gallery, select Qwen3-8B model
3. Click "Train" and configure:
   - Training dataset: train_qwen.jsonl
   - Validation dataset: validation_qwen.jsonl
4. Submit training job

## Example

```json
{{
  "instruction": "You are an editor tasked with choosing the catchier one from several drafted headlines for the same article. Catchier means the one that is likely to generate more clicks.\\n\\nYou are presented with several headlines. Which one is catchier? Return only the number before the headline.\\n\\n1. Headline A\\n2. Headline B\\n3. Headline C",
  "output": "3"
}}
```

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    readme_path = os.path.join(OUTPUT_DIR, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"\n[OK] Created README: {readme_path}")

if __name__ == "__main__":
    main()
