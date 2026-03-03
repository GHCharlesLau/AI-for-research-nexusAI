# Qwen3-8B Fine-tuning Datasets

This folder contains the training, validation, and test datasets for fine-tuning Qwen3-8B on the headline selection task.

## Data Format

Qwen3-8B uses the following JSONL format:

```json
{"instruction": "user instruction here", "output": "expected response here"}
```

## Files

| File | Examples | Description |
|------|----------|-------------|
| `train_qwen.jsonl` | 9900 | Training dataset |
| `validation_qwen.jsonl` | 1237 | Validation dataset for hyperparameter tuning |
| `test_qwen.jsonl` | 1239 | Test dataset for final evaluation |

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
{
  "instruction": "You are an editor tasked with choosing the catchier one from several drafted headlines for the same article. Catchier means the one that is likely to generate more clicks.\n\nYou are presented with several headlines. Which one is catchier? Return only the number before the headline.\n\n1. Headline A\n2. Headline B\n3. Headline C",
  "output": "3"
}
```

---
