"""
Evaluate the fine-tuned DeepSeek-7B-Chat-LoRA model for CTR prediction.
With incremental saving for long-running evaluations.
"""

import torch
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
base_model_path = "deepseek-ai/deepseek-llm-7b-chat"
adapter_path = os.path.expanduser("~/large-data/ctr-deepseek-correct/final")
data_path = "/home/users/s1155227960/projects/AIBusinessLOLA/data/test.csv"
results_dir = "/home/users/s1155227960/projects/AIBusinessLOLA/results"
checkpoint_path = os.path.join(results_dir, "eval_checkpoint.csv")
os.makedirs(results_dir, exist_ok=True)


def load_model():
    """Load base model and merge LoRA adapter weights."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # Force single GPU
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
        {"role": "system", "content": "You are an editor tasked with choosing catchy headlines for articles. Catchy means the headline that is likely to generate more clicks. Please predict the corresponding CTR (Click Through Rate： i.e., clicks over impressions) of the headline, ranging from 0 to 1. ONLY give the NUMBER, nothing else!"},
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
            use_cache=True  # Enable KV cache for faster generation
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    try:
        pred_ctr = float(response.strip().split()[0])
        pred_ctr = max(0.0, min(1.0, pred_ctr))
    except (ValueError, IndexError):
        pred_ctr = 0.01
    
    return pred_ctr, response


def run_evaluation(model, tokenizer, test_df, checkpoint_path):
    """Run predictions with checkpoint saving."""
    # Load existing checkpoint if exists
    if os.path.exists(checkpoint_path):
        existing_df = pd.read_csv(checkpoint_path)
        start_idx = len(existing_df)
        results = existing_df.to_dict('list')
        print(f"Resuming from checkpoint: {start_idx} samples already processed")
    else:
        results = {
            'test_id': [], 'headline': [], 'actual_CTR': [],
            'predicted_CTR': [], 'raw_response': []
        }
        start_idx = 0
    
    print(f"Running predictions on {len(test_df)} samples (starting from {start_idx})...")
    
    save_every = 50  # Save checkpoint every 50 samples
    
    for idx in tqdm(range(start_idx, len(test_df)), initial=start_idx, total=len(test_df)):
        row = test_df.iloc[idx]
        pred_ctr, raw_response = predict_ctr(model, tokenizer, row['headline'])
        
        results['test_id'].append(row['test_id'])
        results['headline'].append(row['headline'])
        results['actual_CTR'].append(row['CTR'])
        results['predicted_CTR'].append(pred_ctr)
        results['raw_response'].append(raw_response)
        
        # Save checkpoint periodically
        if (idx + 1) % save_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
    
    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv(checkpoint_path, index=False)
    
    return results_df


def compute_metrics(results_df):
    """Compute evaluation metrics."""
    actual = results_df['actual_CTR'].values
    predicted = np.clip(results_df['predicted_CTR'].values, 0, 1)
    
    metrics = {
        'MSE': mean_squared_error(actual, predicted),
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted),
        'R2': r2_score(actual, predicted),
        'Pearson_r': pearsonr(actual, predicted)[0],
        'Spearman_r': spearmanr(actual, predicted)[0],
    }
    return metrics


def plot_results(results_df, metrics, save_dir):
    """Generate visualization plots."""
    actual = results_df['actual_CTR'].values
    predicted = np.clip(results_df['predicted_CTR'].values, 0, 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.scatter(actual, predicted, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    max_val = max(actual.max(), predicted.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    z = np.polyfit(actual, predicted, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, actual.max(), 100)
    ax.plot(x_line, p(x_line), 'g-', lw=2, alpha=0.8, 
            label=f'Regression (r={metrics["Pearson_r"]:.3f})')
    
    ax.set_xlabel('Actual CTR', fontsize=12)
    ax.set_ylabel('Predicted CTR', fontsize=12)
    ax.set_title('Predicted vs Actual CTR (DeepSeek-7B-LoRA)', fontsize=14)
    ax.legend(loc='upper left')
    
    textstr = '\n'.join([
        f'Pearson r: {metrics["Pearson_r"]:.4f}',
        f'Spearman r: {metrics["Spearman_r"]:.4f}',
        f'RMSE: {metrics["RMSE"]:.6f}',
        f'MAE: {metrics["MAE"]:.6f}',
        f'R²: {metrics["R2"]:.4f}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    ax = axes[1]
    errors = predicted - actual
    ax.hist(errors, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
    ax.axvline(x=errors.mean(), color='green', linestyle='-', lw=2, 
               label=f'Mean Error: {errors.mean():.6f}')
    ax.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Prediction Errors', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'deepseek_ctr_prediction_evaluation.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Best arm identification
    fig, ax = plt.subplots(figsize=(10, 6))
    
    correct_best = 0
    total_tests = 0
    top3_correct = 0
    
    for test_id in results_df['test_id'].unique():
        test_data = results_df[results_df['test_id'] == test_id]
        if len(test_data) > 1:
            actual_best = test_data.loc[test_data['actual_CTR'].idxmax(), 'headline']
            pred_best_idx = test_data['predicted_CTR'].idxmax()
            pred_best = test_data.loc[pred_best_idx, 'headline']
            
            if actual_best == pred_best:
                correct_best += 1
            
            top3_actual = test_data.nlargest(3, 'actual_CTR')['headline'].tolist()
            if pred_best in top3_actual:
                top3_correct += 1
            
            total_tests += 1
    
    best_arm_acc = correct_best / total_tests if total_tests > 0 else 0
    top3_acc = top3_correct / total_tests if total_tests > 0 else 0
    
    methods = ['Best Arm\n(Top-1)', 'Top-3\nAccuracy']
    accuracies = [best_arm_acc, top3_acc]
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='white', alpha=0.8)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Best Arm Identification Performance (DeepSeek-7B-LoRA)', fontsize=14)
    ax.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'deepseek_best_arm_identification.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}")
    
    return best_arm_acc, top3_acc


def main():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("Fine-tuned DeepSeek-7B-Chat-LoRA Model Evaluation")
    print("=" * 60)
    
    print("\nLoading test data...")
    test_df = pd.read_csv(data_path)
    print(f"Total test samples: {len(test_df)}")
    
    # Sample 10%
    test_df = test_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print(f"Sampled 10%: {len(test_df)} examples")
    
    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded successfully!")
    
    print("\nRunning evaluation...")
    results_df = run_evaluation(model, tokenizer, test_df, checkpoint_path)
    
    # Copy final results
    predictions_path = os.path.join(results_dir, "test_predictions_deepseek_lora.csv")
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    print("\nComputing metrics...")
    metrics = compute_metrics(results_df)
    
    print("\nGenerating visualizations...")
    best_arm_acc, top3_acc = plot_results(results_df, metrics, results_dir)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key:<25} {value:.6f}")
    
    print(f"\n{'Best Arm Accuracy':<25} {best_arm_acc:.2%}")
    print(f"{'Top-3 Accuracy':<25} {top3_acc:.2%}")
    print("-" * 40)
    
    metrics_path = os.path.join(results_dir, "deepseek_evaluation_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Fine-tuned DeepSeek-7B-Chat-LoRA Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write(f"\nBest Arm Accuracy: {best_arm_acc:.2%}\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.2%}\n")
    
    print(f"\nMetrics saved to {metrics_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
