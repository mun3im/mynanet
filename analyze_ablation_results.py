#!/usr/bin/env python3
"""
Analyze ablation study results and perform statistical significance testing.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, binomtest

def extract_metrics_from_report(report_path):
    """Extract metrics from training_report.txt file."""
    metrics = {}

    if not os.path.exists(report_path):
        return None

    with open(report_path, 'r') as f:
        content = f.read()

    # Extract test accuracy (look for first occurrence which is FP32)
    acc_match = re.search(r'Test Accuracy:\s+([\d.]+)%?', content)
    if acc_match:
        acc_value = float(acc_match.group(1))
        # If value is > 1, it's already a percentage, otherwise convert
        metrics['test_accuracy'] = acc_value if acc_value > 1 else acc_value * 100

    # Extract TFLite metrics - look for INT8 section
    # Find all test accuracy values and take the second one (INT8)
    all_acc = re.findall(r'Test Accuracy:\s+([\d.]+)%?', content)
    if len(all_acc) >= 2:
        tflite_value = float(all_acc[1])
        metrics['tflite_accuracy'] = tflite_value if tflite_value > 1 else tflite_value * 100
    elif len(all_acc) == 1:
        # Fallback: use the same as test accuracy
        metrics['tflite_accuracy'] = metrics.get('test_accuracy', 0)

    # Extract model size
    size_match = re.search(r'Model size:\s+([\d.]+)\s*KB', content)
    if size_match:
        metrics['model_size_kb'] = float(size_match.group(1))

    # Extract F1 score from classification report files
    dir_path = os.path.dirname(report_path)

    # Try to read FP32 classification report
    fp32_report = os.path.join(dir_path, 'classification_report_fp32.txt')
    if os.path.exists(fp32_report):
        with open(fp32_report, 'r') as f:
            fp32_content = f.read()
        # Look for weighted avg line
        f1_match = re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+', fp32_content)
        if f1_match:
            metrics['precision'] = float(f1_match.group(1))
            metrics['recall'] = float(f1_match.group(2))
            metrics['f1_score'] = float(f1_match.group(3))

    # Try to read INT8 classification report for TFLite F1
    int8_report = os.path.join(dir_path, 'classification_report_int8.txt')
    if os.path.exists(int8_report):
        with open(int8_report, 'r') as f:
            int8_content = f.read()
        # Look for weighted avg line
        f1_match = re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+', int8_content)
        if f1_match:
            metrics['tflite_f1_score'] = float(f1_match.group(3))

    return metrics

def parse_directory_name(dir_name):
    """Parse configuration from directory name."""
    config = {}

    # Extract dropout
    drop_match = re.search(r'drop(\d+)', dir_name)
    if drop_match:
        config['dropout'] = int(drop_match.group(1)) / 100.0

    # Extract augmentation
    if 'mixup' in dir_name:
        config['augmentation'] = 'mixup'
    elif 'specaugment' in dir_name:
        config['augmentation'] = 'specaugment'
    elif 'baseline' in dir_name:
        config['augmentation'] = 'baseline'
    else:
        config['augmentation'] = 'none'

    # Extract random seed
    seed_match = re.search(r'rand(\d+)', dir_name)
    if seed_match:
        config['seed'] = int(seed_match.group(1))

    return config

def load_all_results():
    """Load results from all directories."""
    results = []

    # Find all results directories
    dirs = [d for d in os.listdir('.') if d.startswith('results/4a_')]

    for dir_name in dirs:
        report_path = os.path.join(dir_name, 'training_report.txt')
        metrics = extract_metrics_from_report(report_path)

        if metrics:
            config = parse_directory_name(dir_name)
            result = {**config, **metrics, 'directory': dir_name}
            results.append(result)

    return pd.DataFrame(results)

def load_predictions(dir_name):
    """Load predictions from a results directory."""
    # Try to find prediction file
    pred_files = [
        os.path.join(dir_name, 'predictions.npy'),
        os.path.join(dir_name, 'test_predictions.npy'),
    ]

    for pred_file in pred_files:
        if os.path.exists(pred_file):
            return np.load(pred_file)

    return None

def load_true_labels(dir_name):
    """Load true labels from a results directory."""
    label_files = [
        os.path.join(dir_name, 'true_labels.npy'),
        os.path.join(dir_name, 'test_labels.npy'),
    ]

    for label_file in label_files:
        if os.path.exists(label_file):
            return np.load(label_file)

    return None

def mcnemar_test(pred1, pred2, true_labels):
    """Perform McNemar's test between two models."""
    # Create contingency table
    # Table: [[both_correct, model1_correct_model2_wrong],
    #         [model1_wrong_model2_correct, both_wrong]]

    correct1 = (pred1 == true_labels)
    correct2 = (pred2 == true_labels)

    both_correct = np.sum(correct1 & correct2)
    both_wrong = np.sum(~correct1 & ~correct2)
    model1_only = np.sum(correct1 & ~correct2)
    model2_only = np.sum(~correct1 & correct2)

    # McNemar's test focuses on disagreements
    # Test statistic: (|b - c| - 1)^2 / (b + c)
    # where b = model1_only, c = model2_only

    n_disagreements = model1_only + model2_only

    if n_disagreements == 0:
        # No disagreements, models are identical
        return {
            'statistic': 0.0,
            'pvalue': 1.0,
            'model1_better': model1_only,
            'model2_better': model2_only,
            'both_correct': both_correct,
            'both_wrong': both_wrong
        }

    # Use exact binomial test for small samples, chi-square for large
    if n_disagreements < 25:
        # Exact test: binomial test
        pvalue = 2 * min(binomtest(model1_only, n_disagreements, 0.5).pvalue,
                        binomtest(model2_only, n_disagreements, 0.5).pvalue)
        statistic = abs(model1_only - model2_only)
    else:
        # Chi-square approximation with continuity correction
        statistic = (abs(model1_only - model2_only) - 1)**2 / n_disagreements
        pvalue = 1 - chi2.cdf(statistic, df=1)

    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'model1_better': model1_only,
        'model2_better': model2_only,
        'both_correct': both_correct,
        'both_wrong': both_wrong
    }

def analyze_results(df):
    """Perform comprehensive analysis."""
    print("=" * 80)
    print("ABLATION STUDY ANALYSIS")
    print("=" * 80)
    print()

    # Summary statistics
    print(f"Total experiments: {len(df)}")
    print(f"Unique dropout values: {sorted(df['dropout'].unique())}")
    print(f"Unique augmentations: {sorted(df['augmentation'].unique())}")
    print()

    # Overall statistics
    print("Overall Statistics:")
    print("-" * 80)
    metrics = ['test_accuracy', 'f1_score', 'tflite_accuracy']
    for metric in metrics:
        if metric in df.columns:
            print(f"{metric:20s}: {df[metric].mean():.4f} ± {df[metric].std():.4f} "
                  f"(range: {df[metric].min():.4f} - {df[metric].max():.4f})")
    print()

    # Top 10 configurations
    print("Top 10 Configurations (by TFLite Accuracy):")
    print("-" * 80)
    top_configs = df.nlargest(10, 'tflite_accuracy')
    for idx, row in top_configs.iterrows():
        print(f"Rank {list(top_configs.index).index(idx) + 1}:")
        print(f"  Dropout: {row['dropout']:.2f}, Augmentation: {row['augmentation']}")
        print(f"  Test Acc: {row['test_accuracy']:.4f}, F1: {row['f1_score']:.4f}, "
              f"TFLite Acc: {row['tflite_accuracy']:.4f}")
        print(f"  Directory: {row['directory']}")
        print()

    # Effect of dropout
    print("Effect of Dropout:")
    print("-" * 80)
    dropout_effect = df.groupby('dropout')[['test_accuracy', 'f1_score', 'tflite_accuracy']].agg(['mean', 'std'])
    print(dropout_effect)
    print()

    # Effect of augmentation
    print("Effect of Augmentation:")
    print("-" * 80)
    aug_effect = df.groupby('augmentation')[['test_accuracy', 'f1_score', 'tflite_accuracy']].agg(['mean', 'std'])
    print(aug_effect)
    print()

    # Interaction effects
    print("Interaction Effects (Dropout × Augmentation):")
    print("-" * 80)
    interaction = df.groupby(['dropout', 'augmentation'])['tflite_accuracy'].mean().unstack()
    print(interaction)
    print()

    return top_configs

def perform_statistical_tests(df, top_configs):
    """Perform McNemar's test on top configurations."""
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING (McNemar's Test)")
    print("=" * 80)
    print()

    # Try to load predictions for top 5 configs
    top_5 = top_configs.head(5)

    predictions = {}
    true_labels = None

    for idx, row in top_5.iterrows():
        dir_name = row['directory']
        pred = load_predictions(dir_name)
        labels = load_true_labels(dir_name)

        if pred is not None:
            predictions[dir_name] = pred
            if true_labels is None and labels is not None:
                true_labels = labels

    if len(predictions) < 2:
        print("WARNING: Could not find prediction files for statistical testing.")
        print("Predictions should be saved as 'predictions.npy' or 'test_predictions.npy'")
        print("True labels should be saved as 'true_labels.npy' or 'test_labels.npy'")
        print()
        return

    # Perform pairwise McNemar's tests
    config_names = list(predictions.keys())
    n_configs = len(config_names)

    print(f"Comparing {n_configs} configurations pairwise:")
    print()

    results_matrix = np.zeros((n_configs, n_configs))

    for i, config1 in enumerate(config_names):
        for j, config2 in enumerate(config_names):
            if i >= j:
                continue

            pred1 = predictions[config1]
            pred2 = predictions[config2]

            result = mcnemar_test(pred1, pred2, true_labels)
            results_matrix[i, j] = result['pvalue']

            # Extract config details
            row1 = df[df['directory'] == config1].iloc[0]
            row2 = df[df['directory'] == config2].iloc[0]

            print(f"Config {i+1} vs Config {j+1}:")
            print(f"  {config1}")
            print(f"    Dropout: {row1['dropout']:.2f}, Aug: {row1['augmentation']}, "
                  f"TFLite Acc: {row1['tflite_accuracy']:.4f}")
            print(f"  {config2}")
            print(f"    Dropout: {row2['dropout']:.2f}, Aug: {row2['augmentation']}, "
                  f"TFLite Acc: {row2['tflite_accuracy']:.4f}")
            print(f"  McNemar's p-value: {result['pvalue']:.4f}")

            if result['pvalue'] < 0.05:
                if result['model1_better'] > result['model2_better']:
                    print(f"  *** Config {i+1} is significantly better (p < 0.05) ***")
                else:
                    print(f"  *** Config {j+1} is significantly better (p < 0.05) ***")
            else:
                print(f"  No significant difference (p >= 0.05)")

            print(f"  Disagreements: Config {i+1} better: {result['model1_better']}, "
                  f"Config {j+1} better: {result['model2_better']}")
            print()

def generate_recommendations(df, top_configs):
    """Generate recommendations based on analysis."""
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    best_config = top_configs.iloc[0]

    print("1. BEST CONFIGURATION:")
    print(f"   Dropout: {best_config['dropout']:.2f}")
    print(f"   Augmentation: {best_config['augmentation']}")
    print(f"   Test Accuracy: {best_config['test_accuracy']:.4f}")
    print(f"   F1 Score: {best_config['f1_score']:.4f}")
    print(f"   TFLite Accuracy: {best_config['tflite_accuracy']:.4f}")
    if 'model_size_kb' in best_config and pd.notna(best_config['model_size_kb']):
        print(f"   Model Size: {best_config['model_size_kb']:.2f} KB")
    print()

    # Analyze dropout effect
    best_dropout = df.groupby('dropout')['tflite_accuracy'].mean().idxmax()
    print(f"2. OPTIMAL DROPOUT RATE: {best_dropout:.2f}")
    dropout_stats = df.groupby('dropout')['tflite_accuracy'].agg(['mean', 'std'])
    print("   Dropout analysis:")
    for dropout, row in dropout_stats.iterrows():
        print(f"     {dropout:.2f}: {row['mean']:.4f} ± {row['std']:.4f}")
    print()

    # Analyze augmentation effect
    best_aug = df.groupby('augmentation')['tflite_accuracy'].mean().idxmax()
    print(f"3. OPTIMAL AUGMENTATION: {best_aug}")
    aug_stats = df.groupby('augmentation')['tflite_accuracy'].agg(['mean', 'std'])
    print("   Augmentation analysis:")
    for aug, row in aug_stats.iterrows():
        print(f"     {aug:15s}: {row['mean']:.4f} ± {row['std']:.4f}")
    print()

    # Variance analysis
    print("4. VARIANCE ANALYSIS:")
    print(f"   Overall TFLite accuracy std: {df['tflite_accuracy'].std():.4f}")
    print(f"   This represents variation across configurations.")
    print(f"   Running multiple seeds will estimate within-configuration variance.")
    print()

    # Top 5 for further testing
    print("5. TOP 5 CONFIGURATIONS FOR MULTI-SEED TESTING:")
    for i, (idx, row) in enumerate(top_configs.head(5).iterrows(), 1):
        print(f"   {i}. Dropout: {row['dropout']:.2f}, Aug: {row['augmentation']}, "
              f"TFLite Acc: {row['tflite_accuracy']:.4f}")
    print()

    return top_configs.head(5)

def create_visualizations(df):
    """Create visualization plots."""
    print("Creating visualizations...")

    # Set style
    sns.set_style("whitegrid")

    # 1. Heatmap of dropout × augmentation
    plt.figure(figsize=(10, 6))
    interaction = df.groupby(['dropout', 'augmentation'])['tflite_accuracy'].mean().unstack()
    sns.heatmap(interaction, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'TFLite Accuracy'})
    plt.title('TFLite Accuracy: Dropout × Augmentation')
    plt.xlabel('Augmentation')
    plt.ylabel('Dropout Rate')
    plt.tight_layout()
    plt.savefig('ablation_heatmap.png', dpi=300, bbox_inches='tight')
    print("  Saved: ablation_heatmap.png")

    # 2. Box plot for dropout effect
    plt.figure(figsize=(10, 6))
    df_plot = df.copy()
    # Sort dropout values numerically and create ordered categories
    dropout_order = sorted(df_plot['dropout'].unique())
    df_plot['dropout'] = pd.Categorical(df_plot['dropout'].astype(str),
                                        categories=[str(d) for d in dropout_order],
                                        ordered=True)
    sns.boxplot(data=df_plot, x='dropout', y='tflite_accuracy', palette='Set2')
    plt.title('Effect of Dropout on TFLite Accuracy')
    plt.xlabel('Dropout Rate')
    plt.ylabel('TFLite Accuracy')
    plt.tight_layout()
    plt.savefig('dropout_effect.png', dpi=300, bbox_inches='tight')
    print("  Saved: dropout_effect.png")

    # 3. Box plot for augmentation effect
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='augmentation', y='tflite_accuracy', palette='Set3')
    plt.title('Effect of Augmentation on TFLite Accuracy')
    plt.xlabel('Augmentation Method')
    plt.ylabel('TFLite Accuracy')
    plt.tight_layout()
    plt.savefig('augmentation_effect.png', dpi=300, bbox_inches='tight')
    print("  Saved: augmentation_effect.png")

    # 4. Top 10 configurations
    plt.figure(figsize=(12, 6))
    top_10 = df.nlargest(10, 'tflite_accuracy')
    top_10['config'] = top_10.apply(lambda x: f"D{x['dropout']:.2f}_{x['augmentation']}", axis=1)
    plt.barh(range(len(top_10)), top_10['tflite_accuracy'])
    plt.yticks(range(len(top_10)), top_10['config'])
    plt.xlabel('TFLite Accuracy')
    plt.title('Top 10 Configurations')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('top_10_configs.png', dpi=300, bbox_inches='tight')
    print("  Saved: top_10_configs.png")

    print()

def save_results_csv(df, top_5):
    """Save results to CSV files."""
    # Save all results
    df_sorted = df.sort_values('tflite_accuracy', ascending=False)
    df_sorted.to_csv('ablation_all_results.csv', index=False)
    print("Saved: ablation_all_results.csv")

    # Save top 5
    top_5.to_csv('ablation_top5_configs.csv', index=False)
    print("Saved: ablation_top5_configs.csv")
    print()

def main():
    """Main analysis function."""
    # Load all results
    print("Loading results from all directories...")
    df = load_all_results()
    print(f"Loaded {len(df)} experiments\n")

    # Perform analysis
    top_configs = analyze_results(df)

    # Statistical testing
    perform_statistical_tests(df, top_configs)

    # Generate recommendations
    top_5 = generate_recommendations(df, top_configs)

    # Create visualizations
    create_visualizations(df)

    # Save results
    save_results_csv(df, top_5)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - ablation_all_results.csv")
    print("  - ablation_top5_configs.csv")
    print("  - ablation_heatmap.png")
    print("  - dropout_effect.png")
    print("  - augmentation_effect.png")
    print("  - top_10_configs.png")
    print()

    return df, top_5

if __name__ == "__main__":
    main()
