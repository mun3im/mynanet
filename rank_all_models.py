#!/usr/bin/env python3
"""
Rank all trained models by test accuracy
Parse training_report.txt files from results_linux and results_macos
"""

import re
import os
from pathlib import Path
from collections import defaultdict

def extract_metrics(report_path):
    """Extract key metrics from training_report.txt"""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metrics = {
            'path': str(report_path),
            'dir': os.path.dirname(report_path),
            'fp32_acc': None,
            'int8_acc': None,
            'best_val_acc': None,
            'train_time': None,
            'model_size_kb': None,
            'dataset': 'seabird',  # default
            'split_ratio': None,
            'warmup_epochs': None,
            'total_samples': None,
            'platform': None,
        }

        # Extract dataset info
        if 'MYGARDENBIRD' in content or 'mygardenbird' in content.lower():
            metrics['dataset'] = 'mygardenbird'

        # Extract split ratio
        split_match = re.search(r'Training:\s+(\d+)\s+\((\d+\.\d+)%\)\s+Validation:\s+(\d+)\s+\((\d+\.\d+)%\)\s+Test:\s+(\d+)\s+\((\d+\.\d+)%\)', content)
        if split_match:
            train_pct = float(split_match.group(2))
            val_pct = float(split_match.group(4))
            test_pct = float(split_match.group(6))
            metrics['split_ratio'] = f"{int(train_pct)}/{int(val_pct)}/{int(test_pct)}"

        # Extract total samples
        samples_match = re.search(r'Total Samples:\s+(\d+)', content)
        if samples_match:
            metrics['total_samples'] = int(samples_match.group(1))

        # Extract platform
        platform_match = re.search(r'Platform:\s+(\w+)\s+(\w+)', content)
        if platform_match:
            metrics['platform'] = platform_match.group(1)

        # Extract FP32 accuracy
        fp32_match = re.search(r'FP32 \(\.keras\).*?Test Accuracy:\s+([\d.]+)%', content, re.DOTALL)
        if fp32_match:
            metrics['fp32_acc'] = float(fp32_match.group(1))
        else:
            # Try alternative format
            fp32_match = re.search(r'FP32 \(\.keras\)\s*:\s*([\d.]+)%', content)
            if fp32_match:
                metrics['fp32_acc'] = float(fp32_match.group(1))

        # Extract INT8 accuracy - try multiple patterns
        int8_match = re.search(r'INT8 TFLite Evaluation:.*?Test Accuracy:\s+([\d.]+)%', content, re.DOTALL)
        if int8_match:
            metrics['int8_acc'] = float(int8_match.group(1))
        else:
            # Try alternative format (older reports)
            int8_match = re.search(r'INT8 \(TFLite\).*?Test Accuracy:\s+([\d.]+)%', content, re.DOTALL)
            if int8_match:
                metrics['int8_acc'] = float(int8_match.group(1))
            else:
                # Try yet another format
                int8_match = re.search(r'INT8 \(\.tflite\|TFLite\)\s*:\s*([\d.]+)%', content)
                if int8_match:
                    metrics['int8_acc'] = float(int8_match.group(1))

        # Extract best validation accuracy
        val_match = re.search(r'Best (?:Finetune )?Val Acc:\s+([\d.]+)%', content)
        if val_match:
            metrics['best_val_acc'] = float(val_match.group(1))

        # Extract training time
        time_match = re.search(r'Total Duration:\s+([\dh\sm]+)', content)
        if time_match:
            metrics['train_time'] = time_match.group(1).strip()

        # Extract model size
        size_match = re.search(r'INT8 \(\.tflite\)\s*:\s*([\d.]+)\s*KB', content)
        if size_match:
            metrics['model_size_kb'] = float(size_match.group(1))

        # Extract warmup epochs
        warmup_match = re.search(r'Warmup Epochs:\s+(\d+)', content)
        if warmup_match:
            metrics['warmup_epochs'] = int(warmup_match.group(1))

        return metrics

    except Exception as e:
        print(f"Error parsing {report_path}: {e}")
        return None

def main():
    # Find all training reports
    results_dirs = ['results_linux', 'results_macos', 'result_macos']
    all_reports = []

    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            for root, dirs, files in os.walk(results_dir):
                if 'training_report.txt' in files:
                    report_path = os.path.join(root, 'training_report.txt')
                    all_reports.append(report_path)

    print(f"\n{'='*100}")
    print(f"Found {len(all_reports)} training reports")
    print(f"{'='*100}\n")

    # Parse all reports
    results = []
    for report_path in all_reports:
        metrics = extract_metrics(report_path)
        if metrics and metrics['fp32_acc'] is not None:
            results.append(metrics)

    # Sort by FP32 test accuracy (descending)
    results.sort(key=lambda x: x['fp32_acc'] if x['fp32_acc'] else 0, reverse=True)

    # Print rankings
    print(f"\n{'='*100}")
    print(f"COMPLETE MODEL RANKING (by FP32 Test Accuracy)")
    print(f"{'='*100}\n")
    print(f"{'Rank':<6} {'FP32%':<8} {'INT8%':<8} {'ValAcc%':<8} {'Dataset':<12} {'Split':<10} {'WarmEp':<8} {'Size(KB)':<10} {'Directory':<80}")
    print(f"{'-'*100}")

    # Track Stage9c result - match by unique characteristics (warm100 + mixup0.25)
    stage9c_rank = None
    stage9c_result = None

    for rank, result in enumerate(results, 1):
        dir_name = os.path.basename(result['dir'])
        fp32 = f"{result['fp32_acc']:.2f}" if result['fp32_acc'] else "N/A"
        int8 = f"{result['int8_acc']:.2f}" if result['int8_acc'] else "N/A"
        val = f"{result['best_val_acc']:.2f}" if result['best_val_acc'] else "N/A"
        dataset = result['dataset']
        split = result['split_ratio'] if result['split_ratio'] else "N/A"
        warmup = str(result['warmup_epochs']) if result['warmup_epochs'] else "N/A"
        size = f"{result['model_size_kb']:.1f}" if result['model_size_kb'] else "N/A"

        # Highlight Stage9c result - identify by warm100 and mixup0.25
        marker = ""
        if 'warm100' in dir_name and 'mixup0.25' in dir_name and dataset == 'mygardenbird':
            marker = " *** STAGE9C (MyGardenBird) ***"
            stage9c_rank = rank
            stage9c_result = result

        print(f"{rank:<6} {fp32:<8} {int8:<8} {val:<8} {dataset:<12} {split:<10} {warmup:<8} {size:<10} {dir_name[:80]}{marker}")

    # Summary statistics
    print(f"\n{'='*100}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*100}\n")

    # Group by dataset
    by_dataset = defaultdict(list)
    for result in results:
        by_dataset[result['dataset']].append(result['fp32_acc'])

    for dataset, accs in sorted(by_dataset.items()):
        avg_acc = sum(accs) / len(accs)
        max_acc = max(accs)
        min_acc = min(accs)
        print(f"{dataset.upper():<15} Count: {len(accs):<3}  Avg: {avg_acc:.2f}%  Max: {max_acc:.2f}%  Min: {min_acc:.2f}%")

    # Group by split ratio
    print(f"\n{'-'*100}")
    print(f"BY SPLIT RATIO:")
    print(f"{'-'*100}\n")

    by_split = defaultdict(list)
    for result in results:
        if result['split_ratio']:
            by_split[result['split_ratio']].append(result['fp32_acc'])

    for split, accs in sorted(by_split.items()):
        avg_acc = sum(accs) / len(accs)
        max_acc = max(accs)
        min_acc = min(accs)
        print(f"{split:<15} Count: {len(accs):<3}  Avg: {avg_acc:.2f}%  Max: {max_acc:.2f}%  Min: {min_acc:.2f}%")

    # Stage9c comparison
    if stage9c_rank and stage9c_result:
        print(f"\n{'='*100}")
        print(f"STAGE9C ANALYSIS")
        print(f"{'='*100}\n")

        stage9c = stage9c_result
        print(f"Stage9c Rank:           #{stage9c_rank} out of {len(results)}")
        print(f"Dataset:                {stage9c['dataset'].upper()}")
        print(f"Test Accuracy (FP32):   {stage9c['fp32_acc']:.2f}%")
        int8_str = f"{stage9c['int8_acc']:.2f}%" if stage9c['int8_acc'] else "N/A"
        print(f"Test Accuracy (INT8):   {int8_str}")
        print(f"Split Ratio:            {stage9c['split_ratio']}")
        print(f"Total Samples:          {stage9c['total_samples']}")
        print(f"Warmup Epochs:          {stage9c['warmup_epochs']}")
        size_str = f"{stage9c['model_size_kb']:.1f} KB" if stage9c['model_size_kb'] else "N/A"
        print(f"Model Size:             {size_str}")

        # Compare to best seabird results
        seabird_results = [r for r in results if r['dataset'] == 'seabird']
        if seabird_results:
            best_seabird = seabird_results[0]
            print(f"\n{'-'*100}")
            print(f"COMPARISON TO BEST SEABIRD RESULT:")
            print(f"{'-'*100}\n")
            print(f"Best Seabird:           {best_seabird['fp32_acc']:.2f}% ({os.path.basename(best_seabird['dir'])})")
            print(f"Stage9c (MyGardenBird): {stage9c['fp32_acc']:.2f}%")
            print(f"Difference:             {stage9c['fp32_acc'] - best_seabird['fp32_acc']:.2f}%")

            if stage9c['fp32_acc'] > best_seabird['fp32_acc']:
                print(f"\n✓ Stage9c OUTPERFORMS best Seabird result!")
            elif stage9c['fp32_acc'] < best_seabird['fp32_acc']:
                print(f"\n✗ Stage9c UNDERPERFORMS best Seabird result by {best_seabird['fp32_acc'] - stage9c['fp32_acc']:.2f}%")
            else:
                print(f"\n= Stage9c MATCHES best Seabird result")

        # Compare to seabird 80/10/10 splits
        seabird_80_10_10 = [r for r in results if r['dataset'] == 'seabird' and r['split_ratio'] == '80/10/10']
        if seabird_80_10_10:
            avg_seabird_80 = sum(r['fp32_acc'] for r in seabird_80_10_10) / len(seabird_80_10_10)
            print(f"\n{'-'*100}")
            print(f"COMPARISON TO SEABIRD 80/10/10 SPLIT:")
            print(f"{'-'*100}\n")
            print(f"Seabird 80/10/10 Avg:   {avg_seabird_80:.2f}% (n={len(seabird_80_10_10)})")
            print(f"Stage9c (MyGardenBird): {stage9c['fp32_acc']:.2f}%")
            print(f"Difference:             {stage9c['fp32_acc'] - avg_seabird_80:.2f}%")

        # Impact of dataset change
        print(f"\n{'-'*100}")
        print(f"IMPACT OF DATASET CHANGE:")
        print(f"{'-'*100}\n")

        # Find seabird results with similar configuration
        similar_seabird = [r for r in results
                          if r['dataset'] == 'seabird'
                          and r['split_ratio'] == '80/10/10'
                          and 'v1_dscnn' in r['dir'].lower()]

        if similar_seabird:
            avg_similar = sum(r['fp32_acc'] for r in similar_seabird) / len(similar_seabird)
            print(f"Similar Seabird configs (v1_dscnn, 80/10/10): {avg_similar:.2f}% (n={len(similar_seabird)})")
            print(f"Stage9c (MyGardenBird, 80/10/10):             {stage9c['fp32_acc']:.2f}%")
            print(f"Gain from dataset change:                     {stage9c['fp32_acc'] - avg_similar:.2f}%")

            # Detail each similar result
            print(f"\nSimilar Seabird results:")
            for i, r in enumerate(similar_seabird, 1):
                print(f"  {i}. {r['fp32_acc']:.2f}% - {os.path.basename(r['dir'])}")

    print(f"\n{'='*100}\n")

if __name__ == "__main__":
    main()
