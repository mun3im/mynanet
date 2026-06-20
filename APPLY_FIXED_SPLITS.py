#!/usr/bin/env python3
"""
Apply fixed train/val/test directory structure to all model files.
This script surgically replaces the data loading section.
"""

import os
import sys

# The new data loading code to insert
NEW_DATA_LOADING = '''    # Count total files for progress bar (from fixed directories)
    total_files = 0

    for split in ['test', 'val', 'train']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir) and not class_name.startswith('.'):
                    files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                    total_files += len(files)

    # Add augmented samples to total if enabled (not for mixup)
    train_dir = os.path.join(data_dir, 'train')
    train_count = 0
    if os.path.exists(train_dir):
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.isdir(class_dir) and not class_name.startswith('.'):
                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                train_count += len(files)

    if augmentation_mode in ['baseline', 'specaugment']:
        total_files += train_count

    print(f"\\nDataset Structure: Fixed 75:10:15 split from directories")
    print(f"  Dataset root: {data_dir}")
    print(f"  Test dir:  {os.path.join(data_dir, 'test')}")
    print(f"  Val dir:   {os.path.join(data_dir, 'val')}")
    print(f"  Train dir: {os.path.join(data_dir, 'train')}")

    print(f"\\nAugmentation Strategy:")
    if augmentation_mode == 'baseline':
        print(f"  Train: baseline augmentation (time/pitch shift)")
    elif augmentation_mode == 'mixup':
        print(f"  Train: mixup (alpha={mixup_alpha})")
    elif augmentation_mode == 'specaugment':
        print(f"  Train: SpecAugment (freq/time masking)")
    else:
        print(f"  Train: no augmentation")
    print(f"  Total samples to process: {total_files}")

    with tqdm(total=total_files, desc="Loading from fixed directories", unit="file") as pbar:
        # Load TEST set (never augmented)
        test_dir = os.path.join(data_dir, 'test')
        if os.path.exists(test_dir):
            for class_name in sorted(os.listdir(test_dir)):
                class_dir = os.path.join(test_dir, class_name)
                if not os.path.isdir(class_dir) or class_name.startswith('.'):
                    continue

                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                for f in files:
                    try:
                        audio, _ = librosa.load(os.path.join(class_dir, f), sr=TARGET_SR)
                        audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \\
                            np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                        spec = compute_spec(audio, TARGET_SR, gmin, gmax)
                        X_test.append(spec)
                        y_test.append(labels[class_name])
                        test_paths.append(os.path.join(class_dir, f))
                        pbar.update(1)
                    except Exception as e:
                        failed_files.append(f"{os.path.join(class_dir, f)}: {str(e)}")
                        pbar.update(1)
                        continue

        # Load VALIDATION set (never augmented)
        val_dir = os.path.join(data_dir, 'val')
        if os.path.exists(val_dir):
            for class_name in sorted(os.listdir(val_dir)):
                class_dir = os.path.join(val_dir, class_name)
                if not os.path.isdir(class_dir) or class_name.startswith('.'):
                    continue

                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                for f in files:
                    try:
                        audio, _ = librosa.load(os.path.join(class_dir, f), sr=TARGET_SR)
                        audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \\
                            np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                        spec = compute_spec(audio, TARGET_SR, gmin, gmax)
                        X_val.append(spec)
                        y_val.append(labels[class_name])
                        val_paths.append(os.path.join(class_dir, f))
                        pbar.update(1)
                    except Exception as e:
                        failed_files.append(f"{os.path.join(class_dir, f)}: {str(e)}")
                        pbar.update(1)
                        continue

        # Load TRAINING set (with optional augmentation)
        train_dir = os.path.join(data_dir, 'train')
        if os.path.exists(train_dir):
            for class_name in sorted(os.listdir(train_dir)):
                class_dir = os.path.join(train_dir, class_name)
                if not os.path.isdir(class_dir) or class_name.startswith('.'):
                    continue

                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                for f in files:
                    try:
                        audio, _ = librosa.load(os.path.join(class_dir, f), sr=TARGET_SR)
                        audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \\
                            np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                        # Original sample
                        spec = compute_spec(audio, TARGET_SR, gmin, gmax)
                        X_train.append(spec)
                        y_train.append(labels[class_name])
                        train_paths.append(os.path.join(class_dir, f))
                        pbar.update(1)

                        # Augmented sample (mode-dependent)
                        if augmentation_mode == 'baseline':
                            aug_audio = augment_baseline(audio, TARGET_SR, time_shift_ms, pitch_shift_steps)
                            aug_spec = compute_spec(aug_audio, TARGET_SR, gmin, gmax)
                            X_train.append(aug_spec)
                            y_train.append(labels[class_name])
                            train_paths.append(os.path.join(class_dir, f) + "_aug")
                            pbar.update(1)
                        elif augmentation_mode == 'specaugment':
                            aug_spec = augment_specaugment(spec)
                            X_train.append(aug_spec)
                            y_train.append(labels[class_name])
                            train_paths.append(os.path.join(class_dir, f) + "_specaug")
                            pbar.update(1)

                    except Exception as e:
                        failed_files.append(f"{os.path.join(class_dir, f)}: {str(e)}")
                        if augmentation_mode in ['baseline', 'specaugment']:
                            pbar.update(2)
                        else:
                            pbar.update(1)
                        continue
'''

def fix_file(filepath):
    """Fix a single model file."""
    print(f"Processing {filepath}...")

    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found, skipping")
        return False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the data loading section
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if '# Count total files for progress bar' in line:
            start_idx = i
        if start_idx and '# Convert to numpy arrays' in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        print(f"  ⚠️  Could not find data loading section, skipping")
        return False

    # Replace the section
    new_lines = lines[:start_idx] + [NEW_DATA_LOADING + '\n'] + lines[end_idx:]

    # Update docstring
    for i in range(min(20, len(new_lines))):
        if 'Data Split:' in new_lines[i] and '90 test' in new_lines[i]:
            new_lines[i] = new_lines[i].replace(
                'Data Split: 90 test / 60 val / 450 train per class (600 total)',
                'Data Split: Fixed 75:10:15 (train/val/test) from dataset directories'
            )

    # Comment out the deprecated constants
    for i in range(len(new_lines)):
        if new_lines[i].startswith('TEST_SIZE_PER_CLASS = '):
            new_lines[i] = '# DEPRECATED: ' + new_lines[i]
        if new_lines[i].startswith('VAL_SIZE_PER_CLASS = '):
            new_lines[i] = '# DEPRECATED: ' + new_lines[i]

    # Write back
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

    print(f"  ✅ Fixed successfully")
    return True

def main():
    models = [
        "1_baseline_2dcnn.py",
        "2a_depthwise_cnn.py",
        "2b_mobilenetv3_64x300.py",
        "2c_mobilenetv3_pretrained_224x224.py",
        "3a_transformer_encoder.py",
        "4a_tcn_baseline.py",
        "4b_tcn_shallow.py",
        "4c_tcn_wide.py",
        "4d_tcn_deep.py",
        "4e_tcn_kernel2.py",
        "4f_tcn_kernel5.py",
        "4g_tcn_no_residual.py",
        "4h_tcn_lightweight.py",
        "4j_tcn_optimized.py",
    ]

    print("=" * 70)
    print("Applying fixed train/val/test directory structure")
    print("Dataset: /Volumes/Evo/seabird16k")
    print("Structure: train/ val/ test/ subdirectories (75:10:15)")
    print("=" * 70)
    print()

    # Create backup
    backup_dir = "backup_before_fixed_splits"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"Creating backup in {backup_dir}/...")
        for model in models:
            if os.path.exists(model):
                os.system(f"cp {model} {backup_dir}/")
        print("✓ Backup complete\n")

    # Fix all files
    fixed_count = 0
    for model in models:
        if fix_file(model):
            fixed_count += 1

    print()
    print("=" * 70)
    print(f"✅ Complete! Fixed {fixed_count}/{len(models)} files")
    print("=" * 70)
    print()
    print("All models now load from fixed train/val/test directories.")
    print("Ready for execution!")
    print()

if __name__ == "__main__":
    main()
