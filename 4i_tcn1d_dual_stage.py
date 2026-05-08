import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_curve, auc
import seaborn as sns
import argparse
import platform

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
SAMPLE_RATE = 16000
DURATION = 3
AUDIO_LENGTH = SAMPLE_RATE * DURATION  # 48000
INPUT_SHAPE = (AUDIO_LENGTH, 1)

DENSE_UNITS = 256
LEARNING_RATE = 0.001

# Dataset and output paths
DATASET_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SPECTROGRAM_DIR = f"/Volumes/Evo/cache_mygardenbird_raw_audio"  # Unused now
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

# Argument parser
parser = argparse.ArgumentParser(description="TCN training for audio classification (two-stage).")
parser.add_argument("-t", "--transform", choices=["mels", "raw"], default="raw", help="Input type (only 'raw' supported now)")
parser.add_argument("--warmup_epochs", type=int, default=15, help="Warmup epochs")
parser.add_argument("--finetune_epochs", type=int, default=25, help="Fine-tuning epochs")
parser.add_argument("--warmup_lr", type=float, default=0.001, help="Warmup learning rate")
parser.add_argument("--finetune_lr", type=float, default=0.0001, help="Fine-tuning learning rate")
parser.add_argument("--class_weights", action="store_true", default=True, help="Enable class weights")
parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")
args = parser.parse_args()

# Updated output directory
OUTPUT_DIR = f"results_tcn_two_stage_{args.transform}_linux_v3_raw_1d_tcn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting TCN training with configuration:")
print(f" Model: Pure 1D Temporal CNN")
print(f" Transform: {args.transform}")
print(f" Input Shape: {INPUT_SHAPE}")
print(f" Class weights: {args.class_weights}")
print(f" Data augmentation: {not args.no_augmentation} (time-domain)")
print(f" Output directory: {OUTPUT_DIR}")


# -------------------------------
# 1. Data Preprocessing
# -------------------------------

def augment_audio(audio, sr=SAMPLE_RATE, noise_std=0.005, max_shift=2000):
    """Apply time-domain augmentation: noise, time shift."""
    # Add Gaussian noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, noise_std, audio.shape)
        audio = audio + noise

    # Time shift
    if np.random.rand() > 0.5:
        shift = np.random.randint(-max_shift, max_shift)
        audio = np.roll(audio, shift)
        if shift > 0:
            audio[:shift] = 0
        elif shift < 0:
            audio[shift:] = 0

    return audio


def preprocess_audio(audio_path, sr=SAMPLE_RATE, duration=DURATION, augment=False):
    try:
        # Load raw audio
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)

        # Trim or pad
        if len(audio) < AUDIO_LENGTH:
            pad_length = AUDIO_LENGTH - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        else:
            audio = audio[:AUDIO_LENGTH]

        # Augment
        if augment:
            audio = augment_audio(audio, sr=sr)

        # Shape: (48000,) -> (48000, 1)
        return audio.reshape(-1, 1).astype(np.float32)

    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {e}")
        return None


def load_and_preprocess_data(data_dir):
    X_all, y_all = [], []
    class_labels = {}
    class_index = 0

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        print(f"Processing class: {class_name}")
        class_samples = 0
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav') and not file_name.startswith('._'):
                audio_path = os.path.join(class_dir, file_name)
                # No augmentation during loading
                audio = preprocess_audio(audio_path, augment=False)
                if audio is not None:
                    X_all.append(audio)
                    if class_name not in class_labels:
                        class_labels[class_name] = class_index
                        class_index += 1
                    y_all.append(class_labels[class_name])
                    class_samples += 1
        print(f"Generated {class_samples} samples")

    print("Class distribution:")
    for k, v in class_labels.items():
        count = y_all.count(v)
        print(f"{k}: {count}")

    return np.array(X_all), np.array(y_all), class_labels


# -------------------------------
# 2. Pure 1D TCN Model
# -------------------------------

class TCNBlock(tf.keras.layers.Layer):
    def __init__(self, nb_filters, kernel_size=3, dilation_rate=1, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        # Define layers in __init__
        self.conv1 = tf.keras.layers.Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            name=f"{self.name}_conv1"
        )
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.dropout1 = tf.keras.layers.SpatialDropout1D(dropout_rate)

        self.conv2 = tf.keras.layers.Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            name=f"{self.name}_conv2"
        )
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        self.dropout2 = tf.keras.layers.SpatialDropout1D(dropout_rate)

        # Optional 1x1 conv for residual if channels don't match
        self.residual_conv = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels != self.nb_filters:
            self.residual_conv = tf.keras.layers.Conv1D(
                self.nb_filters, 1, padding='same', name=f"{self.name}_res_conv"
            )
            self.residual_conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs

        # First conv block
        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.act1(x)
        x = self.dropout1(x, training=training)

        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)
        x = self.dropout2(x, training=training)

        # Residual connection
        residual = inputs
        if self.residual_conv is not None:
            residual = self.residual_conv(residual, training=training)

        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation('relu')(x)
        return x

class Pure1DTemporalCNN(tf.keras.Model):
    def __init__(self, num_classes=10, nb_filters=32, kernel_size=3, dropout=0.3):
        super().__init__()

        # Initial conv block
        self.conv_initial = tf.keras.layers.Conv1D(nb_filters, 7, padding='causal', activation='relu')
        self.norm_initial = tf.keras.layers.BatchNormalization()
        self.dropout_initial = tf.keras.layers.SpatialDropout1D(dropout)

        # TCN Blocks with increasing dilation
        self.tcn_blocks = []
        current_filters = nb_filters
        for layer_idx in range(6):
            dilation_rate = 2 ** layer_idx
            self.tcn_blocks.append(
                TCNBlock(nb_filters=current_filters,
                         kernel_size=kernel_size,
                         dilation_rate=dilation_rate,
                         dropout_rate=dropout,
                         name=f"tcn_block_{layer_idx}")
            )
            # Increase filters after layer 3
            if layer_idx == 3:
                current_filters *= 2

        # Global pooling and classifier
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        x = self.conv_initial(inputs)
        x = self.norm_initial(x, training=training)
        x = self.dropout_initial(x, training=training)

        for block in self.tcn_blocks:
            x = block(x, training=training)

        x = self.global_pool(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.classifier(x)
        return x

def create_model(num_classes=10):
    model = Pure1DTemporalCNN(num_classes=num_classes, nb_filters=32)

    # Build model by calling on dummy input
    dummy_input = tf.random.normal((1, 48000, 1))
    _ = model(dummy_input)  # Triggers full build

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------
# 3. Custom Callbacks
# -------------------------------

class F1History(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, verbose=1):
        super().__init__()
        self.validation_data = validation_data
        self.verbose = verbose
        self.f1_scores = []
        self.best_f1 = 0
        self.wait = 0
        self.patience = 10
        self.best_weights = None
        self.restore_best_weights = True

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        val_pred_proba = self.model.predict(X_val, verbose=0)
        val_pred_classes = np.argmax(val_pred_proba, axis=1)
        current_f1 = f1_score(y_val, val_pred_classes, average='macro')
        self.f1_scores.append(current_f1)

        if self.verbose > 0:
            print(f" - val_f1_macro: {current_f1:.4f}")

        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1

        if self.wait >= self.patience:
            if self.verbose > 0:
                print(f"Early stopping at epoch {epoch + 1}. Best F1: {self.best_f1:.4f}")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.model.stop_training and self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print("Restoring model weights from the epoch with best F1 score.")
            self.model.set_weights(self.best_weights)


# -------------------------------
# 4. Training Function
# -------------------------------

def create_tf_dataset(X, y, batch_size=32, shuffle=False, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(1000)
    if augment:
        def tf_augment(x, y):
            noise = tf.random.normal(tf.shape(x), stddev=0.005)
            x = x + noise
            shift = tf.random.uniform((), minval=-2000, maxval=2000, dtype=tf.int32)
            x = tf.roll(x, shift, axis=0)
            return x, y
        dataset = dataset.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_model_improved(data_dir, transform_type, warmup_epochs=15, finetune_epochs=25,
                         warmup_lr=0.001, finetune_lr=0.0001, use_class_weights=True,
                         use_augmentation=True, results_dir="results"):
    # Load data
    X, y, class_labels = load_and_preprocess_data(data_dir)
    num_classes = len(class_labels)

    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    # Compute class weights
    if use_class_weights:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    else:
        class_weight_dict = None

    # Create datasets
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=32, shuffle=True, augment=use_augmentation)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=32)
    test_dataset = create_tf_dataset(X_test, y_test, batch_size=32)

    # Create model
    model = create_model(num_classes=num_classes)

    # Print model summary
    print(model.summary())

    # Callbacks
    f1_callback = F1History(validation_data=(X_val, y_val), verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_macro', factor=0.5, patience=5, verbose=1)
    callbacks = [f1_callback, reduce_lr]

    histories = []

    # Phase 1: Warmup
    print(f"Phase 1: Warmup Training - {warmup_epochs} epochs")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=warmup_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    warmup_history = model.fit(
        train_dataset,
        epochs=warmup_epochs,
        validation_data=val_dataset,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    histories.append(warmup_history)

    # Phase 2: Fine-tuning
    print(f"Phase 2: Fine-tuning - {finetune_epochs} epochs")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    finetune_history = model.fit(
        train_dataset,
        epochs=finetune_epochs,
        validation_data=val_dataset,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    histories.append(finetune_history)

    # Final evaluation
    val_pred_proba = model.predict(X_val, verbose=0)
    val_pred_classes = np.argmax(val_pred_proba, axis=1)
    val_f1 = f1_score(y_val, val_pred_classes, average='macro')
    val_accuracy = accuracy_score(y_val, val_pred_classes)

    test_pred_proba = model.predict(X_test, verbose=0)
    test_pred = np.argmax(test_pred_proba, axis=1)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"Validation - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    print(f"Test - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")

    # Save plots
    save_combined_training_plots(histories, results_dir)
    plot_roc_curves(y_val, val_pred_proba, class_labels, results_dir, "validation")
    plot_roc_curves(y_test, test_pred_proba, class_labels, results_dir, "test")
    save_confusion_matrix(y_val, val_pred_classes, class_labels, os.path.join(results_dir, "confusion_matrix_val.png"),
                          f"Validation Confusion Matrix (Acc: {val_accuracy:.3f}, F1: {val_f1:.3f})")
    save_confusion_matrix(y_test, test_pred, class_labels, os.path.join(results_dir, "confusion_matrix_test.png"),
                          f"Test Confusion Matrix (Acc: {test_accuracy:.3f}, F1: {test_f1:.3f})")

    # Save results
    results_file = os.path.join(results_dir, "comprehensive_results.txt")
    with open(results_file, "w") as f:
        f.write("=== COMPREHENSIVE RESULTS ===\n")
        f.write(f"Model: Pure 1D Temporal CNN\n")
        f.write(f"Transform: {transform_type}\n")
        f.write(f"Input Shape: {INPUT_SHAPE}\n")
        f.write(f"Use Class Weights: {use_class_weights}\n")
        f.write(f"Data Augmentation: {use_augmentation}\n")
        f.write(f"Warmup LR: {warmup_lr}, Fine-tune LR: {finetune_lr}\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}\n")

    # Cleanup
    del X, y, X_train, X_val, X_test, y_train, y_val, y_test
    del train_dataset, val_dataset, test_dataset
    tf.keras.backend.clear_session()

    return {
        'val_accuracy': float(val_accuracy),
        'val_f1': float(val_f1),
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1)
    }


# -------------------------------
# 5. Plotting Functions
# -------------------------------

def save_combined_training_plots(histories, results_dir, phase="combined"):
    os.makedirs(results_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, history in enumerate(histories):
        epochs = range(1, len(history.history['loss']) + 1)
        label = "Warmup" if i == 0 else "Fine-tune"
        ax1.plot(epochs, history.history['loss'], label=f"{label} Loss")
        ax1.plot(epochs, history.history['val_loss'], linestyle="--", label=f"{label} Val Loss")
        ax2.plot(epochs, history.history['accuracy'], label=f"{label} Acc")
        ax2.plot(epochs, history.history['val_accuracy'], linestyle="--", label=f"{label} Val Acc")

    ax1.set_title("Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(results_dir, "training_history.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training plots to {output_file}")


def plot_roc_curves(y_true, y_pred_proba, class_labels, results_dir, phase="validation"):
    n_classes = len(class_labels)
    y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
    fpr, tpr, roc_auc = {}, {}, {}

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_labels.keys()):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.2f})")

    roc_auc["macro"] = np.mean(list(roc_auc.values()))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {phase.capitalize()} (Macro AUC = {roc_auc['macro']:.2f})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_file = os.path.join(results_dir, f"roc_curves_{phase}.png")
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {roc_file}")
    return roc_auc


def save_confusion_matrix(y_true, y_pred, class_labels, filename, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=list(class_labels.keys()), yticklabels=list(class_labels.keys()))
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {filename}")


# -------------------------------
# 6. Main
# -------------------------------

def main():
    start = time.time()
    results = train_model_improved(
        DATASET_DIR,
        args.transform,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        warmup_lr=args.warmup_lr,
        finetune_lr=args.finetune_lr,
        use_class_weights=args.class_weights,
        use_augmentation=not args.no_augmentation,
        results_dir=OUTPUT_DIR
    )
    total_time = time.time() - start
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print("Final Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()