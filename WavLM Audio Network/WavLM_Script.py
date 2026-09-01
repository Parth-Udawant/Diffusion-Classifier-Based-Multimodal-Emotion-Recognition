import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

from transformers import (
    WavLMModel,
    AutoFeatureExtractor,
    get_cosine_schedule_with_warmup
)

DATA_PATH = "/media/csedept/cse2016/AV_RP/crema-d-mirror/AudioWAV"
OUTPUT_DIR = "./wavlm_large_cremad_attentive_perfected"

BATCH_SIZE = 16
EPOCHS_INITIAL = 20
EPOCHS_FINE = 30
PLATEAU_PATIENCE = 3
MIN_EPOCHS_BEFORE_PLATEAU = 5

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 5.0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

EMOTION_MAP = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "SAD": 4,
    "NEU": 5
}

IDX_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}
NUM_CLASSES = len(EMOTION_MAP)

class CremaDDataset(Dataset):
    def __init__(self, files, feature_extractor):
        self.files = files
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sr = torchaudio.load(path)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        waveform = waveform.mean(dim=0)

        max_len = int(MAX_AUDIO_LENGTH * SAMPLE_RATE)
        if waveform.shape[0] > max_len:
            waveform = waveform[:max_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[0]))

        waveform = waveform.numpy()

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = inputs.input_values.squeeze(0)
        emotion = os.path.basename(path).split("_")[2]
        label = EMOTION_MAP[emotion]

        return input_values, label

def get_split():
    files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".wav")]
    actors = sorted(list(set([os.path.basename(f).split("_")[0] for f in files])))

    train_cut = int(0.7 * len(actors))
    val_cut = int(0.85 * len(actors))

    train_actors = actors[:train_cut]
    val_actors = actors[train_cut:val_cut]
    test_actors = actors[val_cut:]

    train_files = [f for f in files if os.path.basename(f).split("_")[0] in train_actors]
    val_files = [f for f in files if os.path.basename(f).split("_")[0] in val_actors]
    test_files = [f for f in files if os.path.basename(f).split("_")[0] in test_actors]

    return train_files, val_files, test_files

class AttentiveStatsPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        mean = torch.sum(attn_weights * x, dim=1)
        std = torch.sqrt(torch.sum(attn_weights * (x - mean.unsqueeze(1))**2, dim=1) + 1e-9)
        return torch.cat([mean, std], dim=1)

class WavLMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.pool = AttentiveStatsPooling(self.wavlm.config.hidden_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.wavlm.config.hidden_size * 2),
            nn.Linear(self.wavlm.config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        outputs = self.wavlm(input_values=x)
        hidden = outputs.last_hidden_state
        pooled = self.pool(hidden)
        return self.classifier(pooled)

def train():

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    train_files, val_files, test_files = get_split()

    train_loader = DataLoader(CremaDDataset(train_files, feature_extractor),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_loader = DataLoader(CremaDDataset(val_files, feature_extractor),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_loader = DataLoader(CremaDDataset(test_files, feature_extractor),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = WavLMClassifier().to(DEVICE)

    for param in model.wavlm.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    total_steps = len(train_loader) * (EPOCHS_INITIAL + EPOCHS_FINE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_acc = 0
    plateau_counter = 0
    fine_tune_triggered = False

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS_INITIAL + EPOCHS_FINE):

        model.train()
        train_loss = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            plateau_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        else:
            plateau_counter += 1

        if (epoch >= MIN_EPOCHS_BEFORE_PLATEAU and
            plateau_counter >= PLATEAU_PATIENCE and
            not fine_tune_triggered):

            print("\n🔓 Unfreezing WavLM for fine-tuning")

            for param in model.wavlm.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
            fine_tune_triggered = True

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    uar = recall_score(all_labels, all_preds, average="macro")
    precision_macro = precision_score(all_labels, all_preds, average="macro")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)],
        output_dict=True
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "emotion_wise_metrics.csv")
    )

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "uar": uar,
        "precision_macro": precision_macro
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=IDX_TO_EMOTION.values(),
                yticklabels=IDX_TO_EMOTION.values())
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=IDX_TO_EMOTION.values(),
                yticklabels=IDX_TO_EMOTION.values())
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"))

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))

    plt.figure()
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "training_log.csv"),
        index=False
    )

    print("\nFINAL TEST RESULTS")
    print(metrics)
    print("Best Val Accuracy:", best_val_acc)


if __name__ == "__main__":
    train()