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
from torch.utils.data import Dataset, DataLoader

from transformers import (
    VideoMAEModel,
    VideoMAEImageProcessor,
    get_cosine_schedule_with_warmup
)

from decord import VideoReader, cpu

DATA_PATH = "/media/csedept/cse2016/AV_RP/crema-d-mirror/VideoFlash"
OUTPUT_DIR = "./81_videomae_large_cremad_final_stable"

BATCH_SIZE = 4
NUM_FRAMES = 16

EPOCHS_FROZEN = 10
EPOCHS_FINETUNE = 35

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

class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn = torch.softmax(self.attention(x), dim=1)
        return torch.sum(attn * x, dim=1)

class CremaDVideoDataset(Dataset):
    def __init__(self, files, processor, train=True):
        self.files = files
        self.processor = processor
        self.train = train

    def sample_frames(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, NUM_FRAMES).astype(int)
        frames = vr.get_batch(indices).asnumpy()

        if self.train and random.random() > 0.5:
            frames = frames[:, :, ::-1, :]

        return frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        frames = self.sample_frames(path)

        inputs = self.processor(
            list(frames),
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].squeeze(0)
        emotion = os.path.basename(path).split("_")[2]
        label = EMOTION_MAP[emotion]

        return pixel_values, label

def get_split():
    files = []
    for root, _, filenames in os.walk(DATA_PATH):
        for f in filenames:
            if f.endswith((".mp4", ".avi", ".flv")):
                files.append(os.path.join(root, f))

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

class VideoMAEStable(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")
        hidden = self.backbone.config.hidden_size

        self.att_pool = AttentivePooling(hidden)

        self.embedding_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state

        mean_pool = torch.mean(tokens, dim=1)
        att_pool = self.att_pool(tokens)
        pooled = torch.cat([mean_pool, att_pool], dim=1)

        embedding = self.embedding_head(pooled)
        logits = self.classifier(embedding)

        return logits, embedding

def compute_class_weights(files):
    counts = np.zeros(NUM_CLASSES)
    for f in files:
        emotion = os.path.basename(f).split("_")[2]
        counts[EMOTION_MAP[emotion]] += 1

    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

def train():

    train_files, val_files, test_files = get_split()
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large")

    train_loader = DataLoader(CremaDVideoDataset(train_files, processor, True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_loader = DataLoader(CremaDVideoDataset(val_files, processor, False),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_loader = DataLoader(CremaDVideoDataset(test_files, processor, False),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = VideoMAEStable().to(DEVICE)

    for p in model.backbone.parameters():
        p.requires_grad = False

    class_weights = compute_class_weights(train_files)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        list(model.embedding_head.parameters()) +
        list(model.classifier.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

    scaler = torch.cuda.amp.GradScaler()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0

    for epoch in range(EPOCHS_FROZEN + EPOCHS_FINETUNE):

        if epoch == EPOCHS_FROZEN:
            print("🔓 Unfreezing backbone")
            for p in model.backbone.parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": 1e-5},
                {"params": model.embedding_head.parameters(), "lr": 5e-5},
                {"params": model.classifier.parameters(), "lr": 5e-5},
            ], weight_decay=1e-4)

            total_steps_ft = len(train_loader) * EPOCHS_FINETUNE
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps_ft),
                num_training_steps=total_steps_ft
            )
        elif epoch == 0:
            total_steps = len(train_loader) * EPOCHS_FROZEN
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )

        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits, _ = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()
                preds = torch.argmax(logits, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

    print("\nBest Validation Accuracy:", best_val)

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    model.eval()

    def extract(loader, split):
        preds_all, labels_all, embeds_all = [], [], []

        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(DEVICE)
                logits, emb = model(x)
                preds = torch.argmax(logits, 1)

                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(y.numpy())
                embeds_all.append(emb.cpu().numpy())

        embeds_all = np.concatenate(embeds_all)

        np.save(os.path.join(OUTPUT_DIR, f"{split}_embeddings_512.npy"), embeds_all)
        np.save(os.path.join(OUTPUT_DIR, f"{split}_labels.npy"), np.array(labels_all))

        return labels_all, preds_all

    train_labels, train_preds = extract(train_loader, "train")
    val_labels, val_preds = extract(val_loader, "val")
    test_labels, test_preds = extract(test_loader, "test")

    acc = accuracy_score(test_labels, test_preds)
    f1m = f1_score(test_labels, test_preds, average="macro")
    f1w = f1_score(test_labels, test_preds, average="weighted")
    uar = recall_score(test_labels, test_preds, average="macro")
    prec = precision_score(test_labels, test_preds, average="macro")

    metrics = {
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "uar": uar,
        "precision_macro": prec
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(
        test_labels,
        test_preds,
        target_names=[IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)],
        output_dict=True
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "emotion_wise_metrics.csv")
    )

    cm = confusion_matrix(test_labels, test_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=IDX_TO_EMOTION.values(),
                yticklabels=IDX_TO_EMOTION.values())
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=IDX_TO_EMOTION.values(),
                yticklabels=IDX_TO_EMOTION.values())
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"))
    plt.close()

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
    plt.close()

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "training_log.csv"),
        index=False
    )

    print("\nAll metrics, curves, embeddings, and logs saved successfully.")

if __name__ == "__main__":
    train()