import os
import json
import csv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score
)

import matplotlib.pyplot as plt

EMBED_DIR = "/media/csedept/cse2016/AV_RP/aligned_embeddings"
SAVE_DIR = "./experiment_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 100
LR = 3e-5

LATENT_DIM = 512
DIFFUSION_STEPS = 50

class EmbeddingDataset(Dataset):
    def __init__(self, audio, video, labels):
        self.audio = torch.tensor(audio).float()
        self.video = torch.tensor(video).float()
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio[idx], self.video[idx], self.labels[idx]

class DiffusionSchedule:
    def __init__(self, steps):
        beta = torch.linspace(1e-4, 0.02, steps)
        self.alpha = (1 - beta).to(DEVICE)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(DEVICE)

schedule = DiffusionSchedule(DIFFUSION_STEPS)

class FusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio_classifier = nn.Linear(512, 6)
        self.video_classifier = nn.Linear(512, 6)

        self.fusion = nn.Sequential(
            nn.Linear(1024,1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self,audio,video):
        audio_logits = self.audio_classifier(audio)
        video_logits = self.video_classifier(video)

        fused = torch.cat([audio,video],dim=1)
        fused = self.fusion(fused)

        return fused, audio_logits, video_logits

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )

        self.backbone = nn.Sequential(
            nn.Linear(LATENT_DIM + 64,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU()
        )

        self.v_head = nn.Linear(512,LATENT_DIM)
        self.cls_head = nn.Linear(512,6)

    def forward(self,x,t):
        t = t.unsqueeze(1).float()
        t_emb = self.time_embed(t)

        h = torch.cat([x,t_emb],dim=1)
        h = self.backbone(h)

        v = self.v_head(h)
        logits = self.cls_head(h)

        return v, logits

def load_split(split):
    audio = np.load(f"{EMBED_DIR}/{split}_audio_embeddings_512.npy")
    video = np.load(f"{EMBED_DIR}/{split}_video_embeddings_512.npy")
    labels = np.load(f"{EMBED_DIR}/{split}_labels.npy")
    return audio, video, labels

def evaluate(fusion, denoiser, loader):
    fusion.eval()
    denoiser.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for audio,video,label in loader:
            audio = audio.to(DEVICE)
            video = video.to(DEVICE)

            x0,_,_ = fusion(audio,video)
            logits = denoiser(x0, torch.zeros(x0.shape[0], device=DEVICE))[1]

            p = torch.argmax(logits,1)

            preds.extend(p.cpu().numpy())
            labels.extend(label.numpy())

    acc = accuracy_score(labels,preds)
    return acc, preds, labels

def save_results(log_rows, preds, labels):

    with open(os.path.join(SAVE_DIR,"training_log.csv"),"w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","loss","val_accuracy"])
        writer.writerows(log_rows)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    uar = recall_score(labels, preds, average="macro")
    precision_macro = precision_score(labels, preds, average="macro")

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "uar": uar,
        "precision_macro": precision_macro
    }

    with open(os.path.join(SAVE_DIR,"metrics.json"),"w") as f:
        json.dump(metrics,f,indent=4)

    print("\nFinal Test Metrics:")
    print(metrics)
    
    cm = confusion_matrix(labels,preds)

    plt.figure(figsize=(6,5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(SAVE_DIR,"confusion_matrix.png"))
    plt.close()

def train():

    train_audio,train_video,train_labels = load_split("train")
    val_audio,val_video,val_labels = load_split("val")
    test_audio,test_video,test_labels = load_split("test")

    train_loader = DataLoader(EmbeddingDataset(train_audio,train_video,train_labels),
                              batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(EmbeddingDataset(val_audio,val_video,val_labels),
                            batch_size=BATCH_SIZE)

    test_loader = DataLoader(EmbeddingDataset(test_audio,test_video,test_labels),
                             batch_size=BATCH_SIZE)

    fusion = FusionNetwork().to(DEVICE)
    denoiser = Denoiser().to(DEVICE)

    optimizer = optim.AdamW(
        list(fusion.parameters()) + list(denoiser.parameters()),
        lr=LR
    )

    mse = nn.MSELoss()

    best_val = 0
    log_rows = []

    for epoch in range(EPOCHS):

        fusion.train()
        denoiser.train()

        losses = []

        for audio,video,label in tqdm(train_loader):

            audio = audio.to(DEVICE)
            video = video.to(DEVICE)
            label = label.to(DEVICE)

            x0, audio_logits, video_logits = fusion(audio,video)

            t = torch.randint(0,DIFFUSION_STEPS,(x0.shape[0],),device=DEVICE)
            noise = torch.randn_like(x0)

            alpha_bar = schedule.alpha_bar[t].unsqueeze(1)

            xt = torch.sqrt(alpha_bar)*x0 + torch.sqrt(1-alpha_bar)*noise
            v_target = torch.sqrt(alpha_bar)*noise - torch.sqrt(1-alpha_bar)*x0

            v_pred, logits = denoiser(xt,t)

            diff_loss = mse(v_pred,v_target)
            cls_loss = nn.functional.cross_entropy(logits,label)

            audio_loss = nn.functional.cross_entropy(audio_logits,label)
            video_loss = nn.functional.cross_entropy(video_logits,label)

            loss = diff_loss + 2*cls_loss + 0.5*(audio_loss+video_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        val_acc, _, _ = evaluate(fusion,denoiser,val_loader)

        log_rows.append([epoch,np.mean(losses),val_acc])

        print(f"\nEpoch {epoch} | Loss {np.mean(losses):.4f} | ValAcc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc

            torch.save({
                "fusion":fusion.state_dict(),
                "denoiser":denoiser.state_dict()
            },os.path.join(SAVE_DIR,"best_model.pt"))

    checkpoint = torch.load(os.path.join(SAVE_DIR,"best_model.pt"))

    fusion.load_state_dict(checkpoint["fusion"])
    denoiser.load_state_dict(checkpoint["denoiser"])

    test_acc, test_preds, test_labels = evaluate(fusion,denoiser,test_loader)

    save_results(log_rows, test_preds, test_labels)

if __name__=="__main__":
    train()