import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

from transformers import (
    WavLMModel,
    AutoFeatureExtractor,
    VideoMAEModel,
    VideoMAEImageProcessor
)

from decord import VideoReader, cpu

AUDIO_PATH = "/media/csedept/cse2016/AV_RP/crema-d-mirror/AudioWAV"
VIDEO_PATH = "/media/csedept/cse2016/AV_RP/crema-d-mirror/VideoFlash"

AUDIO_MODEL = "/media/csedept/cse2016/AV_RP/80_wavlm_large_cremad_attentive_perfected/best_model.pt"
VIDEO_MODEL = "/media/csedept/cse2016/AV_RP/81_videomae_large_cremad_final_stable/best_model.pt"

OUTPUT_DIR = "/media/csedept/cse2016/AV_RP/aligned_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 5.0
NUM_FRAMES = 16
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_MAP = {
    "ANG":0,
    "DIS":1,
    "FEA":2,
    "HAP":3,
    "SAD":4,
    "NEU":5
}

class AttentiveStatsPooling(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,x):

        w = torch.softmax(self.attention(x),dim=1)

        mean = torch.sum(w*x,dim=1)
        std = torch.sqrt(torch.sum(w*(x-mean.unsqueeze(1))**2,dim=1)+1e-9)

        return torch.cat([mean,std],dim=1)


class WavLMClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        hidden = self.wavlm.config.hidden_size

        self.pool = AttentiveStatsPooling(hidden)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,6)
        )

    def forward(self,x,return_embedding=False):

        out = self.wavlm(input_values=x)
        hidden = out.last_hidden_state

        pooled = self.pool(hidden)

        embedding = self.classifier[:4](pooled)

        if return_embedding:
            return embedding

        logits = self.classifier[4](embedding)

        return logits

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

        self.classifier = nn.Linear(512, 6)

    def forward(self, x, return_embedding=False):

        outputs = self.backbone(pixel_values=x)

        tokens = outputs.last_hidden_state

        mean_pool = torch.mean(tokens, dim=1)
        att_pool = self.att_pool(tokens)

        pooled = torch.cat([mean_pool, att_pool], dim=1)

        embedding = self.embedding_head(pooled)

        if return_embedding:
            return embedding

        logits = self.classifier(embedding)

        return logits

def get_split():

    files = sorted([f for f in os.listdir(AUDIO_PATH) if f.endswith(".wav")])

    actors = sorted(list(set([f.split("_")[0] for f in files])))

    train_cut = int(0.7*len(actors))
    val_cut = int(0.85*len(actors))

    train_actors = actors[:train_cut]
    val_actors = actors[train_cut:val_cut]
    test_actors = actors[val_cut:]

    train=[]
    val=[]
    test=[]

    for f in files:

        actor = f.split("_")[0]

        if actor in train_actors:
            train.append(f)

        elif actor in val_actors:
            val.append(f)

        else:
            test.append(f)

    return train,val,test

audio_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large")

def process_audio(path):

    waveform,sr = torchaudio.load(path)

    if sr!=SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform,sr,SAMPLE_RATE)

    waveform = waveform.mean(0)

    max_len = int(MAX_AUDIO_LENGTH*SAMPLE_RATE)

    if waveform.shape[0] > max_len:
        waveform = waveform[:max_len]
    else:
        waveform = torch.nn.functional.pad(waveform,(0,max_len-waveform.shape[0]))

    waveform = waveform.numpy()

    inputs = audio_processor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    return inputs.input_values.squeeze(0)

def process_video(path):

    vr = VideoReader(path,ctx=cpu(0))

    total = len(vr)

    idx = np.linspace(0,total-1,NUM_FRAMES).astype(int)

    frames = vr.get_batch(idx).asnumpy()

    inputs = video_processor(list(frames),return_tensors="pt")

    return inputs["pixel_values"].squeeze(0)

def extract(split_name,files,audio_model,video_model):

    audio_emb=[]
    video_emb=[]
    labels=[]

    for f in tqdm(files):

        audio_file = os.path.join(AUDIO_PATH,f)

        video_file = os.path.join(
            VIDEO_PATH,
            f.replace(".wav",".flv")
        )

        if not os.path.exists(video_file):
            continue

        label = EMOTION_MAP[f.split("_")[2]]

        audio_input = process_audio(audio_file).unsqueeze(0).to(DEVICE)
        video_input = process_video(video_file).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            a = audio_model(audio_input,return_embedding=True)
            v = video_model(video_input,return_embedding=True)

        audio_emb.append(a.cpu().numpy())
        video_emb.append(v.cpu().numpy())
        labels.append(label)

    audio_emb = np.vstack(audio_emb)
    video_emb = np.vstack(video_emb)
    labels = np.array(labels)

    np.save(f"{OUTPUT_DIR}/{split_name}_audio_embeddings_512.npy",audio_emb)
    np.save(f"{OUTPUT_DIR}/{split_name}_video_embeddings_512.npy",video_emb)
    np.save(f"{OUTPUT_DIR}/{split_name}_labels.npy",labels)

    print(split_name,"done",audio_emb.shape)

def main():

    print("Loading models")

    audio_model = WavLMClassifier().to(DEVICE)
    video_model = VideoMAEStable().to(DEVICE)

    audio_model.load_state_dict(torch.load(AUDIO_MODEL,map_location=DEVICE))
    video_model.load_state_dict(torch.load(VIDEO_MODEL,map_location=DEVICE))

    audio_model.eval()
    video_model.eval()

    train,val,test = get_split()

    extract("train",train,audio_model,video_model)
    extract("val",val,audio_model,video_model)
    extract("test",test,audio_model,video_model)

    print("All embeddings extracted")


if __name__ == "__main__":
    main()