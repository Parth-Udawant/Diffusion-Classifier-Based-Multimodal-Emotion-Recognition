# Diffusion Classifier-Based Multi-Modal Emotion Recognition

A multimodal emotion recognition framework that combines **audio and visual cues** using transformer-based representation learning and a **diffusion classifier-based multimodal classification network**.

This project was developed as an **M.Tech Computer Science Thesis Project** and was conducted at the **National Institute of Technology Goa (NIT Goa)** as the final-year thesis project.

---

## Overview

Human emotions are expressed through multiple modalities, particularly through speech and facial expressions. This project develops a multimodal emotion recognition pipeline that independently learns emotional representations from audio and video and subsequently combines them in a shared latent space.

The framework uses:

* **WavLM-Large** for audio/speech representation learning
* **VideoMAE-Large** for visual/video representation learning
* **Multimodal concatenation-based fusion**
* **Velocity-parameterization** for diffusion training
* **Multimodal emotion classification through a dedicated diffusion based classification head**

## Project Structure

The repository is intentionally organized according to the four major components of the proposed framework.

```text
Diffusion-Based-Multi-Modal-Emotion-Recognition/
│
├── README.md
│
├── Embedding Extraction/
│   └── Aligned_Embedding_Extraction.py
│
├── WavLM Audio Network/
│   └── WavLM_Script.py
│
├── VideoMAE Video Network/
│   └── VideoMAE_Script.py
│
└── Diffusion Classifier Network/
    └── Diffusoin_Classifier_Network.py
```

---

# Dataset

This project uses the **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)** for multimodal emotion recognition.

The dataset contains synchronized audio and visual recordings representing multiple emotional categories.

### Emotion Classes

The implementation uses six emotion categories:

| Label | Emotion   |
| ----- | --------- |
| `ANG` | Anger     |
| `DIS` | Disgust   |
| `FEA` | Fear      |
| `HAP` | Happiness |
| `SAD` | Sadness   |
| `NEU` | Neutral   |

### Download CREMA-D

The CREMA-D dataset is **not included in this repository**.

Download the dataset from the official CREMA-D repository:

[CREMA-D Dataset Repository](https://github.com/CheyneyComputerScience/CREMA-D)

After downloading and extracting the dataset, the audio and video files should be made available to the scripts.

The original implementation expects separate audio and video directories corresponding to the CREMA-D dataset structure.

---

# Methodology

The proposed framework consists of four stages.

### Stage 1: Modality-Specific Representation Learning

Audio and visual modalities are processed independently using pretrained transformer architectures.

* Audio → **WavLM-Large**
* Video → **VideoMAE-Large**

### Stage 2: Aligned Embedding Extraction

The trained WavLM and VideoMAE networks are used to extract embeddings for corresponding audio-video samples.

The extraction process maintains correspondence between:

* Audio embedding
* Video embedding
* Emotion label

An actor-independent dataset split is used so that actors assigned to the training set do not appear in validation or test partitions.

### Stage 3: Multimodal Fusion

The 512-dimensional audio and visual embeddings are concatenated to form a 1024-dimensional multimodal representation.

A fully connected fusion network then transforms this representation into a 512-dimensional latent representation.

### Stage 4: Diffusion-Based Classification

The fused 512-dimensional representation is used as the clean latent representation for a diffusion process.

---

# Installation

Create a Python environment and install the required dependencies.

The implementation uses libraries including:

```bash
pip install torch torchaudio transformers numpy pandas scikit-learn matplotlib seaborn tqdm decord
```

---

# Citation

If you use this implementation or build upon this project, please cite the associated thesis/research work when the corresponding publication is available.

---

**Created By Parth Udawant**
