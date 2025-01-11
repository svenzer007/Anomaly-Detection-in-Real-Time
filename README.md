# Anomaly Detection in Real-Time Surveillance Videos
---

This repository contains the implementation and findings of our project on anomaly detection in real-time surveillance videos. The work replicates and extends the methodologies presented in [Kosman, E. (2022). Pytorch implementation of Real-World Anomaly Detection in Surveillance Videos (Version 1.0.0) [Computer software]. https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch](https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch) and based on [Sultani et al., 2018](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained Models](#pre-trained-models)
- [Extracted Features](#extracted-features)
- [C3D Model](#c3d-model)
- [Training Loss](#training-loss)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction

Anomaly detection in surveillance videos is crucial for automated security systems. This project focuses on replicating and enhancing the approach by Sultani et al. using a PyTorch implementation.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up the environment:**
   Ensure you have [Anaconda](https://www.anaconda.com/products/individual) installed. Then, create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate your_env_name
   ```

---

## Data Preparation

1. **Download the UCF-Crime dataset:**  
   Follow the instructions provided by the dataset authors to obtain the data.

2. **Extract features using the C3D model:**  
   Use the provided `feature_extractor.py` script to extract features from the videos.

---

## Training

To train the anomaly detection model:

```bash
python TrainingAnomalyDetector_public.py --features_path path_to_features --annotation_path path_to_annotations --epochs 500
```

---

## Evaluation

To evaluate the trained model:

```bash
python generate_ROC.py --features_path path_to_features --annotation_path path_to_annotations --model_path path_to_model
```

---

## Pre-trained Models

Pre-trained models trained for 500 epochs are available for download:

- [Trained Models](https://drive.google.com/drive/folders/1YGEBYNNFwxbf3uLOIsjs7NoyFKQhtBHd?usp=sharing)

---

## Extracted Features

Extracted features used in the project can be found here:

- [Extracted Features](https://drive.google.com/drive/folders/1S925QpBLGf2I8ySpuTXrItfQARo-4iID?usp=sharing)

---

## C3D Model

Download the pre-trained C3D model used for feature extraction:

- [C3D Model](https://drive.google.com/drive/folders/1ma43hGsazibXhvOQE3Dl6BaHWJZ8OfL1?usp=sharing)

---

## Training Loss

The loss values recorded during training are available here:

- [Training Loss](https://drive.google.com/drive/folders/1PU0gjVvv-z_CJNk6BeD_NFQ3krBAnbho?usp=drive_link)

---

## Results

Our model achieved the following performance metrics:

- **Accuracy:** 85%
- **Precision:** 80%
- **Recall:** 75%

---

## Future Work

Future improvements include:

- Implementing dictionary learning techniques.
- Enhancing model robustness to diverse surveillance scenarios.
- Exploring transformer-based feature extractors.

---

## References

- Sultani, W., Chen, C., & Shah, M. (2018). Real-world Anomaly Detection in Surveillance Videos. *CVPR*. [Paper](https://arxiv.org/abs/1801.04264)
