# Anomaly Detection in Real-Time Surveillance Videos

> PyTorch implementation of real-time anomaly detection in surveillance videos, based on [Sultani et al., 2018 (CVPR)](https://arxiv.org/abs/1801.04264) and extended from [ekosman's PyTorch reimplementation](https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch).

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Feature Extraction](#feature-extraction)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained Models & Downloads](#pretrained-models--downloads)
- [Results](#results)
- [Demo Usage](#demo-usage)
- [Future Work](#future-work)
- [References](#references)
- [FAQ](#faq)

---

## Overview

This project addresses anomaly detection in surveillance videos using deep learning. It implements a fully connected anomaly classifier trained on features extracted by pre-trained 3D CNNs like C3D.

Key features:
- Feature extraction using C3D
- Anomaly detection using a shallow neural network
- ROC-based evaluation
- Offline and real-time video demo support

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
````

2. **Set up the environment (requires Anaconda):**

```bash
conda env create -f environment.yml
conda activate your_env_name
```

---

## Dataset Preparation

1. **Download the UCF-Crime Dataset:**

   Follow instructions from the [UCF-Crime dataset repo](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018).

2. **Organize the dataset:**

```
dataset/
├── Abnormal/
├── Normal/
```

---

## Feature Extraction

Use the pre-trained C3D model to extract features:

```bash
python feature_extractor.py \
  --dataset_path ./dataset \
  --model_type c3d \
  --pretrained_3d ./pretrained/c3d.pickle
```

---

## Training

Train the anomaly detection model on extracted features:

```bash
python TrainingAnomalyDetector_public.py \
  --features_path ./features \
  --annotation_path ./annotations/Train_annotations.txt \
  --epochs 500
```

---

## Evaluation

Generate the ROC curve for your trained model:

```bash
python generate_ROC.py \
  --features_path ./features \
  --annotation_path ./annotations/Test_annotations.txt \
  --model_path ./trained_models/epoch_500.pt
```

---

## Pretrained Models & Downloads

| Resource              | Link                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| 📦 Pretrained Models  | [Google Drive](https://drive.google.com/drive/folders/1YGEBYNNFwxbf3uLOIsjs7NoyFKQhtBHd?usp=sharing) |
| 🎞️ C3D Model Weights | [Google Drive](https://drive.google.com/drive/folders/1ma43hGsazibXhvOQE3Dl6BaHWJZ8OfL1?usp=sharing) |
| 📊 Extracted Features | [Google Drive](https://drive.google.com/drive/folders/1S925QpBLGf2I8ySpuTXrItfQARo-4iID?usp=sharing) |
| 📉 Training Logs      | [Google Drive](https://drive.google.com/drive/folders/1PU0gjVvv-z_CJNk6BeD_NFQ3krBAnbho?usp=sharing) |

---

## Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 85%   |
| Precision | 80%   |
| Recall    | 75%   |

*Note: C3D features may yield slightly lower AUC than reported in the original paper due to different weight versions.*

---

## Demo Usage

### Offline (Video File)

```bash
python video_demo.py \
  --feature_extractor ./pretrained/c3d.pickle \
  --feature_method c3d \
  --ad_model ./trained_models/epoch_500.pt \
  --n_segments 32
```

### Real-Time (Webcam)

```bash
python AD_live_prediction.py \
  --feature_extractor ./pretrained/c3d.pickle \
  --feature_method c3d \
  --ad_model ./trained_models/epoch_500.pt \
  --clip_length 64
```

---


---

## References

* Sultani, W., Chen, C., & Shah, M. (2018). *Real-world Anomaly Detection in Surveillance Videos*. CVPR. [arXiv:1801.04264](https://arxiv.org/abs/1801.04264)
* Kosman, E. (2022). *PyTorch implementation of Real-World Anomaly Detection in Surveillance Videos*. [GitHub](https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch)

BibTeX:

```bibtex
@software{Kosman_Pytorch_implementation_of_2022,
author = {Kosman, Eitan},
title = {{Pytorch implementation of Real-World Anomaly Detection in Surveillance Videos}},
year = {2022},
version = {1.0.0},
url = {https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch}
}
```

---

## ❓ FAQ

**Q:** `video_demo.py` doesn’t show videos?
**A:** Install [LAVFilters](http://forum.doom9.org/showthread.php?t=156191).

**Q:** What’s the second column in `Train_annotations.txt`?
**A:** It’s the video length in frames. Not used for training logic.

---

## 🧑‍💻 Maintainers & Contributions

Contributions welcome! Please open an issue or PR if you find bugs or want to improve this project.

---

```

Let me know if you want this version customized with:
- Your GitHub repo name
- Contributor names
- Additional model links

Or if you’d like a smaller "quickstart" version, I can provide that too.
```
