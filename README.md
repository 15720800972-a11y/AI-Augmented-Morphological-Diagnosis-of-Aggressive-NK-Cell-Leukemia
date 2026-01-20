# AI-Augmented-Morphological-Diagnosis-of-Aggressive-NK-Cell-Leukemia
.AI-Augmented Morphological Diagnosis of Aggressive NK-Cell Leukemia: A Single-Cell Multiple Instance Learning Framework with Cytoplasmic Granule Clustering as a Novel Hallmark
# Self-Supervised Pathology Image Analysis System

This repository contains a deep learning framework for pathology image analysis, integrating **Self-Supervised Learning (MoCo v2)**, **Multiple Instance Learning (MIL)**, and **Prototype-based Classification**.

The system is designed to handle label scarcity in medical imaging by leveraging large unlabeled datasets for pre-training and data expansion.

## 🌟 Key Features

1.  **MoCo Pre-training**: Self-supervised training using Momentum Contrast (MoCo) to learn robust feature representations without labels.
2.  **Dataset Expansion (Retrieval)**: A search engine to retrieve similar images from an unlabeled pool using annotated "seed" images, facilitating efficient dataset labeling.
3.  **MIL Classification**: Attention-based Multiple Instance Learning for patient-level diagnosis (handling bags of instances).
4.  **Prototype-based Inference**: A lightweight classifier using feature centroids (prototypes) for diagnosing unknown samples.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Required packages: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `Pillow`, `tqdm`.*

---
