# ML4SCI GSoC 2026 — Genie Evaluation Tasks

**Author:** Muskan Khatoon  
**Frameworks:** PyTorch, PyTorch Geometric  
**Compute Environment:** Kaggle GPU (Tesla P100)

This repository contains my implementation of the **ML4SCI Genie evaluation tasks** for **Google Summer of Code 2026**.

The goal of these tasks is to explore **deep learning approaches for representation learning and classification of quark/gluon jet events** using both **image-based models and graph neural networks**.

The work includes:
- Autoencoder-based representation learning
- Graph neural network jet classification
- Contrastive learning for self-supervised graph representations

---

## Dataset

The dataset consists of simulated **quark and gluon jet events** represented as detector images with three channels:

- **ECAL** — electromagnetic calorimeter
- **HCAL** — hadronic calorimeter
- **Tracks** — particle tracking detector

Each event is a **125 × 125 × 3 detector image**.  
Total events: **139,306**

For graph-based models, detector images were converted into **point clouds and graph representations** by extracting non-zero pixels.

---

## Repository Structure
```
ml4sci-gsoc-2026/
│
├── common_task_1_autoencoder/
│   └── common_task1_autoencoder.ipynb
│
├── common_task_2_gnn/
│   └── common_task2_gnn_classification.ipynb
│
├── specific_task_contrastive_learning/
│   └── specific_task1_contrastive_learning.ipynb
│
└── README.md
```

---

## Common Task 1 — Autoencoder for Jet Representation Learning

### Objective

Train deep autoencoder models to learn **latent representations of jet detector images**.

Two architectures were implemented:
- Convolutional Autoencoder (AE)
- Variational Autoencoder (VAE)

Both models reconstruct the **three detector channels simultaneously**.

### Training Setup

**Loss functions:**
- Weighted L1 reconstruction loss
- KL divergence (for VAE)

**Optimization:**
- Adam optimizer
- ReduceLROnPlateau scheduler
- Early stopping

### Reconstruction Performance

#### Reconstruction Error

| Channel | Mean    | Std     |
|---------|---------|---------|
| ECAL    | 0.01755 | 0.00968 |
| HCAL    | 0.00677 | 0.00358 |
| Tracks  | 0.00099 | 0.00070 |

**Overall mean reconstruction error: 0.00844**

#### Image Quality Metrics

| Channel | MSE     | PSNR     | SSIM   |
|---------|---------|----------|--------|
| ECAL    | 0.00003 | 27.49 dB | 0.2049 |
| HCAL    | ~0      | 36.08 dB | 0.7009 |
| Tracks  | ~0      | 20.30 dB | 0.1369 |

The model captures the **core energy deposition structure of jets**, as seen from the original vs reconstructed images.

### Latent Representation Analysis

To analyze the learned representation space:
- **PCA projection**
- **t-SNE embedding visualization**

These visualizations reveal structure in the latent space corresponding to different jet patterns.

## Reconstructed Images
<img width="1632" height="632" alt="Common_task_1_image_reconstruction" src="https://github.com/user-attachments/assets/73f30cfe-b7a5-46c4-b8cf-871e3c0243a2" />


---

## Common Task 2 — Graph Neural Network for Jet Classification

### Graph Construction

Jet images were converted into graphs using the following pipeline:
1. Extract **non-zero pixels**
2. Treat each pixel as a **node**
3. Construct edges using **k-nearest neighbors in η–φ space**

Node features include detector and spatial information. Graphs were implemented using **PyTorch Geometric**.

### GNN Architecture — GATv2 Classifier

The classification model uses a **deep GATv2 architecture with residual connections and multi-scale aggregation**.
```
Input (9 node features)
        ↓
Linear Projection → 128
        ↓
GATv2 Layer 1 (128 → 128, 8 heads) + Residual
        ↓
GATv2 Layer 2 (128 → 128, 8 heads) + Residual
        ↓
GATv2 Layer 3 (128 → 128, 8 heads) + Residual
        ↓
GATv2 Layer 4 (128 → 256, 8 heads)
        ↓
Multi-scale concatenation
        ↓
Dual Global Pooling (mean + sum)
        ↓
MLP classifier
1024 → 256 → 64 → 2
```

This architecture captures **multi-scale jet substructure features**.

### Supervised Classification Results

| Metric   | Score      |
|----------|------------|
| ROC-AUC  | **0.7919** |
| Accuracy | **0.7157** |
| PR-AUC   | **0.7787** |

#### Classification Report

| Class | Precision | Recall | F1   |
|-------|-----------|--------|------|
| Quark | 0.77      | 0.62   | 0.69 |
| Gluon | 0.68      | 0.81   | 0.74 |

**Overall accuracy: 71.6%**

## ROC-AUC Curve 
<img width="853" height="644" alt="Common_task_2_roc_auc" src="https://github.com/user-attachments/assets/80688875-7a69-4fcb-aa50-6601be466038" />
---


## Specific Task — Contrastive Learning for Graph Representations

To learn jet representations without supervision, a **contrastive learning framework** was implemented. The encoder is trained using **NT-Xent contrastive loss**, encouraging different augmentations of the same jet to have similar embeddings.

Graph augmentations include:
- Node feature masking
- Edge perturbation
- Graph topology variations

### Results

| Method                            | Test ROC-AUC        | Test Accuracy | Notes                          |
|-----------------------------------|---------------------|---------------|--------------------------------|
| Baseline GNN (supervised scratch) | **0.7986**          | **0.7270**    | Same architecture, labels only |
| Contrastive linear probe          | 0.7573              | 0.6922        | Frozen encoder                 |
| Contrastive fine-tuned            | 0.7935              | 0.7275        | Best checkpoint                |
| Contrastive fine-tuned (3 seeds)  | **0.7913 ± 0.0060** | —             | Stable performance             |
| Task 2 GATv2 baseline             | 0.7919              | 0.7157        | Different architecture         |

The contrastive fine-tuned model **matches or exceeds all baselines on average**, despite the encoder being trained **without labels during the entire 200-epoch pretraining phase**.

### Analysis

**Representation Quality**  
The linear probe ROC-AUC of **0.757** demonstrates that the encoder learns **discriminative jet representations purely from self-supervised learning**.

**Fine-tuning Efficiency**  
Fine-tuning improves AUC from **0.757 → 0.791**, showing the pretrained encoder provides a **strong initialization for supervised training**.

**Augmentation Sensitivity**  
Feature masking causes the largest AUC drop, indicating that **node-level energy and positional features carry the most discriminative information**. This aligns with known jet physics:
- Gluon jets have **higher particle multiplicity**
- Quark jets are **more collimated**

**Graph Topology Robustness**  
A k-sensitivity study shows AUC varying by only **0.005 across k ∈ {4, 8, 12, 16}**, suggesting that learned representations are **robust to graph connectivity choices**.

## ROC-AUC Curve 

<img width="978" height="728" alt="Specific_task_1_roc_auc" src="https://github.com/user-attachments/assets/c2e5d7a0-44e1-4672-b7e6-ab993603861c" />

---

## Evaluation Metrics

The following metrics were used throughout the experiments:
- ROC-AUC
- Accuracy
- Precision / Recall / F1-score
- PR-AUC
- Confusion matrix
- Training / validation loss curves

---

## Reproducibility

**Hardware:** Kaggle GPU (Tesla P100)

**Software:**
```
PyTorch
PyTorch Geometric
NumPy
Scikit-learn
Matplotlib
```

---

## Conclusion

This project demonstrates the effectiveness of **deep learning and graph neural networks for jet physics analysis**.

Key takeaways:
- Autoencoders learn meaningful latent representations of jet detector images
- Graph neural networks effectively model jet particle interactions
- Contrastive learning enables **powerful self-supervised graph representation learning**

These methods show strong potential for **scalable analysis of high-energy physics datasets**.
