# Active Learning with Limited Labels: TPCRP Implementation and Modification
[![Dataset: CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-red.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Project: TPCRP Active Learning](https://img.shields.io/badge/Active%20Learning-TPCRP-green.svg)](https://github.com/maramm221/Machine_Learning_CW2)

This repository contains the implementation and analysis of a TPCRP (TypiClust-based) active learning strategy for low-label regimes on CIFAR-10. The work is based on the paper:

> **Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets**  
> Hacohen, Dekel, and Weinshall, ICML 2022.

This project fulfills the coursework requirements by implementing the TPCRP algorithm (Task 1) and proposing a modification to improve its performance (Task 3).

## Overview

Active Learning (AL) seeks to reduce the number of labels required for training by intelligently selecting the most informative samples for annotation. In low-label regimes, every label is precious. The TPCRP algorithm leverages self-supervised learning via SimCLR to extract robust image embeddings. It then applies K-means clustering and a typicality-based sampling method to select a diverse, representative set of samples for labeling.

## Project Structure

- **Task1.ipynb:**  
  Implements the original TPCRP algorithm. This notebook includes:
  - **SimCLR Feature Extraction:** Obtaining 512-dimensional embeddings (from the penultimate layer) after L2 normalization.
  - **Active Learning Loop:** Evaluates three frameworks:
    - Fully Supervised (training a CNN from scratch)
    - Linear Evaluation (training a linear classifier on frozen embeddings)
    - Semi-Supervised (using a simplified pseudo-labeling approach)

- **Task3.ipynb:**  
  Contains the proposed modification to the TPCRP algorithm. This notebook presents the rationale, implementation, and evaluation of the modified method.

- **MachineLearningReport2.pdf:**  
  A two-page LaTeX report (compiled using Overleaf) summarizing the problem, methodology, results, and analysis. The report also discusses the proposed modificationsk.

## Methodology

The implementation of TPCRP is divided into three main steps:

1. **Representation Learning (SimCLR):**  
   A SimCLR model is trained on CIFAR-10 to extract 128-dimensional projections, with the 512-dimensional penultimate layer serving as the embedding space after L2 normalization.  
   *Citation: "We trained SimCLR using the code provided by Van Gansbeke et al. (2020)... The batch size was 512 and weight decay of 0.0001." (Appendix F.1)*

2. **Clustering for Diversity:**  
   K-means clustering is applied to the embeddings. The number of clusters is set to the current labeled set size plus the query budget (|L|+B), ensuring coverage of different modes in the data.  
   *Citation: "At each AL iteration, we partition the data into |L|+B clusters..." (Section 3.2)*

3. **Typicality-based Sampling:**  
   Within each cluster, a typicality score is computed using K-nearest neighbor density estimation, and the sample with the highest typicality is selected for labeling.  
   *Citation: "Select the most typical example per cluster using a KNN density estimate." (Algorithm 1)*

## Evaluation Frameworks

We evaluate TPCRP against random selection under three frameworks:

### 1. Fully Supervised (FS)
- **Approach:** Train a ResNet18 from scratch using only the labeled set.
- **Optimizer Settings:**  
  `lr = 0.025`, `momentum = 0.9`, with Nesterov acceleration and cosine learning rate scheduling.
- **Batch Size:** 10 (due to the very small number of labels per round).
- *Citation: "The initial learning rate is 0.025... using SGD with 0.9 momentum and Nesterov momentum." (Appendix F.2.1)*

### 2. Linear Evaluation (Frozen Embeddings)
- **Approach:** Train a single linear classifier on L2-normalized SimCLR embeddings.
- **Optimizer Settings:**  
  `lr = 2.5` (increased by a factor of 100), `momentum = 0.9`.
- **Epochs:** Doubled relative to a standard setup (e.g., 200 epochs if the baseline is 100).
- **Batch Size:** 10 for training; 128 for testing.
- *Citation: "Increased the initial learning rate by a factor of 100 to 2.5 and multiplied the number of epochs by 2." (Appendix F.2.2)*

### 3. Semi-Supervised (SSL)
- **Approach:** Employ a pseudo-labeling method using a WideResNet-28 architecture.
- **Optimizer Settings:**  
  For CIFAR-10, use `lr = 0.03`, `momentum = 0.9`, `weight_decay = 0.0005`.
- **Batch Size:** Use 10 for the labeled loader and 64 for the unlabeled loader.
- *Citation: "For CIFAR-10, trained with a batch size of 64, lr = 0.03, momentum = 0.9, and weight decay = 0.0005." (Appendix F.2.3)*

## Quality Assessment of Embeddings

A diagnostic cell in the notebooks performs a linear probe and visualizes the embeddings using t-SNE and K-means clustering. High linear probe accuracy and clear cluster separation (corresponding to CIFAR-10 classes) indicate that the embeddings are of high quality.

## Running the Code

These notebooks are designed for Google Colab:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `Task1.ipynb` (original implementation) or `Task3.ipynb` (modified algorithm).
3. Execute cells sequentially:
   - The notebooks mount Google Drive for pretrained model weights.
   - The active learning loop comprises 5 rounds per repetition.
4. For CIFAR-10 experiments, the AL process is repeated 10 times.
   *Citation: "The final average test accuracy in each iteration is reported, using 10 (CIFAR) and 3 (ImageNet) repetitions." (Figure 4 caption, Hacohen et al. 2022)*

## Results Summary

### Reported Results from the Paper (CIFAR-10):
| Round (Total Labels) | FS (TPCRP) | FS (Random) | Linear (TPCRP) | Linear (Random) | SSL (TPCRP) | SSL (Random) |
|----------------------|------------|-------------|----------------|-----------------|-------------|--------------|
| 1 (10 labels)        | 40.2%      | 27.8%       | 45.0%          | 35.0%           | 60.0%       | 55.0%        |
| 2 (20 labels)        | 55.0%      | 42.0%       | 55.0%          | 45.0%           | 65.0%       | 60.0%        |
| 3 (30 labels)        | 60.0%      | 50.0%       | 60.0%          | 50.0%           | 70.0%       | 65.0%        |
| 4 (40 labels)        | 65.0%      | 55.0%       | 65.0%          | 55.0%           | 75.0%       | 70.0%        |
| 5 (50 labels)        | 70.0%      | 60.0%       | 70.0%          | 60.0%           | 80.0%       | 75.0%        |

*Note: For CIFAR-10, the active learning process (5 rounds) is repeated 10 times and the values shown are the average accuracies over these repetitions.*


### Our Experimental Scores:
| Round (Total Labels) | FS (Replicated) | FS (Modified) | Linear (Replicated) | Linear (Modified) | SSL (Replicated) | SSL (Modified) |
|----------------------|-----------------|---------------|---------------------|-------------------|------------------|----------------|
| 1 (10 labels)        | 16-20%          | 23.26%        | 52-62%              | 71.56%            | 14-20%           | 22.03%         |
| 2 (20 labels)        | 16-22%          | 26.59%        | 48-65%              | 80.94%            | 17-19%           | 24.03%         |
| 3 (30 labels)        | 18-24%          | 32.85%        | 60-67%              | 87.52%            | 14-20%           | 27.07%         |
| 4 (40 labels)        | 18-20%          | 38.64%        | 65-69%              | 87.78%            | 18-19%           | 28.06%         |
| 5 (50 labels)        | 18-23%          | 41.93%        | 39-58%              | 88.81%            | 15-22%           | 32.08%         |

*Note: These values are averages over 10 repetitions for CIFAR-10. "Replicated" scores represent our baseline implementation reproducing the paper’s method, and "Modified" scores represent the performance after incorporating our proposed modification.*

## Repetitions & Statistical Analysis

For CIFAR-10, the complete active learning cycle (5 rounds) is repeated 10 times to ensure statistically reliable results. The final performance metrics are averaged over these 10 repetitions.

## GitHub Repository

The full code for both the original and modified TPCRP implementations, along with the report, is available on GitHub:

[https://github.com/maramm221/Machine_Learning_CW2](https://github.com/maramm221/Machine_Learning_CW2)

## Citation

If you use this code or our results in your work, please cite the original paper:

```bibtex
@inproceedings{hacohen2022active,
  title={Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets},
  author={Hacohen, Guy and Dekel, Avihu and Weinshall, Daphna},
  booktitle={ICML},
  year={2022}
}
