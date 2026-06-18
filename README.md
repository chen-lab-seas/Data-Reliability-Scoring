# Data Reliability Scoring

Code for the paper:

**Data Reliability Scoring**
Yiling Chen, Shi Feng, Paul Kattuman, Fang-Yi Yu
arXiv:2510.17085

## Overview

This paper introduces the **Gram Determinant Score**, a method for assessing the reliability of reported data without access to ground truth. Given reported labels x-hat and auxiliary observations y drawn from an unknown experiment P(y|x), the score measures how well the reported labels separate the observation distributions across classes.

**Score definition (Definition 4.5):** For reported labels x-hat in {1,...,d} and observations y, define the d x d matrix

    G-hat(x, x') = (1/N^2) * sum_{n: x-hat_n = x} sum_{n': x-hat_{n'} = x'} 1[y_n = y_{n'}]

The plug-in Gram determinant score is Gamma = det(G-hat). Geometrically, Gamma measures the volume of the parallelepiped spanned by the class-conditional observation distributions: more reliable (less manipulated) data yields more distinct per-class distributions and therefore a larger score.

**Kernel variant (Definition 4.9/4.11):** For continuous y, replace the indicator with a kernel K(y,y'), e.g., Gaussian K(y,y') = exp(-||y-y'||^2 / sigma^2) or inner product K(y,y') = <y,y'>.

**Key properties:**
- Theorem 4.3: Score preserves reliability orderings under exact match, Blackwell dominance, and approximate Hamming corruption.
- Proposition 4.4: Score rankings do not depend on the unknown experiment matrix P.

## Repository Structure

```
Data-Reliability-Scoring/
  notebooks/
    synthetic_data_exp.ipynb        -- Experiment 1 (Section 5.1), Figures 2-3, and score comparison (Appendix G)
    additional_synthetic_data_exp.ipynb  -- Supplemental Exp 1 and Exp 2 variants with Gaussian/delta kernels and sample-size studies
    employment_data_exp.ipynb       -- Experiment 3 (Section 5.3), Table 1: CES employment data
    image_data_exp.ipynb            -- Experiment 2 (Section 5.2), Figure 4: CIFAR-10 SimCLR embeddings
  data/
    DTS_FedTaxDpst_20051003_20230213.csv  -- Treasury Daily Tax Statement withheld tax deposits
    tri_000000_NSA.csv                    -- BLS CES Total Nonfarm Employment vintage table (NSA)
```

## Requirements

```
numpy
matplotlib
pandas
scipy
torch
torchvision
tqdm
```

## Notebooks

### synthetic_data_exp.ipynb

Reproduces **Figure 2** and the score comparison in **Appendix G**.

- **Cell 0 (Section 5.1, Figure 2):** N=4000, d=5, M=100 Monte Carlo runs. Generates categorical data X ~ Uniform{0..4}, Y ~ P[X] for a random experiment matrix P. Applies six manipulation policies at varying corruption rates (keep-prob p in [0.5, 1.0]) and plots the plug-in Gram determinant score against p, Hamming distance, and L2 norm. All six policies are ranked consistently by the score.

- **Cell 1 (Appendix G):** Compares the Gram determinant score against four alternatives (topd_volume, max_correlation, kyfan_sum, chi2_mi) across five manipulation policies, M=80 runs.

### additional_synthetic_data_exp.ipynb

Supplemental experiments varying kernel type (delta vs. Gaussian), manipulation type (uniform random vs. Gaussian noise), experimental setup (Exp 1 categorical Y vs. Exp 2 continuous Y), and sample size.

| Cell | Setting | Kernel | Manipulation |
|------|---------|--------|--------------|
| 0  | Exp 1 (N=10000) | Delta | Uniform random |
| 1  | Exp 1 (N=10000) | Delta | Gaussian noise (sigma_0) |
| 2  | Exp 1 (N=10000) | Gaussian | Uniform random |
| 3  | Exp 1 (N=10000) | Gaussian | Gaussian noise |
| 4  | Exp 2 (N=1000, d=4) | Gaussian | Uniform random |
| 5  | Exp 2 (N=1000, d=4) | Gaussian | Gaussian noise |
| 6  | Exp 2 (N=1000, d=4) | Delta (quantile buckets) | Uniform random |
| 7  | Exp 2 (N=1000, d=4) | Delta (quantile buckets) | Gaussian noise |
| 8  | **Figure 3**: sample-size study, Exp 1 | Gaussian | Uniform random |
| 9  | Sample-size study, Exp 1 | Gaussian | Gaussian noise |
| 10 | Sample-size study, Exp 2 | Gaussian | Uniform random |
| 11 | Sample-size study, Exp 2 | Delta (quantile buckets) | Gaussian noise |

Figure 3 (cells 8-9) plots the fraction of Monte Carlo trials in which the score ranking matches the ground-truth manipulation-severity ordering, for N in {250, 500, 1000, 2000, 4000}.

### employment_data_exp.ipynb

Reproduces **Table 1** (Section 5.3).

Uses real BLS CES employment data (three vintages of Total Nonfarm employment) as reported data x-hat, and Treasury withheld income and employment tax deposits as external observations y.

- **Cell 0:** Loads and aligns the vintage table and Treasury DTS series. Produces `merged_XY1` with columns X1_first, X2_after1m, X3_final, Y1.
- **Cell 1:** Computes month-over-month increments, discretizes into d=4 quantile buckets, and computes the plug-in Gram determinant score for each vintage. Expected ordering: Final > 1-Month Revision > First Release, matching the paper's Table 1 values (3.504e6, 24.920e6, 33.919e6).

### image_data_exp.ipynb

Reproduces **Figure 4** (Section 5.2).

Uses CIFAR-10 images with a SimCLR ResNet-18 encoder (8-dim projections) as observations y. Class labels serve as x, and manipulated labels as x-hat. The kernelized score uses the inner product kernel K(y,y') = <y,y'>.

- **Cell 0:** Trains SimCLR (ResNet-18, projection_dim=8, InfoNCE loss, tau=0.5, 60 epochs, batch=256, lr=5e-3, Adam) on CIFAR-10 training set. Extracts L2-normalized 8-dim embeddings for the 10000 test images.
- **Cell 1 (Figure 4):** Computes the kernelized score (convex hull volume of per-class mean embeddings) for the random manipulation policy across p in [0.6, 1.0], 101 seeds. Plots Figure 4 (p, Hamming, L2 vs. score).
- **Cell 2:** Sample-size ranking study in the image setting using 2D synthetic embeddings (class centroids on a circle) and all six manipulation policies.

## Data

**Treasury DTS (`DTS_FedTaxDpst_20051003_20230213.csv`):** Daily Treasury Statement federal tax deposit records from 2005-10-03 to 2023-02-13. The experiment uses the "Withheld Income and Employment Taxes" category, aggregated to monthly totals.

**BLS CES vintage table (`tri_000000_NSA.csv`):** Total Nonfarm Employment (not seasonally adjusted), BLS CES vintage releases. Each column is a vintage snapshot; the code extracts the first release, one-month revision, and final (benchmark) value for each reference month.

## How to Run

Run the notebooks in order within their respective sections. The employment notebook requires the two CSV files in `../data/` relative to the notebooks directory. The image notebook downloads CIFAR-10 automatically on first run and saves `simclr_model.pth` in the working directory.

```bash
jupyter notebook notebooks/synthetic_data_exp.ipynb
jupyter notebook notebooks/additional_synthetic_data_exp.ipynb
jupyter notebook notebooks/employment_data_exp.ipynb
jupyter notebook notebooks/image_data_exp.ipynb
```

The employment and synthetic notebooks run in minutes on CPU. The image notebook requires a GPU for the SimCLR training step (cell 0); evaluation and scoring (cells 1-2) run on CPU.

## Citation

```bibtex
@article{chen2025datareliability,
  title={Data Reliability Scoring},
  author={Chen, Yiling and Feng, Shi and Kattuman, Paul and Yu, Fang-Yi},
  journal={arXiv preprint arXiv:2510.17085},
  year={2025}
}
```
