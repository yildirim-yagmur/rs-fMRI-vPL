# Neural Consolidation of Visual Perceptual Learning (vPL)
# rs-fMRI

Preprocessing and analysis codebase for my Master's thesis investigating the neural mechanisms of visual perceptual learning (vPL) and spatial generalization. 

This repository features a surface-based longitudinal fMRI preprocessing pipeline (handling native surface projection and confound regression), Bayesian linear mixed-effects modeling to evaluate functional connectivity changes across the visual-parietal hierarchy, and whole-brain exploratory analyses using Network-Based Statistics (NBS).

## Repository Structure

The analysis is divided into three main Python scripts:

### 1. `denoising_pipeline.py` (Preprocessing & Denoising)
A longitudinal, surface-based resting-state fMRI denoising pipeline. 
* Extracts tissue-based confounds from WM/CSF masks.
* Builds a design matrix (1 Framewise Displacement + 10 WM/CSF components + cardiac physio).
* Projects volumetric functional data onto the native cortical surface (using longitudinal FreeSurfer templates to minimize stochastic processing noise).
* Denoises surface data using `nilearn.signal.clean` (simultaneous regression, bandpass filtering 0.008-0.09 Hz, linear detrending).
* Extracts region-of-interest (ROI) timeseries across visual and parietal hubs.

### 2. `Hypothesis_driven_Bayesian_Models.py` (Hypothesis Testing)
Evaluates targeted functional connectivity (FC) changes across the visual-parietal hierarchy using Bayesian hierarchical linear mixed-effects modeling (via PyMC).
* Computes Fisher Z-transformed Pearson correlations for specific ROI pathways (e.g., V1, V2, hV4, and IPS).
* Fits a series of Bayesian models (Null, Main Effects, Main Effects + Behavior, Full Interactions) to account for Group, Session, and Behavioral Learning Index (LI) effects.
* Utilizes `arviz` for posterior predictive checks, LOO-CV model comparison, and high-quality posterior distribution visualizations (ridge plots).

### 3. `exploratory_wholebrain.py` (Network-Based Statistics)
An exploratory, data-driven whole-brain analysis using the HCP-MMP1.0 parcellation.
* Computes full connectome FC matrices (edges: ~64,620) for all subjects.
* **Stage 1:** Mass-univariate edge-level Welch t-tests for Group x Session interactions and Session main effects with False Discovery Rate (BH-FDR) correction.
* **Stage 2 (NBS):** Permutation-based Network-Based Statistics (Zalesky et al., 2010). Tests for the largest connected-component extents among suprathreshold edges (10,000 permutations) to identify broad network-level plasticity without the harsh penalty of edge-level FDR.

## Core Dependencies
* `numpy`, `pandas`, `scipy`
* `nibabel`, `nilearn` (Neuroimaging I/O and processing)
* `pymc`, `arviz`, `pytensor` (Bayesian modeling and visualization)
* `pingouin`, `statsmodels` (Frequentist statistics and FDR correction)
* `matplotlib`, `seaborn` (Visualization)

## Author
**Yagmur Yildirim**  
Master's Thesis Project
