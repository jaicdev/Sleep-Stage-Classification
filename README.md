# Sleep Stage Classification and EDA

This repository provides Python scripts for sleep stage classification using EEG data. It includes code for both Exploratory Data Analysis (EDA) and machine learning models (with and without class balancing). These scripts are based on MNE-Python and SciKit-Learn.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Scripts](#scripts)
  - [SSC_EDA.py](#sscedapy)
  - [SSCwithclassbalance.py](#sscwithclassbalancepy)
  - [SSCwithoutclassbalance.py](#sscwithoutclassbalancepy)
  - [Dataset Information](#dataset-information)

## Overview
The project contains three main scripts:
1. `SSC_EDA.py`: Performs exploratory data analysis (EDA) on EEG data, including signal plotting, power spectral density, event distributions, and channel correlations.
2. `SSCwithclassbalance.py`: A machine learning pipeline for classifying sleep stages, with class balancing.
3. `SSCwithoutclassbalance.py`: Similar to the previous script, but without class balancing.

## Requirements
Before running the scripts, ensure you have the following Python packages installed:
```bash
pip install numpy matplotlib seaborn mne scipy scikit-learn pywavelets


## Usage

### Data Preparation
- The scripts expect EEG data in `.edf` format. You'll need a folder containing `.edf` files for sleep scoring.
- Place your `.edf` files in a folder and update the `folder_path` in the scripts to point to your data folder.

### Running the Scripts
To run any script, simply execute it in your terminal:
```bash
python <script_name>.py
```

For example, to run the `SSC_EDA.py` script:
```bash
python SSC_EDA.py
```

## Scripts

### SSC_EDA.py

This script performs exploratory data analysis on sleep stage EEG data, providing various visualizations:
- Raw EEG signal plots
- Power Spectral Density (PSD) plots
- Event distribution (sleep stages)
- Epoch variance and mean distributions
- Correlation matrix between EEG channels

#### Example
```python
python SSC_EDA.py
```

#### Expected Outputs
- Plots of raw EEG signals and their PSD
- Bar charts for sleep stage distribution
- Histograms for epoch variance and mean
- Correlation matrix heatmap

### SSCwithclassbalance.py

This script performs sleep stage classification using machine learning models, with class balancing. It uses Random Forest, Gradient Boosting, and Support Vector Machine (SVM) classifiers.

#### Example
```python
python SSCwithclassbalance.py
```

#### Expected Outputs
- Confusion matrices and classification reports for each model
- Model comparison in terms of accuracy

### SSCwithoutclassbalance.py

Similar to the above, but without class balancing. This script uses the same machine learning models (Random Forest, Gradient Boosting, and SVM) and evaluates their performance.

#### Example
```python
python SSCwithoutclassbalance.py
```

#### Expected Outputs
- Confusion matrices and classification reports for each model
- Model comparison in terms of accuracy

## Dataset Information

The data used for this project is the **Haaglanden Medisch Centrum Sleep Staging Database** available through [PhysioNet](https://physionet.org/content/hmc-sleep-staging/1.1/). This dataset consists of 151 polysomnographic sleep recordings collected from patients referred for PSG examinations at the HMC sleep center in 2018.

### How to Cite

When using this resource, please cite:
```
Alvarez-Estevez, D., & Rijsman, R. (2022). Haaglanden Medisch Centrum sleep staging database (version 1.1). PhysioNet. https://doi.org/10.13026/t79q-fr32
```

Additionally, please cite the original publication:
```
Alvarez-Estevez D, Rijsman RM (2021) Inter-database validation of a deep learning approach for automatic sleep scoring. PLoS ONE 16(8): e0256111. https://doi.org/10.1371/journal.pone.0256111
```

The standard citation for PhysioNet is:
```
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
```

### Dataset Description
- **Subjects**: 151 PSG recordings (85 male, 66 female), average age of 53.9 ± 15.4.
- **Signals**: EEG, EOG, EMG, ECG.
- **Format**: Data is provided in the EDF format.
- **Annotation**: Hypnograms are provided for sleep stages (W, N1, N2, N3, REM).

## License
This project is licensed under the MIT License.
