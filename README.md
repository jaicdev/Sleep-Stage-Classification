# Sleep Stage Classification

This repository provides Python scripts for sleep stage classification using EEG data. It includes code for both Exploratory Data Analysis (EDA) and machine learning models (with and without class balancing). These scripts are based on MNE-Python and SciKit-Learn. The classification scripts also support running the analysis using different wavelet transforms, such as Daubechies (`db4`), Coiflet (`coif5`), and Biorthogonal (`bior1.3`).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Wavelet Variants](#wavelet-variants)
- [Scripts](#scripts)
  - [SSC_EDA.py](#sscedapy)
  - [SSCwithSMOTE.py](#sscwithsmotepy)
  - [SSCwithclassbalance.py](#sscwithclassbalancepy)
  - [SSCwithoutclassbalance.py](#sscwithoutclassbalancepy)
  
## Overview
The project contains four main scripts:
1. `SSC_EDA.py`: Performs exploratory data analysis (EDA) on EEG data, including signal plotting, power spectral density, event distributions, and channel correlations.
2. `SSCwithSMOTE.py`: A machine learning pipeline for classifying sleep stages using SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance and support for multiple wavelet transforms.
3. `SSCwithclassbalance.py`: A machine learning pipeline for classifying sleep stages using class weighting to handle class imbalance and support for multiple wavelet transforms.
4. `SSCwithoutclassbalance.py`: Similar to the previous scripts, but without class balancing, and also supporting multiple wavelet transforms.

## Requirements
Before running the scripts, ensure you have the following Python packages installed:
```bash
pip install numpy matplotlib seaborn mne scipy scikit-learn pywavelets imbalanced-learn
```

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

To run the classification scripts with wavelet support, you can specify the wavelet type as follows:
```bash
python SSCwithclassbalance.py --wavelet coif5
```
If no wavelet is specified, the default wavelet (`db4`) is used.

## Wavelet Variants
The classification scripts (`SSCwithSMOTE.py`, `SSCwithclassbalance.py`, and `SSCwithoutclassbalance.py`) support different wavelet families for feature extraction using the PyWavelets library. You can use the following wavelet variants:
- **Daubechies (db)**: For example, `db4`
- **Coiflet (coif)**: For example, `coif5`
- **Biorthogonal (bior)**: For example, `bior1.3`

You can specify the wavelet to use by passing it as a command-line argument when running the script:
```bash
python SSCwithSMOTE.py --wavelet coif5
```

If no wavelet is specified, the default wavelet `db4` is used.

## Scripts

### SSC_EDA.py

This script performs exploratory data analysis on sleep stage EEG data, providing various visualizations:
- Raw EEG signal plots
- Power Spectral Density (PSD) plots
- Event distribution (sleep stages)
- Epoch variance and mean distributions
- Correlation matrix between EEG channels

#### Example
```bash
python SSC_EDA.py
```

#### Expected Outputs
- Plots of raw EEG signals and their PSD
- Bar charts for sleep stage distribution
- Histograms for epoch variance and mean
- Correlation matrix heatmap

### SSCwithSMOTE.py

This script performs sleep stage classification using machine learning models with SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance. It supports Random Forest, Gradient Boosting, and Support Vector Machine (SVM) classifiers. You can also choose which wavelet to use for feature extraction by passing it as a command-line argument.

#### Example
```bash
python SSCwithSMOTE.py --wavelet coif5
```

#### Expected Outputs
- Confusion matrices and classification reports for each model
- Model comparison in terms of accuracy

#### Command-line Arguments
- `--wavelet`: Specify the wavelet type to use. Supported wavelets include `db4`, `coif5`, `bior1.3`. Default is `db4`.

### SSCwithclassbalance.py

This script performs sleep stage classification using machine learning models with class weighting to handle class imbalance. It supports Random Forest, Gradient Boosting, and Support Vector Machine (SVM) classifiers. You can also choose which wavelet to use for feature extraction by passing it as a command-line argument.

#### Example
```bash
python SSCwithclassbalance.py --wavelet coif5
```

#### Expected Outputs
- Confusion matrices and classification reports for each model
- Model comparison in terms of accuracy

#### Command-line Arguments
- `--wavelet`: Specify the wavelet type to use. Supported wavelets include `db4`, `coif5`, `bior1.3`. Default is `db4`.

### SSCwithoutclassbalance.py

This script performs sleep stage classification using machine learning models **without any class balancing techniques**. It supports Random Forest, Gradient Boosting, and SVM classifiers, and allows for wavelet-based feature extraction.

#### Example
```bash
python SSCwithoutclassbalance.py --wavelet bior1.3
```

#### Expected Outputs
- Confusion matrices and classification reports for each model
- Model comparison in terms of accuracy

#### Command-line Arguments
- `--wavelet`: Specify the wavelet type to use. Supported wavelets include `db4`, `coif5`, `bior1.3`. Default is `db4`.

## License
This project is licensed under the MIT License.
