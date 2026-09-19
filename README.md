# Mammography Mass Classification 

## Overview

This Streamlit application provides an interactive interface for a
CNN model that classifies mammographic masses as benign or
malignant.

The application supports two trained CNN models:

- Cropped Abnormality Model
- Full Mammography Model


## Requirements

Python 3.x is required.

Install the required packages using:

```bash
pip install -r requirements.txt

```
### Breast Tumor Classification & Detection

This repository contains an end-to-end data science and machine learning pipeline focused on detecting and classifying breast tumors (Malignant vs. Benign). Leveraging predictive modeling, the project analyzes diagnostic features to provide automated, reliable insights that assist in medical imaging assessment and diagnostic workflows. 

### Features

* **Data Preprocessing & Cleaning:** Comprehensive handling of missing data, feature scaling (Standardization/Normalization), and class imbalance adjustments.
* **Exploratory Data Analysis (EDA):** Insightful visual graphs charting feature correlations, distribution ranges, and variance profiles between benign and malignant cases.
* **Multi-Model Training Pipeline:** Implementation and comparative benchmarking of industry-standard diagnostic models.
* **Performance Metrics Evaluation:** Rigorous evaluation metrics reporting Accuracy, Precision, Recall (Sensitivity), F1-Score, and Confusion Matrix profiles.

### Repository Structure

* breast cancer detection.ipynb: The primary Jupyter Notebook containing dataset exploration, exploratory engineering, model training, and performance validation graphs.
* README.md: Documentation outlining setup instruction guidelines and project architectures.

### Getting Started

Follow these steps to set up the local environment and run the notebook workspace. 

### Prerequisites

Ensure you have **Python 3.8+** installed along with standard data science tooling packages. 

### Installation & Environment Setup

1. **Clone the repository:** 

bash

git clone https://github.com/gladyssoumaa/breast-tumor-classification.git
cd breast-tumor-classification

Use code with caution.
2. **Create and activate a virtual environment (Optional but Recommended):** 

bash

python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Use code with caution.
3. **Install Core Dependencies:** 

bash

pip install jupyter numpy pandas matplotlib seaborn scikit-learn

Use code with caution.

*(Note: If your notebook utilizes Deep Learning networks, additionally install tensorflow or torch depending on your implementation framework).*

### Running the Notebook

Launch the local interactive environment interface server: 

bash

jupyter notebook

Use code with caution.

Click open breast cancer detection.ipynb from your browser directory list and execute cells sequentially. 

### Model Architecture & Methodology

The analytical workflow within the notebook is structured as follows: 

1. **Feature Selection:** Reducing dimensionality by identifying highly correlated morphological variations (e.g., radius, texture, perimeter smoothness).
2. **Model Optimization:** Training architectures (such as **Logistic Regression, Support Vector Machines, Random Forests**, or **Convolutional Neural Networks**) to isolate tumor margins.
3. **Validation:** Tuning parameters via Cross-Validation partitions to eliminate predictive bias and guard against overfitting.
