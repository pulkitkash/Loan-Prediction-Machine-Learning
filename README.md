# Loan Eligibility Prediction System 🏦

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An intelligent machine learning system that predicts loan eligibility with 88% accuracy, helping financial institutions make data-driven credit decisions.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Scope](#future-scope)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

Traditional loan evaluation is often manual, time-consuming, and prone to inconsistency. This project addresses these challenges by building an **automated, data-driven loan eligibility prediction system** that:

- Processes applicant data through a robust preprocessing pipeline
- Uses **XGBoost** (Gradient Boosting) for high-accuracy predictions
- Provides instant decisions via a user-friendly **Flask web interface**
- Handles real-world data issues (missing values, outliers, class imbalance)

### Business Impact
- ⚡ **90% reduction** in loan processing time
- 📈 **Consistent decisions** across all applications
- 🎯 **Data-driven insights** for credit risk assessment

## ✨ Key Features

### Core Functionality
| Feature | Description |
|---------|-------------|
| **ML-Powered Prediction** | XGBoost model analyzing 32 financial parameters |
| **Real-time Web Interface** | User-friendly form for instant loan eligibility checks |
| **High Accuracy** | 88% accuracy with 0.89 AUC-ROC |
| **Feature Engineering** | Loan-to-income ratio, EMI calculations, log transforms |

### Technical Highlights
- 🔄 **Complete preprocessing pipeline** (imputation, encoding, scaling)
- 📊 **5-fold cross-validation** for robust model selection
- 🎨 **Interactive dashboard** with prediction history
- 🔒 **Stateless API** with consistent feature alignment
- 📈 **Visual analytics** (confusion matrix, ROC curves, feature importance)

## 🛠 Technology Stack

### Core Technologies
```python
{
    "Language": "Python 3.8+",
    "ML Framework": "XGBoost, Scikit-learn",
    "Web Framework": "Flask",
    "Data Processing": "Pandas, NumPy",
    "Visualization": "Matplotlib, Seaborn",
    "Serialization": "Pickle"
}
