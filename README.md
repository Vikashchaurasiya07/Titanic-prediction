# 🚢 Titanic Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project predicts whether a passenger survived the Titanic disaster using **machine learning**. A logistic regression model is trained on historical Titanic data to estimate survival based on passenger features like age, sex, class, and port of embarkation.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Technologies](#technologies)
- [Files](#files)
- [License](#license)

---

## 📝 Project Overview
The goal of this project is to predict survival for Titanic passengers. The workflow includes:

1. Data cleaning and preprocessing
2. Feature engineering (e.g., extracting titles from names)
3. Encoding categorical variables
4. Training a **Logistic Regression** model
5. Model evaluation and visualization
6. Saving the trained model as `titanic_logistic_model.pkl`

---

## 📊 Dataset
Dataset: `Titanic-Dataset.csv`  

Key features:

| Feature | Description |
|---------|-------------|
| PassengerId | Unique ID for each passenger |
| Pclass | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Name | Passenger name |
| Sex | Gender (male/female) |
| Age | Age in years |
| SibSp | # of siblings/spouses aboard |
| Parch | # of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |
| Survived | Target variable (0 = Died, 1 = Survived) |

> Missing values handled using mean/mode imputation. Categorical features encoded for model training.

---

## ⚙️ Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/Vikashchaurasiya07/Titanic-prediction.git
cd Titanic-prediction
pip install -r requirements.txt
