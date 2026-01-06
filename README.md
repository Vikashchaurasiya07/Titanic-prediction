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

# 📝 Project Overview
The goal of this project is to predict survival for Titanic passengers. The workflow includes:

1. Data cleaning and preprocessing  
2. Feature engineering (e.g., extracting titles from names)  
3. Encoding categorical variables  
4. Training a **Logistic Regression** model  
5. Model evaluation and visualization  
6. Saving the trained model as `titanic_logistic_model.pkl`

---

# 📊 Dataset
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

# ⚙️ Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/Vikashchaurasiya07/Titanic-prediction.git
cd Titanic-prediction
run the file titanic_prediction.py
````

---

# 🚀 Usage

### 1️⃣ Using the saved model in Python

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('titanic_logistic_model.pkl')

# Load new data
data = pd.read_csv('Titanic-Dataset.csv')

# Preprocess new data (fill missing Age, encode Sex and Embarked, etc.)
# Example preprocessing:
# data['Age'].fillna(data['Age'].mean(), inplace=True)
# data['Sex'] = data['Sex'].map({'male':0, 'female':1})
# data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
# data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
# data = pd.get_dummies(data, columns=['Title'], drop_first=True)

predictions = model.predict(data)
print(predictions)
```

### 2️⃣ Run the prediction script

```bash
python titanic_prediction.py
```

---

# 🧠 Model

* **Algorithm**: Logistic Regression
* **Solver**: saga
* **Max iterations**: 1000
* **Features used**: Age, Sex, Pclass, SibSp, Parch, Fare, Embarked, Title (from Name)

Model saved as: `titanic_logistic_model.pkl`

---

# 📈 Evaluation

The model was evaluated using:

* **Accuracy**
* **Confusion Matrix**
* **Classification Report**

Example output:

```
Accuracy: 0.8156424581005587
Confusion Matrix:
 [[88 17]
 [16 58]]
Classification Report:
               precision    recall  f1-score   support

           0       0.85      0.84      0.84       105
           1       0.77      0.78      0.78        74

    accuracy                           0.82       179
   macro avg       0.81      0.81      0.81       179
weighted avg       0.82      0.82      0.82       179

```

---

# 📊 Visualizations

* Confusion Matrix heatmap
* Actual vs Predicted counts bar chart

> screenshots:


<img width="476" height="390" alt="confussion_matrix" src="https://github.com/user-attachments/assets/ddb5d183-2fba-4e16-b31c-207ad79bc466" />

<img width="590" height="390" alt="output" src="https://github.com/user-attachments/assets/a74c268c-817b-4b8a-a739-24da10bb440d" />



---

# 🛠️ Technologies

* Python 3.x
* pandas, numpy
* scikit-learn
* matplotlib
* joblib

---

# 📂 Files

| File                       | Description                                                   |
| -------------------------- | ------------------------------------------------------------- |
| Titanic-Dataset.csv        | Original dataset                                              |
| titanic_prediction.py      | Script to run predictions                                     |
| titanic_logistic_model.pkl | Saved trained model                                           |
| Titanic_Predicition.ipynb  | Jupyter Notebook with preprocessing, training, and evaluation |

---

# 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

