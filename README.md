

````markdown
# Titanic Survival Prediction

This project predicts the survival of passengers aboard the Titanic using machine learning. A logistic regression model is trained on historical Titanic data to estimate whether a passenger would survive based on features like age, sex, class, and other information.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Evaluation](#evaluation)
- [Technologies](#technologies)
- [Files](#files)
- [License](#license)

---

## Project Overview
The goal of this project is to predict whether a passenger survived the Titanic disaster. The dataset contains various features such as passenger class, sex, age, and port of embarkation. The project includes data preprocessing, feature engineering, model training, and evaluation.

---

## Dataset
The dataset used is `Titanic-Dataset.csv` and includes the following key features:

- `PassengerId` – Unique ID for each passenger  
- `Pclass` – Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)  
- `Name` – Passenger’s name  
- `Sex` – Gender (male/female)  
- `Age` – Age in years  
- `SibSp` – # of siblings/spouses aboard  
- `Parch` – # of parents/children aboard  
- `Ticket` – Ticket number  
- `Fare` – Passenger fare  
- `Cabin` – Cabin number  
- `Embarked` – Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)  
- `Survived` – Target variable (0 = Died, 1 = Survived)

> Missing values were handled using mean/median/mode imputation, and categorical features were encoded for model training.

---

## Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/Vikashchaurasiya07/Titanic-prediction.git
cd Titanic-prediction
````

---

## Usage

### 1️⃣ Run predictions from Python

You can use the saved logistic regression model (`titanic_logistic_model.pkl`) in a Python script:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('titanic_logistic_model.pkl')

# Load new data
data = pd.read_csv('Titanic-Dataset.csv')

# Preprocess new data (matching training preprocessing)
# Example: fill missing Age, encode Sex and Embarked, etc.

predictions = model.predict(data)
print(predictions)
```

### 2️⃣ Run the prediction script

```bash
python titanic_prediction.py
```

---

## Model

* **Algorithm**: Logistic Regression
* **Solver**: saga
* **Max iterations**: 1000
* **Features used**: Age, Sex, Pclass, SibSp, Parch, Fare, Embarked, Title (from Name)

The model is saved as `titanic_logistic_model.pkl` for reuse.

---

## Evaluation

The model was evaluated using:

* **Accuracy**
* **Confusion Matrix**
* **Classification Report**

Visualizations include:

* Confusion matrix heatmap
* Actual vs predicted counts comparison

Example accuracy:

```
Accuracy: 0.82
Confusion Matrix:
[[90 15]
 [18 50]]
```

---

## Technologies

* Python 3.x
* pandas, numpy
* scikit-learn
* matplotlib
* joblib (for saving/loading the model)

---

## Files

* `Titanic-Dataset.csv` – Original dataset
* `titanic_prediction.py` – Python script for running predictions
* `titanic_logistic_model.pkl` – Saved logistic regression model
* `Titanic_Predicition.ipynb` – Jupyter notebook with preprocessing, model training, and evaluation

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```

---

If you want, I can also create a **short, GitHub-friendly version with badges, screenshots, and sections highlighted**, which makes your repo look **super professional** — perfect for showcasing your Titanic ML project.  

Do you want me to do that version too?
```
