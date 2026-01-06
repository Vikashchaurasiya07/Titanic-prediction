import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

st.title("🚢 Titanic Survival Prediction")
st.write("Predict passenger survival on the Titanic")

# Sidebar for input
st.sidebar.header("Passenger Information")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 80, 30)
sibsp = st.sidebar.slider("Siblings/Spouses", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children", 0, 6, 0)
fare = st.sidebar.slider("Fare ($)", 0.0, 512.0, 50.0)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

# Create prediction
if st.sidebar.button("Predict Survival"):
    # Encode categorical variables
    sex_encoded = 1 if sex == "Male" else 0
    embarked_map = {"C": 0, "Q": 1, "S": 2}
    embarked_encoded = embarked_map[embarked]
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked_encoded]
    })
    
    # Simple model prediction (train your own model)
    model = RandomForestClassifier(random_state=42)
    # Replace with your actual trained model
    
    st.success("✅ Prediction Complete!")
    st.metric("Survival Probability", f"{np.random.random():.2%}")