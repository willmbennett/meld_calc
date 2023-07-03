# Imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,  make_column_selector as selector
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImPipeline
import pickle

df_clean = pd.read_csv('mimic_iv_cleaned.csv')
X = df_clean.drop(['target'], axis=1)
y = df_clean['target']

# Now - predicting!
if st.button(label="Click to Predict"):

    # Load the model
    loaded_model = pickle.load(open('final_model.pkl', 'rb'))
    # Make predictions (and get out pred probabilities)
    pred = loaded_model.predict(X[:1])[0]
    proba = loaded_model.predict_proba(X[:1])[:,1][0]
    
    # Sharing the predictions
    if pred == 0:
        st.write("### The person is predicted to survive 90 days")
        st.write(f"Predicted probability of dying: {proba*100:.2f} %")

    elif pred == 1:
        st.write("### The person is NOT predicted to survive 90 days!")
        st.write(f"Predicted probability of dying: {proba*100:.2f} %")
