# Imports
import streamlit as st
import pandas as pd
import pickle

df_clean = pd.read_csv('mimic_iv_cleaned.csv')
# Load the model
loaded_model = pickle.load(open('final_model.pkl', 'rb'))

# Opening intro text
st.write("# Calculate Outcome for Cirrhosis Patient")

parient_choice = st.radio(
    "Example patient choice",
    ('Survived', 'Perished'))

if parient_choice == 'Survived':
    selected_patient_data = df_clean[df_clean.target == 1].sample()
else:
    selected_patient_data = df_clean[df_clean.target == 0].sample()

X = selected_patient_data.drop(['target'], axis=1)
y = selected_patient_data['target']

# Now - predicting!
if st.button(label="Click to Predict"):
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
