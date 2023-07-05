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
    "Actual outcome:",
    ('Survived', 'Perished'))

if parient_choice == 'Survived':
    selected_patient_data = df_clean[df_clean.target == 0].sample()
else:
    selected_patient_data = df_clean[df_clean.target == 1].sample()

X = selected_patient_data.drop(['target'], axis=1)
y = selected_patient_data['target']

st.write(f"Total Number of features: {len(X)} %")

key_cols = ['inr_min',
            'pt_min',
            'ptt_min',
            'bun_min',
            'bilirubin_total_min',
            'bilirubin_total_max',
            'bun_max',
            'inr_max',
            'gender',
            'race'
           ]

# INR 
inr_min = df_clean['inr_min'].min()
inr_max = df_clean['inr_max'].max()
inr = st.slider('International Normalised Ratio (INR):', inr_min, inr_max)
st.write(f"INR value set to {inr}')
X['inr_min'] = inr
X['inr_max'] = inr

st.write(f"## Predict Patient Outcome:")

# Now - predicting!
if st.button(label="Click to Predict"):
    # Make predictions (and get out pred probabilities)
    pred = loaded_model.predict(X)[0]
    proba = loaded_model.predict_proba(X)[:,1][0]
    
    # Sharing the predictions
    if pred == 0:
        st.write("### The person is predicted to survive 90 days")
        st.write(f"Predicted probability of dying: {proba*100:.2f} %")

    elif pred == 1:
        st.write("### The person is NOT predicted to survive 90 days!")
        st.write(f"Predicted probability of dying: {proba*100:.2f} %")
