# Imports
import streamlit as st
import pandas as pd
import pickle

df_clean = pd.read_csv('data/mimic_iv_cleaned.csv')
# Load the model
loaded_model = pickle.load(open('models/final_model.pkl', 'rb'))

# Opening intro text
st.write("# Calculate Outcome for Cirrhosis Patient")

parient_choice = st.radio(
    "Actual outcome:",
    ('Survived', 'Perished'))

if parient_choice == 'Survived':
    selected_patient_data = df_clean[df_clean.target == 0].sample(random_state=42)
else:
    selected_patient_data = df_clean[df_clean.target == 1].sample(random_state=42)

X = selected_patient_data.drop(['target'], axis=1)
y = selected_patient_data['target']
st.write(f'patient index {X.index}')

st.write(f"Total Number of features:", len(X.columns))

num_cols = {'inr_min',
            'pt_min',
            'ptt_min',
            'bun_min',
            'bilirubin_total_min',
            'bilirubin_total_max',
            'bun_max',
            'inr_max'}

cat_cols = ['gender', 'race']

# INR 
st.write(f"### International Normalised Ratio (INR)")
st.write('A normal INR is 1.0. Each increase of 0.1 means the blood is slightly thinner (it takes longer to clot). INR is related to the prothrombin time (PT).')
st.write('[Veteran Affairs](https://www.hepatitis.va.gov/hcv/patient/diagnosis/labtests-INR.asp#:~:text=A%20normal%20INR%20is%201.0,the%20prothrombin%20time%20(PT).)')

inr_label = 'Min:'
inr_default_value = X['inr_min'].iloc[0]
inr_min = st.number_input(inr_label, value=inr_default_value, step=0.1)
st.write('New INR Min:', inr_min)
X['inr_min'] = inr_min

st.write(f"## Predict Patient Outcome:")

# Now - predicting!
# if st.button(label="Click to Predict"):
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
