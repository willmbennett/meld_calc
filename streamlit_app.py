# Imports
import streamlit as st
import pandas as pd
import pickle
import numpy as np

df_clean = pd.read_csv('data/mimic_iv_cleaned.csv')
# Load the model
loaded_model = pickle.load(open('models/XGB_SFM.pkl', 'rb'))

col1, col2, col3 = st.columns(3)
col2.image('images/medical_image.gif')

# Opening intro text
st.write("# Will's Modified MELD Calculator")

patient_choice = st.radio(
    "Actual outcome:",
    ('Survived', 'Perished'))

if patient_choice == 'Survived':
    selected_patient_data = df_clean[df_clean.target == 0].sample(random_state=20)
else:
    selected_patient_data = df_clean[df_clean.target == 1].sample(random_state=20)

X = selected_patient_data.drop(['target'], axis=1)
y = selected_patient_data['target']

# Sidebar

with st.sidebar:
    # INR Min
    with st.expander("International Normalised Ratio (INR) Min"):
        st.write('A normal INR is 1.0. Each increase of 0.1 means the blood is slightly thinner (it takes longer to clot). INR is related to the prothrombin time (PT).')
        st.write('[Veteran Affairs](https://www.hepatitis.va.gov/hcv/patient/diagnosis/labtests-INR.asp#:~:text=A%20normal%20INR%20is%201.0,the%20prothrombin%20time%20(PT).)')
        
        inr_min_label = 'INR Min:'
        inr_min_default_value = X['inr_min'].iloc[0]
        inr_min = st.number_input(inr_min_label, value=inr_min_default_value, step=0.1)
        X['inr_min'] = inr_min

    # Anion Gap Min
    with st.expander("Anion Gap"):
        st.write("An anion gap blood test checks the acid-base balance of your blood and if the electrolytes in your blood are properly balanced.")
        st.write("There’s no universal “normal” anion gap, partly because laboratories and healthcare providers can measure and compare different electrolytes in your blood.")
        st.write('[Mayo Clinic](https://my.clevelandclinic.org/health/diagnostics/22041-anion-gap-blood-test)')
        
        aniongap_min_label = 'Anion Gap Min:'
        aniongap_min_default_value = X['aniongap_min'].iloc[0]
        aniongap_min = st.number_input(aniongap_min_label, value=aniongap_min_default_value, step=0.1)
        X['aniongap_min'] = aniongap_min
        
    # Bun Min
    with st.expander("Blood urea nitrogen (BUN) Min"):
        st.write("A common blood test, the blood urea nitrogen (BUN) test reveals important information about how well your kidneys are working. A BUN test measures the amount of urea nitrogen that's in your blood.")
        st.write("In general, around 6 to 24 mg/dL (2.1 to 8.5 mmol/L) is considered normal.")
        st.write('[Mayo Clinic](https://www.mayoclinic.org/tests-procedures/blood-urea-nitrogen/about/pac-20384821)')
        
        bun_min_label = 'BUN Min:'
        bun_min_default_value = X['bun_min'].iloc[0]
        bun_min = st.number_input(bun_min_label, value=bun_min_default_value, step=1.0)
        X['bun_min'] = bun_min
        
    # Bilirubin test
    with st.expander("Total Bilirubin Min"):
        st.write("Bilirubin (bil-ih-ROO-bin) is a yellowish pigment that is made during the breakdown of red blood cells. Bilirubin passes through the liver and is eventually excreted out of the body. Higher than usual levels of bilirubin may indicate different types of liver or bile duct problems.")
        st.write("Typical results for a total bilirubin test are 1.2 milligrams per deciliter (mg/dL) for adults and usually 1 mg/dL for those under 18.")
        st.write('[Mayo Clinic](https://www.mayoclinic.org/tests-procedures/bilirubin/about/pac-20393041)')
        
        bilirubin_total_min_label = 'Bilirubin Total Min:'
        bilirubin_total_min_default_value = X['bilirubin_total_min'].iloc[0]
        bilirubin_total_min = st.number_input(bilirubin_total_min_label, value=bilirubin_total_min_default_value, step=0.1)
        X['bilirubin_total_min'] = bilirubin_total_min
        
    # Age
    with st.expander("Age"):
        age_label = 'Age:'
        age_default_value = X['age'].iloc[0]
        age = st.number_input(age_label, value=age_default_value, step=5)
        X['age'] = age

    # Gender
    with st.expander("Gender"):
        gender_choice = st.radio(
        "Gender:",
        ('Male', 'Female'))

        if gender_choice == 'Male':
            X['gender'] = 'M'
        else:
            X['gender'] = 'F'

    # Race
    with st.expander("Race"):
        race_list = list(df_clean.groupby('race').count().sort_values(by='gender', ascending=False).reset_index()['race'])
        race_index = race_list.index(X['race'].values[0])
        race = st.selectbox('Selected Race:', 
                            race_list,
                            race_index
                           )
        X['race'] = race

# Patient Information
key_cols = ['inr_min',
            'aniongap_min',
            'bun_min',
            'bilirubin_total_min',
            'age',
            'gender',
            'race'
           ]

st.write("### Patient Data:")
# st.dataframe(data=X[key_cols], hide_index=True, use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="INR Min", value=X.inr_min)
    st.metric(label="Bilirubin Total Min", value=X.bilirubin_total_min)

with col2:
    st.metric(label="Anion Gap Min", value=X.aniongap_min)
    st.metric(label="Age", value=X.age)

with col3:
    st.metric(label="BUN Min", value=X.bun_min)
    st.metric(label="Gender", value=X.gender.values[0])


st.metric(label="Race", value=X.race.values[0])
    
# Make predictions (and get out pred probabilities)
pred = loaded_model.predict(X)[0]
proba = loaded_model.predict_proba(X)[:,1][0]

# Sharing the predictions
# st.write(f"The person is{' not' if pred == 1 else ''} predicted to survive 90 days.")
st.metric(label="Model predicts person will die within 90 days:", value=f"{'YES' if pred == 1 else 'NO'}")
st.metric(label="Likelihood to Die Within 90 Days", value=f"{proba*100:.1f} %")


st.write('[GIF credit](https://www.behance.net/gallery/73013043/Healthcare-animated-icons)')
