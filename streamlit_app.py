# Imports
import streamlit as st
import pandas as pd
import pickle
import numpy as np

key_cols = ['inr_min', 'aniongap_min', 'bun_min', 'bilirubin_total_min', 'age', 'gender', 'race']

# Initialization
if 'inr_min' not in st.session_state:
    st.session_state.inr_min = float(0)
if 'aniongap_min' not in st.session_state:
    st.session_state.aniongap_min = float(0)
if 'bun_min' not in st.session_state:
    st.session_state.bun_min = float(0)
if 'bilirubin_total_min' not in st.session_state:
    st.session_state.bilirubin_total_min = float(0)
if 'age' not in st.session_state:
    st.session_state.age = int(0)
if 'gender' not in st.session_state:
    st.session_state.gender = 'F'
if 'race' not in st.session_state:
    st.session_state.race = 'ASIAN'
if 'X' not in st.session_state:
    st.session_state.X = pd.DataFrame()
if 'pred' not in st.session_state:
    st.session_state.pred = int(0)
if 'proba' not in st.session_state:
    st.session_state.proba = float(0)

df_clean = pd.read_csv('data/mimic_iv_cleaned.csv')
# Load the model
loaded_model = pickle.load(open('models/XGB_SFM.pkl', 'rb'))

def make_predictions():
    st.session_state.pred = loaded_model.predict(st.session_state.X)[0]
    st.session_state.proba = loaded_model.predict_proba(st.session_state.X)[:,1][0]

# Opening intro text
st.write("# Will's Modified MELD Calculator")

st.write(st.session_state)

if st.button('Load patient that survived'):
    st.session_state.X = df_clean[df_clean.target == 0].sample(random_state=20).drop(['target'], axis=1)
    for col in key_cols:
        st.session_state[col] = st.session_state.X.iloc[0,col]
    make_predictions()
    
    
if st.button('Load patient that died'):
    st.session_state.X = df_clean[df_clean.target == 1].sample(random_state=20).drop(['target'], axis=1)
    for col in key_cols:
        st.session_state[col] = st.session_state.X.iloc[0,col]
    make_predictions()

# Sidebar

with st.sidebar:
    # INR Min
    with st.expander("International Normalised Ratio (INR) Min"):
        st.write('A normal INR is 1.0. Each increase of 0.1 means the blood is slightly thinner (it takes longer to clot). INR is related to the prothrombin time (PT).')
        st.write('[Veteran Affairs](https://www.hepatitis.va.gov/hcv/patient/diagnosis/labtests-INR.asp#:~:text=A%20normal%20INR%20is%201.0,the%20prothrombin%20time%20(PT).)')
        
        def inr_update():
            st.session_state.X['inr_min'] = st.session_state.inr_min
            make_predictions()
            
        st.number_input('INR Min:', 
                        step=0.1,
                        key='inr_min',
                        on_change=inr_update
                      )
        
        st.image('images/inr_min.png')

    # Anion Gap Min
    with st.expander("Anion Gap"):
        st.write("An anion gap blood test checks the acid-base balance of your blood and if the electrolytes in your blood are properly balanced.")
        st.write("There’s no universal “normal” anion gap, partly because laboratories and healthcare providers can measure and compare different electrolytes in your blood.")
        st.write('[Mayo Clinic](https://my.clevelandclinic.org/health/diagnostics/22041-anion-gap-blood-test)')
        
        def aniongap_update():
            st.session_state.X['aniongap_min'] = st.session_state.aniongap_min
            make_predictions()
            
        st.number_input('Anion Gap Min:', 
                        step=0.1,
                        key='aniongap_min',
                        on_change=aniongap_update
                      )
        
        st.image('images/aniongap_min.png')
        
    # Bun Min
    with st.expander("Blood urea nitrogen (BUN) Min"):
        st.write("A common blood test, the blood urea nitrogen (BUN) test reveals important information about how well your kidneys are working. A BUN test measures the amount of urea nitrogen that's in your blood.")
        st.write("In general, around 6 to 24 mg/dL (2.1 to 8.5 mmol/L) is considered normal.")
        st.write('[Mayo Clinic](https://www.mayoclinic.org/tests-procedures/blood-urea-nitrogen/about/pac-20384821)')
        
        def bun_min_update():
            st.session_state.X['bun_min'] = st.session_state.bun_min
            make_predictions()
            
        st.number_input('BUN Min:', 
                        key='bun_min',
                        on_change=bun_min_update
                      )
        
        st.image('images/bun_min.png')
        
    # Bilirubin test
    with st.expander("Total Bilirubin Min"):
        st.write("Bilirubin (bil-ih-ROO-bin) is a yellowish pigment that is made during the breakdown of red blood cells. Bilirubin passes through the liver and is eventually excreted out of the body. Higher than usual levels of bilirubin may indicate different types of liver or bile duct problems.")
        st.write("Typical results for a total bilirubin test are 1.2 milligrams per deciliter (mg/dL) for adults and usually 1 mg/dL for those under 18.")
        st.write('[Mayo Clinic](https://www.mayoclinic.org/tests-procedures/bilirubin/about/pac-20393041)')
        
        def bilirubin_total_min_update():
            st.session_state.X['bilirubin_total_min'] = st.session_state.bilirubin_total_min
            make_predictions()
            
        st.number_input('Bilirubin Total Min:',
                        step=0.1,
                        key='bilirubin_total_min',
                        on_change=bilirubin_total_min_update
                      )
        
    # Age
    with st.expander("Age"):
        def age_update():
            st.session_state.X['age'] = st.session_state.age
            make_predictions()
            
        st.number_input('Age:', 
                        step=5,
                        key='age',
                        on_change=age_update)
        st.image('images/age.png')

    # Gender
    with st.expander("Gender"):
        def gender_update():
            st.session_state.X['gender'] = st.session_state.gender
            make_predictions()
            
        st.selectbox("Gender:",
                 ('M', 'F'),
                 key='gender',
                 on_change=gender_update)
        st.image('images/gender.png')

    # Race
    with st.expander("Race"):
        def race_update():
            st.session_state.X['race'] = st.session_state.race
            make_predictions()
                
        race_list = list(df_clean.groupby('race').count().sort_values(by='gender', ascending=False).reset_index()['race'])
        
        race = st.selectbox('Selected Race:', 
                            race_list,
                            key='race',
                            on_change=race_update
                           )
        
        st.image('images/race.png')

# Patient Information

st.write("### Patient Data:")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("INR Min", np.round(st.session_state.inr_min,1))
    st.metric("Bilirubin Total Min", np.round(st.session_state.bilirubin_total_min,1))

with col2:
    st.metric("Anion Gap Min", np.round(st.session_state.aniongap_min,1))
    st.metric("Age", st.session_state.age)

with col3:
    st.metric("BUN Min", np.round(st.session_state.bun_min,1))
    st.metric("Gender", st.session_state.gender)
    
st.metric("Race", st.session_state.race)

# Share the predictions
col1, col2 = st.columns(2)
with col1:
    # Sharing the predictions
    # st.write(f"The person is{' not' if st.session_state.pred == 1 else ''} predicted to survive 90 days.")
    st.metric(label="Model predicts person will die within 90 days:", value=f"{'YES' if st.session_state.pred == 1 else 'NO'}")
    st.metric(label="Likelihood to Die Within 90 Days", value=f"{st.session_state.proba*100:.1f} %")

with col2:
    st.image('images/medical_image.gif')
    st.write('[GIF credit](https://www.behance.net/gallery/73013043/Healthcare-animated-icons)')
