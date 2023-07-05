# Imports
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
import pickle

# Create a custom column selector to add in the clustering
from sklearn.base import BaseEstimator, TransformerMixin

class Kmean_cluster(BaseEstimator, TransformerMixin):
    '''select specific columns of a given dataset'''

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X)
        X_df['kmeans_cluster'] = kmeans
        return np.array(X_df)


df_clean = pd.read_csv('data/mimic_iv_cleaned.csv')
# Load the model
loaded_model = pickle.load(open('models/xgb_clus.pkl', 'rb'))

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


# Sidebar
num_cols = ['inr_min',
            'inr_max',
            'bun_min',
            'aniongap_min',
            'bilirubin_total_min',
            'Kmeans Cluster',
            'alp_min',
            'age',
            'ptt_min',
            'pt_min',
            'pt_max']

cat_cols = ['gender', 'race']

with st.sidebar:
    # INR Min
    with st.expander("International Normalised Ratio (INR) Min"):
        st.write('A normal INR is 1.0. Each increase of 0.1 means the blood is slightly thinner (it takes longer to clot). INR is related to the prothrombin time (PT).')
        st.write('[Veteran Affairs](https://www.hepatitis.va.gov/hcv/patient/diagnosis/labtests-INR.asp#:~:text=A%20normal%20INR%20is%201.0,the%20prothrombin%20time%20(PT).)')
        
        inr_min_label = 'INR Min:'
        inr_min_default_value = X['inr_min'].iloc[0]
        inr_min = st.number_input(inr_min_label, value=inr_min_default_value, step=0.1)
        st.write('New INR Min:', round(inr_min,1))
        X['inr_min'] = inr_min

    # INR Max
    with st.expander("International Normalised Ratio (INR) Max"):
        st.write('A normal INR is 1.0. Each increase of 0.1 means the blood is slightly thinner (it takes longer to clot). INR is related to the prothrombin time (PT).')
        st.write('[Veteran Affairs](https://www.hepatitis.va.gov/hcv/patient/diagnosis/labtests-INR.asp#:~:text=A%20normal%20INR%20is%201.0,the%20prothrombin%20time%20(PT).)')
        
        inr_max_label = 'INR Max:'
        inr_max_default_value = X['inr_max'].iloc[0]
        inr_max = st.number_input(inr_max_label, value=inr_max_default_value, step=0.1)
        st.write('New INR Max:', round(inr_max,1))
        X['inr_max'] = inr_max
        
    # Bun Min
    with st.expander("Blood urea nitrogen (BUN) Min"):
        st.write("A common blood test, the blood urea nitrogen (BUN) test reveals important information about how well your kidneys are working. A BUN test measures the amount of urea nitrogen that's in your blood.")
        st.write("In general, around 6 to 24 mg/dL (2.1 to 8.5 mmol/L) is considered normal.")
        st.write('[Mayo Clinic](https://www.mayoclinic.org/tests-procedures/blood-urea-nitrogen/about/pac-20384821)')
        
        bun_min_label = 'BUN Min:'
        bun_min_default_value = X['bun_min'].iloc[0]
        bun_min = st.number_input(bun_min_label, value=bun_min_default_value, step=0.1)
        st.write('New BUN Min:', round(bun_min,1))
        X['bun_min'] = bun_min

    # Anion Gap Min
    with st.expander("Anion Gap"):
        st.write("An anion gap blood test checks the acid-base balance of your blood and if the electrolytes in your blood are properly balanced.")
        st.write("There’s no universal “normal” anion gap, partly because laboratories and healthcare providers can measure and compare different electrolytes in your blood.")
        st.write('[Mayo Clinic](https://my.clevelandclinic.org/health/diagnostics/22041-anion-gap-blood-test)')
        
        aniongap_min_label = 'Anion Gap Min:'
        aniongap_min_default_value = X['aniongap_min'].iloc[0]
        aniongap_min = st.number_input(aniongap_min_label, value=aniongap_min_default_value, step=0.1)
        st.write('New Anion Gap Min:', round(aniongap_min,1))
        X['aniongap_min'] = aniongap_min

# st.write("## Inputs:")
# st.bar_chart(x=list(X[num_cols].columns), y=list(X[num_cols].values), use_container_width=True)

st.write("## Predict Patient Outcome:")

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
