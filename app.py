from sklearn.calibration import LabelEncoder
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the pre-trained model
model = joblib.load('space_debris_predictor.pkl')

# Load the data for visualizations
space_data = pd.read_csv('space_decay.csv')

# ----------------- Preprocess data -----------------
# Convert dates
space_data['LAUNCH_DATE'] = pd.to_datetime(space_data['LAUNCH_DATE'], errors='coerce')
space_data['DECAY_DATE'] = pd.to_datetime(space_data['DECAY_DATE'], errors='coerce')

# Handle missing values
space_data['COUNTRY_CODE'] = space_data['COUNTRY_CODE'].fillna('Unknown')
space_data['RCS_SIZE'] = space_data['RCS_SIZE'].fillna('SMALL')

# Create Time_In_Orbit feature (difference between launch and decay date)
space_data['Time_In_Orbit'] = (space_data['DECAY_DATE'].fillna(pd.Timestamp.today()) - space_data['LAUNCH_DATE']).dt.days

# Map RCS_SIZE to numerical values
size_mapping = {'SMALL': 1, 'MEDIUM': 2, 'LARGE': 3}
space_data['RCS_SIZE_NUM'] = space_data['RCS_SIZE'].map(size_mapping)

# Create a feature for Launch Year
space_data['Launch_Year'] = space_data['LAUNCH_DATE'].dt.year

# ----------------- Streamlit App Layout -----------------
st.title("Space Debris Predictor")
st.write("This tool predicts whether an object in orbit is likely to become space debris based on its characteristics.")

# ----------------- Input Section for Predictions -----------------
def get_user_input():
    rcs_size = st.selectbox('RCS Size:', ['SMALL', 'MEDIUM', 'LARGE'])
    rcs_size_num = {'SMALL': 1, 'MEDIUM': 2, 'LARGE': 3}[rcs_size]
    
    launch_year = st.number_input('Launch Year:', min_value=1950, max_value=2024, step=1)
    
    country_code = st.selectbox('Country:', sorted(space_data['COUNTRY_CODE'].unique()))
    
    # Encode the country code
    le = LabelEncoder()
    space_data['COUNTRY_CODE_ENC'] = le.fit_transform(space_data['COUNTRY_CODE'])
    country_code_enc = le.transform([country_code])[0]
    
    # Calculate time in orbit from launch year to today
    time_in_orbit = (pd.Timestamp.today().year - launch_year) * 365
    
    # Create input feature array
    features = np.array([[time_in_orbit, rcs_size_num, country_code_enc]])
    
    return features

user_input = get_user_input()

# ----------------- Prediction Section -----------------
if st.button("Predict"):
    try:
        prediction = model.predict(user_input)
        st.write(f"Prediction: {'Debris' if prediction[0] else 'Not Debris'}")
    except Exception as e:
        st.error(f"Error: {e}")

# ----------------- Visualization Section -----------------
st.write("### Space Debris Visualizations")

# Visual 1: Scatter Plot - Launch Year vs RCS Size
st.write("#### Scatter Plot: Launch Year vs RCS Size")
scatter_fig = px.scatter(space_data, x='Launch_Year', y='RCS_SIZE_NUM', color='COUNTRY_CODE',
                         labels={'Launch_Year': 'Launch Year', 'RCS_SIZE_NUM': 'RCS Size'},
                         title='Launch Year vs RCS Size')
st.plotly_chart(scatter_fig)

# Visual 2: Bar Chart - Debris by Country
st.write("#### Bar Chart: Debris by Country")
debris_by_country = space_data.groupby('COUNTRY_CODE')['DECAY_DATE'].apply(lambda x: x.notnull().sum()).reset_index()
bar_chart_fig = px.bar(debris_by_country, x='COUNTRY_CODE', y='DECAY_DATE',
                       labels={'DECAY_DATE': 'Debris Count'}, title='Debris Count by Country')
st.plotly_chart(bar_chart_fig)

# Visual 3: Distribution of Time in Orbit
st.write("#### Distribution of Time in Orbit (Days)")
time_in_orbit_fig = px.histogram(space_data, x='Time_In_Orbit', nbins=50, color='RCS_SIZE',
                                 title='Distribution of Time in Orbit (Days)')
st.plotly_chart(time_in_orbit_fig)

# Visual 4: Feature Importance from the Model
st.write("#### Feature Importance in Space Debris Prediction")
if 'space_debris_predictor.pkl':
    feature_importance = model.feature_importances_
    feature_names = ['Time_In_Orbit', 'RCS_SIZE_NUM', 'COUNTRY_CODE_ENC']
    importance_fig = px.bar(x=feature_names, y=feature_importance,
                            labels={'x': 'Features', 'y': 'Importance'},
                            title="Feature Importance in Space Debris Prediction")
    st.plotly_chart(importance_fig)
else:
    st.warning("Model not loaded. Please ensure the model file exists.")

# Visual 5: Launch Year Distribution (from EDA)
st.write("#### Launch Year Distribution")
launch_year_fig = px.histogram(space_data['Launch_Year'].dropna(), nbins=50, title="Launch Year Distribution")
st.plotly_chart(launch_year_fig)
