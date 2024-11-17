import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile  
import datetime
import seaborn as sns
import tempfile
from PIL import Image
import os
import io

# Function to upload model
def upload_model(model_name):
    uploaded_file = st.file_uploader(f"Upload {model_name} Model", type="joblib")
    if uploaded_file is not None:
        model = joblib.load(uploaded_file)
        st.success(f"{model_name} Model uploaded successfully!")
        return model
    else:
        st.warning(f"Please upload the {model_name} model to proceed.")
        return None

# Sidebar for navigation
st.sidebar.title("Available Prediction Models")
tabs = st.sidebar.radio("What do you want to predict?", ["Heart Disease Prediction", "Air Temperature Prediction"])

# Function to display guidance
def display_guidance_heart_disease():
    st.subheader("Heart Disease Prediction Input Guidance")
    st.write(""" 
    - **Age**: The age of the patient (0 to 120 years).
    - **Gender**: Select 'Male' or 'Female'.
    - **Cholesterol Level**: Total cholesterol level (mg/dL). Values above 240 are considered high.
    - **Blood Pressure**: Systolic blood pressure (mmHg). Normal is typically < 120.
    - **Chest Pain Type (cp)**: 
        - 0: No pain
        - 1: Atypical angina
        - 2: Typical angina
        - 3: Non-anginal pain
    - **Fasting Blood Sugar (fbs)**: 
        - 0: < 120 mg/dL
        - 1: > 120 mg/dL
    - **Resting Electrocardiographic Results (restecg)**:
        - 0: Normal
        - 1: Having ST-T wave abnormality
        - 2: Showing probable or definite left ventricular hypertrophy
    - **Max Heart Rate Achieved (thalach)**: Maximum heart rate achieved (bpm).
    - **Exercise Induced Angina (exang)**: 
        - 0: No
        - 1: Yes
    - **ST Depression (oldpeak)**: ST depression induced by exercise relative to rest (mm).
    - **Slope**: 
        - 0: Downsloping
        - 1: Flat
        - 2: Upsloping
    - **Number of Major Vessels (ca)**: Number of major vessels colored by fluoroscopy (0-3).
    - **Thalassemia (thal)**: 
        - 1: Normal
        - 2: Fixed defect
        - 3: Reversable defect
    """)

def display_guidance_air_temp():
    st.subheader("Air Temperature Prediction Input Guidance")
    st.write(""" 
    - **Present Tmax**: Maximum temperature observed today (°C).
    - **Present Tmin**: Minimum temperature observed today (°C).
    - **LDAPS_RHmin**: Minimum relative humidity (0 to 100%).
    - **LDAPS_RHmax**: Maximum relative humidity (0 to 100%).
    - **LDAPS_Tmax_lapse**: Temperature lapse rate for maximum temperature (°C).
    - **LDAPS_Tmin_lapse**: Temperature lapse rate for minimum temperature (°C).
    - **LDAPS_WS**: Wind speed (m/s).
    - **LDAPS_LH**: Latent heat (J/kg).
    - **LDAPS_CC1 to CC4**: Cloud cover values (0 to 100%).
    - **LDAPS_PPT1 to PPT4**: Precipitation values (mm).
    - **Latitude**: Latitude of the observation point (decimal degrees).
    - **Longitude**: Longitude of the observation point (decimal degrees).
    - **Digital Elevation Model (DEM)**: Elevation in meters.
    - **Slope**: Slope of the terrain (degrees).
    - **Solar Radiation**: Solar radiation received (W/m²).
    """)

# Function to create PDF
def create_pdf(location, name, input_data, prediction_result, prediction_type="heart_disease", prediction_label=None, visualization=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    today_date = datetime.date.today()
    tomorrow_date = today_date + datetime.timedelta(days=1)

    # Add user name
    if prediction_type == "heart_disease":
        pdf.cell(100, 10, f"Name: {name}", ln=False)
        pdf.cell(100, 10, f"Report Date: {today_date}", ln=True)
    # Add location
    elif prediction_type == "air_temp":
        pdf.cell(100, 10, f"Location: {location}", ln=False)
        pdf.cell(100, 10, f"Report Date: {today_date}", ln=True)

    # Add a header for the input data table
    pdf.cell(200, 10, "Input Data:", ln=True)

    # Create table headers for two columns
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(95, 10, "Feature", border=1)
    pdf.cell(95, 10, "Value", border=1, ln=True)

    pdf.set_font("Arial", size=12)

    # Add the input data as table rows
    for key, value in input_data.items():
        pdf.cell(95, 10, key, border=1)
        pdf.cell(95, 10, str(value), border=1, ln=True)

    # Add prediction result with units
    pdf.cell(200, 10, ln=True)  # Add a blank line
    pdf.set_font("Arial", style='B', size=12)

    if prediction_type == "heart_disease":
        pdf.cell(200, 10, f"Prediction Result: {prediction_label}", ln=True)  # Has Disease or Not
        pdf.cell(200, 10, f"Probability: {prediction_result:.2f}%", ln=True)  # Percent for heart disease probability
    elif prediction_type == "air_temp":
        pdf.cell(200, 10, f"Predicted Air Temperature for  {tomorrow_date}: {prediction_result:.2f}°C", ln=True)  # Degrees Celsius for air temp

    if visualization is not None:
        # Save the BytesIO object to a temporary file
        pdf.cell(200, 10, ln=True) 
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(visualization.getvalue())
            temp_file_path = temp_file.name
            
        pdf.image(temp_file_path, x=10, y=pdf.get_y(), w=180)
        
    # Save the PDF to a file or return it
    pdf_output = "output.pdf"
    pdf.output(pdf_output)

    return pdf_output

# Heart Disease Prediction Tab
if tabs == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")

    # Upload model
    heart_disease_model = upload_model("Heart Disease")

    if heart_disease_model is not None:
        # Guidance Dropdown
        if st.radio("Show Input Guidance?", ["No", "Yes"]) == "Yes":
            display_guidance_heart_disease()

        with st.form(key='heart_disease_form'):
            name = st.text_input("Enter your name:")
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input('Age', min_value=0, max_value=120, step=1)
                cholesterol = st.number_input('Cholesterol Level', min_value=0)

            with col2:
                gender = st.selectbox('Gender', ('Male', 'Female'))
                blood_pressure = st.number_input('Blood Pressure', min_value=0)

            cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
            fbs = st.selectbox('Fasting Blood Sugar (fbs)', [0, 1])
            restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', [0, 1, 2])
            thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=200)
            exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
            oldpeak = st.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=10.0, step=0.1)
            slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', [0, 1, 2])
            ca = st.selectbox('Number of Major Vessels (ca)', [0, 1, 2, 3, 4])
            thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

            # Prediction Button
            submit_button = st.form_submit_button('Predict Heart Disease')

        if submit_button:
            gender_binary = 1 if gender == 'Male' else 0
            features = np.array([[age, gender_binary, cp, blood_pressure, cholesterol, fbs, restecg,
                                  thalach, exang, oldpeak, slope, ca, thal]])
            prediction = heart_disease_model.predict(features)
            probability = heart_disease_model.predict_proba(features)[0][1] * 100  # Convert to percentage

            prediction_label = "Has Heart Disease" if prediction == 1 else "No Heart Disease"
            st.write(f"Heart Disease Prediction: **{prediction_label}**")
            st.write(f"Probability: **{probability:.2f}%**")

            # Prepare data for PDF
            input_data = {
                "Age": age,
                "Gender": gender,
                "Chest Pain Type (cp)": cp,
                "Blood Pressure": blood_pressure,
                "Cholesterol Level": cholesterol,
                "Fasting Blood Sugar (fbs)": fbs,
                "Resting Electrocardiographic Results (restecg)": restecg,
                "Maximum Heart Rate Achieved (thalach)": thalach,
                "Exercise Induced Angina (exang)": exang,
                "ST Depression (oldpeak)": oldpeak,
                "Slope": slope,
                "Number of Major Vessels (ca)": ca,
                "Thalassemia (thal)": thal
            }

            # Create PDF
            pdf_output_path = create_pdf(name, name, input_data, probability, prediction_type="heart_disease", prediction_label=prediction_label)
            with open(pdf_output_path, "rb") as pdf_file:
                st.download_button(label="Download Prediction Results as PDF", data=pdf_file, file_name=f"{name}_heart_disease_prediction.pdf")

# Air Temperature Prediction Tab
elif tabs == "Air Temperature Prediction":
    st.title("Air Temperature Prediction")

    # Upload model
    air_temp_model = upload_model("Air Temperature")

    if air_temp_model is not None:
        today_date = datetime.date.today()
        st.write(f"Today's Date: {today_date}")

        tomorrow_date = today_date + datetime.timedelta(days=1)

        # Guidance Dropdown
        if st.radio("Show Input Guidance?", ["No", "Yes"]) == "Yes":
            display_guidance_air_temp()

        with st.form(key='air_temp_form'):
            location = st.text_input("Location:")
            col1, col2 = st.columns(2)

            with col1:
                present_tmax = st.number_input('Present Tmax (°C)', min_value=0.0, step=0.1)
                present_tmin = st.number_input('Present Tmin (°C)', min_value=0.0, step=0.1)
                LDAPS_RHmin = st.number_input('LDAPS Relative Humidity Min (%)', min_value=0.0, max_value=100.0, step=0.1)
                LDAPS_RHmax = st.number_input('LDAPS Relative Humidity Max (%)', min_value=0.0, max_value=100.0, step=0.1)
                LDAPS_Tmax_lapse = st.number_input('LDAPS Tmax Lapse Rate (°C)', max_value=100.0, step=0.1)
                LDAPS_Tmin_lapse = st.number_input('LDAPS Tmin Lapse Rate (°C)', max_value=100.0, step=0.1)
                LDAPS_WS = st.number_input('LDAPS Wind Speed (m/s)', min_value=0.0, step=0.1)
                LDAPS_LH = st.number_input('LDAPS Latent Heat (J/kg)', min_value=0.0, step=0.1)
                LDAPS_CC1 = st.number_input('LDAPS Cloud Cover 1 (%)', min_value=0.0, max_value=100.0, step=0.1)
                LDAPS_CC2 = st.number_input('LDAPS Cloud Cover 2 (%)', min_value=0.0, max_value=100.0, step=0.1)

            with col2:
                LDAPS_CC3 = st.number_input('LDAPS Cloud Cover 3 (%)', min_value=0.0, max_value=100.0, step=0.1)
                LDAPS_CC4 = st.number_input('LDAPS Cloud Cover 4 (%)', min_value=0.0, max_value=100.0, step=0.1)
                LDAPS_PPT1 = st.number_input('LDAPS Precipitation 1 (mm)', min_value=0.0, step=0.1)
                LDAPS_PPT2 = st.number_input('LDAPS Precipitation 2 (mm)', min_value=0.0, step=0.1)
                LDAPS_PPT3 = st.number_input('LDAPS Precipitation 3 (mm)', min_value=0.0, step=0.1)
                LDAPS_PPT4 = st.number_input('LDAPS Precipitation 4 (mm)', min_value=0.0, step=0.1)
                lat = st.number_input('Latitude', format="%.6f")
                lon = st.number_input('Longitude', format="%.6f")
                DEM = st.number_input('Digital Elevation Model (m)', step=1)
                slope = st.number_input('Slope (°)', step=0.1)
                solar_radiation = st.number_input('Solar Radiation (W/m²)', step=1)

            # Prediction Button
            submit_button = st.form_submit_button('Predict Air Temperature')

        if submit_button:
            features = np.array([[present_tmax, present_tmin, LDAPS_RHmin, LDAPS_RHmax,
                                  LDAPS_Tmax_lapse, LDAPS_Tmin_lapse, LDAPS_WS,
                                  LDAPS_LH, LDAPS_CC1, LDAPS_CC2, LDAPS_CC3,
                                  LDAPS_CC4, LDAPS_PPT1, LDAPS_PPT2, LDAPS_PPT3,
                                  LDAPS_PPT4, lat, lon, DEM, slope, solar_radiation]])
            prediction = air_temp_model.predict(features)

            st.write(f"Predicted Air Temperature for {tomorrow_date}: **{prediction[0]:.2f} °C**")

            # Prepare data for PDF
            input_data = {
                "Present Tmax": present_tmax,
                "Present Tmin": present_tmin,
                "LDAPS Relative Humidity Min": LDAPS_RHmin,
                "LDAPS Relative Humidity Max": LDAPS_RHmax,
                "LDAPS Tmax Lapse Rate": LDAPS_Tmax_lapse,
                "LDAPS Tmin Lapse Rate": LDAPS_Tmin_lapse,
                "LDAPS Wind Speed": LDAPS_WS,
                "LDAPS Latent Heat": LDAPS_LH,
                "LDAPS Cloud Cover 1": LDAPS_CC1,
                "LDAPS Cloud Cover 2": LDAPS_CC2,
                "LDAPS Cloud Cover 3": LDAPS_CC3,
                "LDAPS Cloud Cover 4": LDAPS_CC4,
                "LDAPS Precipitation 1": LDAPS_PPT1,
                "LDAPS Precipitation 2": LDAPS_PPT2,
                "LDAPS Precipitation 3": LDAPS_PPT3,
                "LDAPS Precipitation 4": LDAPS_PPT4,
                "Latitude": lat,
                "Longitude": lon,
                "Digital Elevation Model": DEM,
                "Slope": slope,
                "Solar Radiation": solar_radiation
            }

            # Visualization
            if prediction[0] < 10:
                color = 'blue'  # Cold
                label = 'Cold (<10°C)'
            elif 10 <= prediction[0] <= 25:
                color = 'orange'  # Moderate
                label = 'Warm (10-25°C)'
            else:
                color = 'red'  # Hot
                label = 'Hot (>25°C)'

            plt.figure(figsize=(6, 4))
            plt.bar(['Predicted Temp'], [prediction[0]], color=color)
            plt.ylabel("Temperature (°C)")
            plt.title(f"Predicted Air Temperature for {tomorrow_date}")
            plt.legend([label], loc='upper right')
            st.pyplot(plt)

            # Save visualization to a temporary file
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Create PDF
            pdf_output_path = create_pdf(location, location, input_data, prediction[0], prediction_type="air_temp", visualization=buf)

            # Read the PDF file in binary mode
            with open(pdf_output_path, "rb") as pdf_file:
                st.download_button(label="Download Prediction Results as PDF", data=pdf_file, file_name=f"{location}_air_temp_prediction.pdf")