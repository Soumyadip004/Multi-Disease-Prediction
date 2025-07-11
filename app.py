import streamlit as st
import pickle
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="🏥")

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(os.path.join(working_dir, 'models', 'Diabetes.pkl'), 'rb'))
Breast_cancer_model = pickle.load(open(os.path.join(working_dir, 'models', 'breast_cancer.pkl'), 'rb'))
heart_model = pickle.load(open(os.path.join(working_dir, 'models', 'heart.pkl'), 'rb'))
Liver_model = pickle.load(open(os.path.join(working_dir, 'models', 'liver.pkl'), 'rb'))
parkinson_model = pickle.load(open(os.path.join(working_dir, 'models', 'parkinson.pkl'), 'rb'))

pneumonia_model_path = os.path.join(working_dir, 'models', 'vgg19_model_01.h5')
model = tf.keras.models.load_model(os.path.join(working_dir, 'models', 'vgg19_model_01.h5'))
kidney_model_path = os.path.join(working_dir, 'models', 'kidney.h5')

if os.path.exists(pneumonia_model_path):
    pneumonia_model = tf.keras.models.load_model(pneumonia_model_path)
else:
    st.error(f"Pneumonia model not found at: {pneumonia_model_path}")

if os.path.exists(kidney_model_path):
    kidney_model = tf.keras.models.load_model(kidney_model_path)
else:
    st.error(f"Kidney model not found at: {kidney_model_path}")

# Sidebar menu
with st.sidebar:
    selected = option_menu("Disease Prediction System",
                           ["Diabetes", "Breast Cancer", "Heart Disease", "Liver Disease", "Kidney Disease", "Parkinson's", "Pneumonia"],
                           icons=["activity", "gender-female", "heart-pulse", "droplet-half", "water", "person", "lungs"],
                           default_index=0)

# Diabetes Prediction
if selected == "Diabetes":
    st.title("Diabetes Prediction")
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0)
        Glucose = st.number_input("Glucose", min_value=0)
        BloodPressure = st.number_input("Blood Pressure", min_value=0)
        SkinThickness = st.number_input("Skin Thickness", min_value=0)
        Insulin = st.number_input("Insulin", min_value=0)
    with col2:
        BMI = st.number_input("BMI")
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
        Age = st.number_input("Age", min_value=0)

    if st.button("Predict Diabetes"):
        bmi_cat = {"NewBMI_Underweight": 0, "NewBMI_Overweight": 0,
                   "NewBMI_Obesity 1": 0, "NewBMI_Obesity 2": 0, "NewBMI_Obesity 3": 0}
        if BMI < 18.5:
            bmi_cat["NewBMI_Underweight"] = 1
        elif 25 <= BMI < 30:
            bmi_cat["NewBMI_Overweight"] = 1
        elif 30 <= BMI < 35:
            bmi_cat["NewBMI_Obesity 1"] = 1
        elif 35 <= BMI < 40:
            bmi_cat["NewBMI_Obesity 2"] = 1
        elif BMI >= 40:
            bmi_cat["NewBMI_Obesity 3"] = 1

        NewInsulinScore_Normal = 1 if 16 <= Insulin <= 166 else 0

        glucose_cat = {"NewGlucose_Low": 0, "NewGlucose_Normal": 0,
                       "NewGlucose_Overweight": 0, "NewGlucose_Secret": 0}
        if Glucose <= 70:
            glucose_cat["NewGlucose_Low"] = 1
        elif 70 < Glucose <= 99:
            glucose_cat["NewGlucose_Normal"] = 1
        elif 99 < Glucose <= 126:
            glucose_cat["NewGlucose_Overweight"] = 1
        else:
            glucose_cat["NewGlucose_Secret"] = 1

        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                DiabetesPedigreeFunction, Age,
                                bmi_cat["NewBMI_Obesity 1"], bmi_cat["NewBMI_Obesity 2"],
                                bmi_cat["NewBMI_Obesity 3"], bmi_cat["NewBMI_Overweight"],
                                bmi_cat["NewBMI_Underweight"], NewInsulinScore_Normal,
                                glucose_cat["NewGlucose_Low"], glucose_cat["NewGlucose_Normal"],
                                glucose_cat["NewGlucose_Overweight"], glucose_cat["NewGlucose_Secret"]]])

        prediction = diabetes_model.predict(input_data)
        st.success("Diabetic" if prediction[0] == 1 else "Not Diabetic")

# Breast Cancer Prediction
elif selected == "Breast Cancer":
    st.title("Breast Cancer Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        texture_mean = st.number_input("Texture Mean")
        smoothness_mean = st.number_input("Smoothness Mean")
        compactness_mean = st.number_input("Compactness Mean")
        concave_points_mean = st.number_input("Concave Points Mean")
        symmetry_mean = st.number_input("Symmetry Mean")
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean")
    with col2:
        texture_se = st.number_input("Texture SE")
        area_se = st.number_input("Area SE")
        smoothness_se = st.number_input("Smoothness SE")
        symmetry_se = st.number_input("Symmetry SE")
        fractal_dimension_se = st.number_input("Fractal Dimension SE")
    with col3:
        texture_worst = st.number_input("Texture Worst")
        area_worst = st.number_input("Area Worst")
        smoothness_worst = st.number_input("Smoothness Worst")
        compactness_worst = st.number_input("Compactness Worst")
        concavity_worst = st.number_input("Concavity Worst")
        concave_points_worst = st.number_input("Concave Points Worst")
        symmetry_worst = st.number_input("Symmetry Worst")
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst")

    if st.button("Predict Breast Cancer"):
        features = np.array([[texture_mean, smoothness_mean, compactness_mean, concave_points_mean,
                              symmetry_mean, fractal_dimension_mean, texture_se, area_se, smoothness_se,
                              symmetry_se, fractal_dimension_se, texture_worst, area_worst, smoothness_worst,
                              compactness_worst, concavity_worst, concave_points_worst, symmetry_worst,
                              fractal_dimension_worst]])
        prediction = Breast_cancer_model.predict(features)
        st.success("Malignant" if prediction[0] == 1 else "Benign")

# Heart Disease Prediction
elif selected == "Heart Disease":
    st.title("Heart Disease Prediction")
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved")
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST depression")
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.number_input("CA", min_value=0)
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    if st.button("Predict Heart Disease"):
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                          oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(data)
        st.success("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease")

# Liver Disease Prediction
elif selected == "Liver Disease":
    st.title("Liver Disease Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        TB = st.number_input("Total Bilirubin (TB)")
        DB = st.number_input("Direct Bilirubin (DB)")
        AAP = st.number_input("Alkaline Aminotransferase (AAP)")
    with col2:
        SgptAA = st.number_input("SGPT Alanine Aminotransferase (SgptAA)")
        SgotAA = st.number_input("SGOT Aspartate Aminotransferase (SgotAA)")
        TP = st.number_input("Total Proteins (TP)")
        ALBA = st.number_input("Albumin (ALBA)")
        AGR = st.number_input("Albumin/Globulin Ratio (A/G ratio)")

    if st.button("Predict Liver Disease"):
        gender_encoded = 1 if gender == "Male" else 0

        input_data = np.array([[age, gender_encoded, TB, DB, AAP,
                                SgptAA, SgotAA, TP, ALBA, AGR]])
        
        prediction = Liver_model.predict(input_data)
        st.success("Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease")


# Kidney Disease Prediction
elif selected == "Kidney Disease":
    st.title("Kidney Disease Prediction")

    st.markdown("Please enter the patient details:")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0)
        bp = st.number_input("Blood Pressure (bp)")
        sg = st.selectbox("Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025])
        al = st.number_input("Albumin (al)")
        su = st.number_input("Sugar (su)")
        rbc = st.selectbox("Red Blood Cells (rbc)", ["Normal", "Abnormal"])
        pc = st.selectbox("Pus Cell (pc)", ["Normal", "Abnormal"])
        pcc = st.selectbox("Pus Cell Clumps (pcc)", ["Not Present", "Present"])
        ba = st.selectbox("Bacteria (ba)", ["Not Present", "Present"])
        bgr = st.number_input("Blood Glucose Random (bgr)")
        bu = st.number_input("Blood Urea (bu)")
        sc = st.number_input("Serum Creatinine (sc)")

    with col2:
        sod = st.number_input("Sodium (sod)")
        pot = st.number_input("Potassium (pot)")
        hemo = st.number_input("Hemoglobin (hemo)")
        pcv = st.number_input("Packed Cell Volume (pcv)")
        wc = st.number_input("White Blood Cell Count (wc)")
        rc = st.number_input("Red Blood Cell Count (rc)")
        htn = st.selectbox("Hypertension (htn)", ["Yes", "No"])
        dm = st.selectbox("Diabetes Mellitus (dm)", ["Yes", "No"])
        cad = st.selectbox("Coronary Artery Disease (cad)", ["Yes", "No"])
        appet = st.selectbox("Appetite (appet)", ["Good", "Poor"])
        pe = st.selectbox("Pedal Edema (pe)", ["Yes", "No"])
        ane = st.selectbox("Anemia (ane)", ["Yes", "No"])

    if st.button("Predict Kidney Disease"):
        # Label encode categorical inputs
        cat_map = {"Normal": 1, "Abnormal": 0, "Present": 1, "Not Present": 0,
                   "Yes": 1, "No": 0, "Good": 1, "Poor": 0}

        data = np.array([[age, bp, sg, al, su,
                          cat_map[rbc], cat_map[pc], cat_map[pcc], cat_map[ba],
                          bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                          cat_map[htn], cat_map[dm], cat_map[cad],
                          cat_map[appet], cat_map[pe], cat_map[ane]]])

        prediction = kidney_model.predict(data)[0][0]
        st.success("Kidney Disease Detected" if prediction > 0.5 else "No Kidney Disease")



# Parkinson’s Prediction
elif selected == "Parkinson's":
    st.title("Parkinson's Prediction")
    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter_percent = st.number_input("Jitter(%)")
    shimmer = st.number_input("Shimmer")

    if st.button("Predict Parkinson’s"):
        prediction = parkinson_model.predict([[fo, fhi, flo, jitter_percent, shimmer]])
        st.success("Parkinson’s Detected" if prediction[0] == 1 else "No Parkinson’s")


# Pneumonia Prediction (Image-based)
elif selected == "Pneumonia":
    st.title("Pneumonia Detection from Chest X-ray")

    # Function to preprocess the image for the model
    def preprocess_image(image):
        image = image.convert("RGB")
        image = image.resize((128, 128))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    # Function to make a prediction
    def predict_pneumonia(image):
        processed_image = preprocess_image(image)
        prediction = pneumonia_model.predict(processed_image)
        return prediction

    # Upload an image
    uploaded_image = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

        # Add a "Predict" button
        if st.button('Predict'):
            prediction = predict_pneumonia(image)
            if prediction[0][0] > 0.5:
                st.success("Prediction: Pneumonia detected.")
            else:
                st.success("Prediction: No pneumonia detected.")
