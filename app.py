import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Heart Disease Prediction")

# Load models
models = {
    "Logistic Regression": pickle.load(open("logistic_model.pkl", "rb")),
    "SVM": pickle.load(open("svm_model.pkl", "rb")),
    "KNN": pickle.load(open("knn_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("decision_tree_model.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb"))
}

st.title("❤️ Heart Disease Prediction System")

selected_model = st.selectbox("Select ML Model", models.keys())
model = models[selected_model]

st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", min_value=1, max_value=120)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure")
chol = st.sidebar.number_input("Cholesterol")
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.sidebar.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved")
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("Oldpeak (ST Depression)", format="%.1f")
slope = st.sidebar.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thal (1 = normal, 2 = fixed, 3 = reversible)", [1, 2, 3])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach, exang,
                            oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Person is likely to have Heart Disease")
    else:
        st.success("✅ Person is unlikely to have Heart Disease")
