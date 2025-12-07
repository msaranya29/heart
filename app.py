import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# ================= Load Models =================
models = {
    "Logistic Regression": pickle.load(open("logistic_model.pkl", "rb")),
    "SVM": pickle.load(open("svm_model.pkl", "rb")),
    "KNN": pickle.load(open("knn_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("decision_tree_model.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb"))
}

# ================= Load Test Data =================
X_test, Y_test = pickle.load(open("test_data.pkl", "rb"))

# ================= UI =================
st.title("❤️ Heart Disease Prediction System")

selected_model = st.selectbox("Select ML Model", models.keys())
model = models[selected_model]

# ================= Model Evaluation =================
st.subheader(f"📊 Model Performance: {selected_model}")

y_pred = model.predict(X_test)

# ---- Confusion Matrix ----
cm = confusion_matrix(Y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# ---- ROC Curve ----
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("ROC curve not available for this model.")

# ---- Feature Importance (Tree Models Only) ----
if selected_model in ["Decision Tree", "Random Forest"]:
    st.subheader("📌 Feature Importance")

    feature_names = X_test.columns
    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importance, y=feature_names, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# ================= Sidebar Input =================
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", 1, 120)
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

# ================= Prediction =================
if st.sidebar.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach, exang,
                            oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Person is likely to have Heart Disease")
    else:
        st.success("✅ Person is unlikely to have Heart Disease")
