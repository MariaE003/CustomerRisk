from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.title("Prédiction du churn client")

# dossier models/ à côté de app.py, sinon dans le dossier parent
ICI = Path(__file__).resolve().parent
MODELES = ICI / "models" if (ICI / "models").exists() else ICI.parent / "models"

preprocessor = joblib.load(MODELES / "cluster_preprocessor.pkl")
kmeans = joblib.load(MODELES / "kmeans_model.pkl")
modele = joblib.load(MODELES / "best_logistic_regression_cluster.pkl")

NOMS_CLUSTERS = {
    0: "Faible facture (peu de services)",
    1: "Nouveaux clients",
    2: "Clients fidèles à forte valeur",
}

tenure = st.number_input("Anciennete (mois)", 0, 100, 12)
monthly = st.number_input("Charges mensuelles", 0.0, 200.0, 70.0)
contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Service internet", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Paiement", ["Electronic check", "Mailed check","Bank transfer (automatic)", "Credit card (automatic)"])
support = st.selectbox("Support technique", ["Yes", "No"])
senior = st.selectbox("Senior", [0, 1])

if st.button("Prédire"):
    client = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": senior,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
    }])

    cluster = int(kmeans.predict(preprocessor.transform(client))[0])
    client["Cluster"] = str(cluster)

    prediction = modele.predict(client)[0]
    proba = modele.predict_proba(client)[0][1]

    st.write(f"**Profil / Cluster :** {cluster} - {NOMS_CLUSTERS[cluster]}")
    if prediction == 1:
        st.error("Churn prédit")
    else:
        st.success("Pas de churn prédit")
    st.metric("Probabilité de churn", f"{proba:.1%}")