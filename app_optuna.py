import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("Prédiction du CA e-commerce")

import mlflow
import mlflow.xgboost

mlflow.set_tracking_uri("sqlite:///mlflow.db")

@st.cache_resource  # charge le modèle une seule fois, pas à chaque mouvement de curseur
def load_model():
    return mlflow.xgboost.load_model("models:/ca_predictor@champion")

model = load_model()
columns = model.get_booster().feature_names

# Les entrées utilisateur
nb_visiteurs = st.slider("Visiteurs / mois", 500, 50000, 20000)
taux_conversion = st.slider("Taux de conversion", 0.005, 0.05, 0.025, format="%.3f")
panier_moyen = st.slider("Panier moyen (€)", 20, 200, 100)
budget_pub = st.slider("Budget pub (€)", 0, 10000, 5000)
anciennete = st.slider("Ancienneté (mois)", 1, 119, 24)
secteur = st.selectbox("Secteur", ["mode", "tech", "maison", "beaute"])

# Même feature engineering qu'à l'entraînement
x = pd.DataFrame([{
    "nb_visiteurs": nb_visiteurs, "taux_conversion": taux_conversion,
    "panier_moyen": panier_moyen, "budget_pub": budget_pub,
    "anciennete_mois": anciennete, "secteur": secteur,
}])
x["nb_commandes"] = x["nb_visiteurs"] * x["taux_conversion"]
x["pub_par_visiteur"] = x["budget_pub"] / x["nb_visiteurs"]
x = pd.get_dummies(x, columns=["secteur"], dtype=int)
x = x.reindex(columns=columns, fill_value=0)  # colonnes manquantes → 0

pred = model.predict(x)[0]
st.metric("CA mensuel prédit", f"{pred:,.0f} €".replace(",", " "))

# Explication SHAP de cette prédiction
st.subheader("Pourquoi ce chiffre ?")
sv = shap.TreeExplainer(model)(x)
fig = plt.figure()
shap.plots.waterfall(sv[0], show=False)
st.pyplot(fig)