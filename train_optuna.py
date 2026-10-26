import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "nb_visiteurs": np.random.randint(500, 50000, n),
    "taux_conversion": np.random.uniform(0.005, 0.05, n),
    "panier_moyen": np.random.uniform(20, 200, n),
    "budget_pub": np.random.uniform(0, 10000, n),
    "anciennete_mois": np.random.randint(1, 120, n),
    "secteur": np.random.choice(["mode", "tech", "maison", "beaute"], n),
})
df["ca"] = (df["nb_visiteurs"] * df["taux_conversion"] * df["panier_moyen"]
            + df["budget_pub"] * 0.5 + np.random.normal(0, 2000, n))

def make_features(d):
    d = d.copy()
    d["nb_commandes"] = d["nb_visiteurs"] * d["taux_conversion"]
    d["pub_par_visiteur"] = d["budget_pub"] / d["nb_visiteurs"]
    return pd.get_dummies(d, columns=["secteur"], dtype=int)

X = make_features(df.drop(columns=["ca"]))
y = df["ca"]

import mlflow
import mlflow.xgboost
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # où MLflow range tout
mlflow.set_experiment("ca_predictor")

params = {"n_estimators": 462, "max_depth": 5, "learning_rate": 0.0316}

with mlflow.start_run():
    model = XGBRegressor(**params, random_state=42)
    model.fit(X, y)

    mlflow.log_params(params)
    info = mlflow.xgboost.log_model(
        model,
        name="model",
        registered_model_name="ca_predictor",  # l'enregistre dans le Model Registry
    )

# On met l'étiquette "champion" sur cette version
MlflowClient().set_registered_model_alias(
    "ca_predictor", "champion", info.registered_model_version
)
print(f"Version {info.registered_model_version} enregistrée comme champion")