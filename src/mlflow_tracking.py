import os
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix)


def enregistrer_experience(model,X_test,y_test,params,nom_run,preprocessing_params=None):
    mlflow.set_experiment("Customer_Churn")

    # predictions
    y_pred = model.predict(X_test)

    # obtenir un score utilisable pour le ROC-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    with mlflow.start_run(run_name=nom_run):

        # 1. enregistrer les hyperparametres
        mlflow.log_params({
            str(k): str(v) for k, v in params.items()
        })

        # 2. enregistrer les informations de preprocessing
        if preprocessing_params:
            mlflow.log_params({
                f"preprocessing_{k}": str(v)
                for k, v in preprocessing_params.items()
            })

        # 3. calculer et enregistrer les metriques
        metriques = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1_score": f1_score(
                y_test, y_pred, zero_division=0
            )
        }

        if y_score is not None:
            metriques["ROC_AUC"] = roc_auc_score(
                y_test, y_score
            )

        mlflow.log_metrics(metriques)

        # 4. enregistrer le modele
        mlflow.sklearn.log_model(
            sk_model=model,
            name="modele_churn",
            serialization_format="pickle"
        )

        # 5. enregistrer la matrice de confusion comme artifact
        matrice = confusion_matrix(y_test, y_pred)

        matrice_df = pd.DataFrame(
            matrice,
            index=["Reel_No", "Reel_Yes"],
            columns=["Predit_No", "Predit_Yes"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            chemin = os.path.join(
                temp_dir, "matrice_confusion.csv"
            )
            matrice_df.to_csv(chemin)
            mlflow.log_artifact(chemin)

    print(f"experience '{nom_run}' enregistre")