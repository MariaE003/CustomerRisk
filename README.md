# CustomerRisk

## 1. Présentation du projet

Ce projet consiste à analyser le comportement des clients d'une entreprise de télécommunications, à identifier différents profils grâce au clustering et à prédire les clients susceptibles de résilier leur abonnement.

L'objectif est de développer une solution de Machine Learning permettant d'aider l'entreprise à mieux comprendre ses clients et à anticiper les résiliations.

## 2. Objectifs

* Analyser et préparer les données clients.
* Segmenter les clients à l'aide d'algorithmes de clustering.
* Interpréter les profils obtenus.
* Construire et comparer plusieurs modèles de classification.
* Optimiser les modèles et évaluer leurs performances.
* Étudier l'apport du clustering dans la prédiction du churn.
* Développer une application Streamlit et la conteneuriser avec Docker.

## 3. Dataset

Le projet utilise le dataset Telco Customer Churn, qui contient des informations sur les clients, leurs abonnements, les services souscrits et leur facturation.

La variable `Churn` indique si un client a résilié son abonnement. Elle constitue la cible de la classification et doit être exclue des variables utilisées pour le clustering.

Source : [Telco Customer Churn Dataset](https://simplonline-v3-prod.s3.eu-west-3.amazonaws.com/media/file/csv/wa-fn-usec-telco-customer-churn-6ab92f83c7883379591536.csv)

## 4. Étapes du projet

### 4.1. Analyse et préparation des données

Réaliser une analyse exploratoire avec Pandas, Matplotlib et Seaborn afin d'identifier les valeurs manquantes, les doublons, les distributions et les relations entre les variables. Préparer ensuite les données avec l'imputation, l'encodage et la standardisation.

### 4.2. Clustering

Utiliser K-Means et DBSCAN pour segmenter les clients selon leurs caractéristiques. Évaluer les résultats avec le Silhouette Score et interpréter les profils obtenus. Utiliser PCA pour visualiser les clusters.

### 4.3. Classification

Entraîner et comparer plusieurs modèles pour prédire le churn :

* Logistic Regression
* Decision Tree
* Random Forest
* SVM
* Gradient Boosting ou XGBoost

Évaluer les performances avec Precision, Recall, F1-score, ROC-AUC et la matrice de confusion. Gérer le déséquilibre des classes avec `class_weight` ou SMOTE, en évitant le data leakage.

### 4.4. Clustering et classification

Ajouter la variable `Cluster` aux caractéristiques des clients, puis comparer les performances des modèles avec et sans cette variable afin de déterminer si la segmentation améliore la prédiction du churn.

### 4.5. Optimisation des modèles

Utiliser StratifiedKFold avec GridSearchCV ou RandomizedSearchCV pour rechercher les meilleurs hyperparamètres. Comparer les performances avant et après optimisation, puis évaluer le modèle retenu sur les données de test.

### 4.6. Suivi des expériences avec MLflow

Enregistrer les hyperparamètres, les métriques, les modèles et les artefacts pertinents afin de comparer les expériences et de conserver les résultats du modèle final.

### 4.7. Application et déploiement

Développer une application Streamlit permettant de saisir les caractéristiques d'un client et d'afficher son cluster, sa prédiction de churn et sa probabilité estimée. Utiliser Docker pour conteneuriser l'application et faciliter son exécution.

## 5. Technologies utilisées

* Python
* Pandas et NumPy
* Matplotlib et Seaborn
* Scikit-learn
* Imbalanced-learn
* MLflow
* Streamlit
* Docker
* Git et GitHub

## 6. Installation et exécution

Cloner le dépôt :

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_PROJET>
```

Créer et activer un environnement virtuel sous Windows :

```bash
python -m venv venv
venv\Scripts\activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lorsque l'application sera disponible, la lancer avec :

```bash
streamlit run app.py
```

## 7. Structure du projet

```text
customer-churn/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── models/
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 8. Résultats attendus

Le projet vise à identifier des profils de clients, à comparer différents modèles de prédiction et à mesurer l'apport du clustering. Il doit également permettre de sélectionner un modèle adapté et de mettre à disposition une application de prédiction du churn.

## 9. Auteur

Projet réalisé dans le cadre de la formation en Intelligence Artificielle, référentiel RNCP Développeur en intelligence artificielle.
