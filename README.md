# 🏦 Prédiction de souscription à un dépôt à terme

Application **Streamlit** qui prédit si un client va souscrire à un dépôt à
terme à partir de ses informations personnelles et de l'historique de
contact d'une campagne marketing bancaire. Le modèle retenu est un
**Decision Tree Classifier**, comparé à la régression logistique et au SVM
dans le notebook d'exploration.

## Structure du projet

```
.
├── app.py                                                      # Application Streamlit (configuration + lancement)
├── requirements.txt                                            # Dépendances Python
├── outils/
│   ├── deployment.py                                           # Classe générique de déploiement Streamlit (inférence)
│   └── utils.py                                                # Fonctions utilitaires
├── data/
│   └── Bank_Data.csv                                           # Jeu de données brut utilisé pour l'entraînement
├── metadata/
│   ├── encoders.joblib                                         # LabelEncoders des variables catégorielles
│   ├── scaler.joblib                                           # Scaler ajusté sur les variables
│   └── uniques.joblib                                          # Valeurs uniques des variables (menus déroulants + classes cibles)
├── models/
│   └── dt_model.joblib                                         # Modèle Decision Tree entraîné (modèle utilisé par app.py)
├── encoders/
│   └── encoders.joblib                                         # Copie des encodeurs
└── Copie de Clas_Log_Reg_SVM_Decision_Tree_uncomplet.ipynb      # Exploration, préparation des données et entraînement des modèles
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

L'application propose deux modes :

- **Prédiction simple** : renseigner les informations d'un client via un
  formulaire et obtenir sa probabilité de souscription.
- **Prédiction multiple** : importer un fichier CSV contenant les colonnes
  `age, job, marital, education, housing, loan, contact, month,
  day_of_week, duration, campaign, pdays, previous, poutcome`, et
  télécharger le fichier avec la colonne `y` (prédiction) ajoutée.

## Entraînement du modèle

Le notebook `Copie de Clas_Log_Reg_SVM_Decision_Tree_uncomplet.ipynb`
contient le pipeline de préparation des données :

1. Analyse exploratoire (distributions, corrélations).
2. Encodage des variables catégorielles (`LabelEncoder`) et sauvegarde des
   encodeurs et des valeurs uniques.
3. Normalisation des variables numériques.
4. Séparation des données en train / test.
5. Entraînement et comparaison de plusieurs modèles : régression logistique,
   SVM et arbre de décision.
6. Évaluation (accuracy, précision, rappel, F1) et sélection du meilleur
   modèle, exporté avec `joblib` dans `models/`.

## Jeu de données

`data/Bank_Data.csv` contient 11640 lignes avec les colonnes suivantes :

| Colonne         | Description                                              |
|-----------------|-----------------------------------------------------------|
| `age`           | Âge du client                                             |
| `job`           | Profession du client                                      |
| `marital`       | Statut marital                                             |
| `education`     | Niveau d'éducation                                         |
| `housing`       | Possède un crédit immobilier (oui/non)                    |
| `loan`          | Possède un crédit personnel (oui/non)                      |
| `contact`       | Type de contact (cellular, telephone)                      |
| `month`         | Mois du dernier contact                                    |
| `day_of_week`   | Jour de la semaine du dernier contact                      |
| `duration`      | Durée du dernier appel (secondes)                          |
| `campaign`      | Nombre de contacts durant la campagne en cours             |
| `pdays`         | Nombre de jours depuis le dernier contact (campagne passée)|
| `previous`      | Nombre de contacts avant la campagne en cours               |
| `poutcome`      | Résultat de la campagne marketing précédente                |
| `y`             | Souscription au dépôt à terme (variable cible)              |

## Déploiement

Cette application est conçue pour être déployée sur **Streamlit Community
Cloud** :

1. Pousser ce dépôt sur GitHub.
2. Aller sur [share.streamlit.io](https://share.streamlit.io), se connecter
   avec GitHub.
3. Cliquer sur **New app**, sélectionner le dépôt, la branche `main` et le
   fichier `app.py`.
4. Déployer : l'application sera accessible via une URL publique du type
   `https://<nom-app>.streamlit.app`.
