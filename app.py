from outils.deployment import StreamlitModelDeployer

# Colonnes des variables prédictives, dans l'ordre utilisé à l'entrainement
FEATURE_COLUMNS = [
    "age", "job", "marital", "education", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign",
    "pdays", "previous", "poutcome",
]
# Colonnes catégorielles parmi les variables prédictives
CATEGORICAL_COLUMNS = [
    "job", "marital", "education", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

deployer = StreamlitModelDeployer(
    model_path="models/dt_model.joblib",
    scaler_path="metadata/scaler.joblib",
    encoders_path="metadata/encoders.joblib",
    uniques_path="metadata/uniques.joblib",
    feature_columns=FEATURE_COLUMNS,
    categorical_columns=CATEGORICAL_COLUMNS,
    target_column="y",
    title="Prédiction de souscription à un dépôt à terme",
    description=(
        "Cette application prédit si un client va souscrire à un dépôt à terme "
        "à partir de ses informations personnelles et de l'historique de contact."
    ),
)

deployer.run()
