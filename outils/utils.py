from sklearn.preprocessing import LabelEncoder
import joblib as jb
import numpy as np


def encode_categorical_variables(df, output_path="encoders.joblib"):
    encoders = {}

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        encoder = LabelEncoder()

        # Apprentissage des classes
        encoder.fit(df[column])

        # Transformation
        df[column] = encoder.transform(df[column])

        # Sauvegarde de l'encoder associé à la colonne
        encoders[column] = encoder

    # Sauvegarde de tous les encodeurs
    jb.dump(encoders, output_path)

    return df, encoders

def save_categorical_uniques(df, cat_data, output_path="uniques.joblib"):
    uniques = {}

    for column in cat_data.columns:
        uniques[column] = list(np.unique(df[column]))

    # Sauvegarder les valeurs uniques
    jb.dump(uniques, output_path)

    return uniques