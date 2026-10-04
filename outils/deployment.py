import joblib as jb
import numpy as np
import pandas as pd
import streamlit as st


class StreamlitModelDeployer:
    """Déploiement générique d'un modèle de classification scikit-learn avec Streamlit.

    Cette classe est indépendante du projet : elle se reconfigure simplement en
    changeant les chemins des artefacts (modèle, scaler, encoders, uniques) et
    la liste des colonnes, ce qui permet de la réutiliser sur n'importe quel
    projet de classification suivant la même convention de sauvegarde
    (encoders et uniques sous forme de dict {nom_colonne: ...}).
    """

    def __init__(
        self,
        model_path,
        scaler_path,
        encoders_path,
        uniques_path,
        feature_columns,
        categorical_columns,
        target_column,
        title,
        description,
        page_icon="🤖",
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoders_path = encoders_path
        self.uniques_path = uniques_path
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        self.numeric_columns = [
            col for col in feature_columns if col not in categorical_columns
        ]
        self.target_column = target_column
        self.title = title
        self.description = description
        self.page_icon = page_icon
        self._load_artifacts()

    def _load_artifacts(self):
        self.model = jb.load(self.model_path)
        self.scaler = jb.load(self.scaler_path)
        self.encoders = jb.load(self.encoders_path)
        self.uniques = jb.load(self.uniques_path)
        self.class_names = self.uniques[self.target_column]

    def _encode_row(self, row):
        values = []
        for col in self.feature_columns:
            value = row[col]
            if col in self.categorical_columns:
                value = self.encoders[col].transform([value])[0]
            values.append(value)
        return np.array(values, dtype=float)

    def predict_one(self, row):
        x_new = self._encode_row(row).reshape(1, -1)
        x_new = self.scaler.transform(x_new)
        y_pred = self.model.predict(x_new)[0]
        return self.class_names[y_pred]

    def predict_dataframe(self, df):
        return [self.predict_one(row) for _, row in df[self.feature_columns].iterrows()]

    def _render_inputs(self):
        row = {}
        cols = st.columns(2)
        for i, col in enumerate(self.feature_columns):
            target = cols[i % 2]
            if col in self.categorical_columns:
                row[col] = target.selectbox(col, self.uniques[col])
            else:
                row[col] = target.number_input(col, step=1.0)
        return row

    def _single_prediction_tab(self):
        st.subheader("Prédiction à partir d'un seul client")
        row = self._render_inputs()
        if st.button("Prédire", key="predict_single"):
            prediction = self.predict_one(row)
            st.success(f"Résultat : **{prediction}**")

    def _batch_prediction_tab(self):
        st.subheader("Prédiction à partir d'un fichier CSV")
        uploaded_file = st.file_uploader("Importer un fichier CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df[self.target_column] = self.predict_dataframe(df)
            st.dataframe(df)
            st.download_button(
                "Télécharger les prédictions",
                df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

    def run(self):
        st.set_page_config(page_title=self.title, page_icon=self.page_icon)
        st.title(self.title)
        st.write(self.description)
        tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction multiple"])
        with tab1:
            self._single_prediction_tab()
        with tab2:
            self._batch_prediction_tab()
