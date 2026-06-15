import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        # Definir nombres de columnas originales del German Credit Dataset
        self.column_names = [
            'status_checking', 'duration_months', 'credit_history', 'purpose',
            'credit_amount', 'savings_account', 'employment_since', 'investment_rate',
            'status_sex', 'other_debtors', 'residence_since', 'property',
            'age', 'other_installments', 'housing', 'existing_credits',
            'job', 'people_liable', 'telephone', 'foreign_worker', 'target'
        ]

    def load_raw_data(self, raw_dir_path):
        """Carga el archivo original data de la UCI separado por espacios."""
        data_file = os.path.join(raw_dir_path, 'german.data')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"No se encontró el archivo original en: {data_file}")

        # Leer el archivo plano separado por espacios en blanco
        df = pd.read_csv(data_file, sep=r'\s+', names=self.column_names, header=None)

        # Ajustar el target: el dataset original usa 1 (Buen crédito) y 2 (Mal crédito)
        # Lo transformamos a la convención estándar: 0 (Bueno) y 1 (Malo)
        df['target'] = df['target'].map({1: 0, 2: 1})
        return df

    def fit_transform(self, df):
        """Aprende y aplica transformaciones sobre el set de entrenamiento."""
        df_cleaned = df.copy()

        # 1. CORRECCIÓN CLAVE: Usar 'object' en lugar de 'str' para capturar texto en Pandas
        categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
            self.encoders[col] = le  # Guardar transformador para producción

        # 2. Escalado de variables numéricas (Excluyendo el target)
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
        if len(numeric_cols) > 0:
            df_cleaned[numeric_cols] = self.scaler.fit_transform(df_cleaned[numeric_cols])

        return df_cleaned

    def transform(self, df):
        """Aplica transformaciones guardadas sobre sets de prueba o producción."""
        df_cleaned = df.copy()

        # Aplicar Encoders
        for col, le in self.encoders.items():
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else 0
                )

        # Aplicar Escalador
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
        if len(numeric_cols) > 0:
            df_cleaned[numeric_cols] = self.scaler.transform(df_cleaned[numeric_cols])

        return df_cleaned

    def save_artifacts(self, models_dir_path):
        """Guarda los objetos de preprocesamiento serializados."""
        os.makedirs(models_dir_path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(models_dir_path, 'scaler.pkl'))
        joblib.dump(self.encoders, os.path.join(models_dir_path, 'encoders.pkl'))