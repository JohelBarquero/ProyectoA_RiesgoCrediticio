import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

        # 1. Codificación de variables categóricas (Texto -> Numérico)
        categorical_cols = df_cleaned.select_dtypes(include=['str', 'category']).columns
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


# Bloque de ejecución principal automatizado
if __name__ == "__main__":
    print("=== Iniciando el Preprocesamiento de Datos ===")

    # Ruta base siempre desde la raíz del proyecto sin importar desde dónde se corra
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # Crear carpetas si no existen
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Instanciar preprocesador
    preprocessor = DataPreprocessor()

    try:
        # 1. Cargar datos desde data/raw/
        df_raw = preprocessor.load_raw_data(RAW_DIR)
        print(f"-> Datos brutos cargados correctamente. Dimensiones: {df_raw.shape}")

        # 2. Separar en Train y Test antes de transformar (Evita filtrado de datos)
        X = df_raw.drop(columns=['target'])
        y = df_raw['target']

        # Se usa stratify=y debido al desbalance detectado en el EDA
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Reensamblar conjuntos para aplicar el Pipeline convenientemente
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        # 3. Procesar datos
        train_processed = preprocessor.fit_transform(train_df)
        test_processed = preprocessor.transform(test_df)

        # 4. Exportar datasets limpios listos para entrenar las Redes Neuronales
        train_processed.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
        test_processed.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
        print("-> Archivos 'train.csv' y 'test.csv' exportados exitosamente a 'data/processed/'.")

        # 5. Salvar los transformadores para la API
        preprocessor.save_artifacts(MODELS_DIR)
        print("-> Artefactos 'scaler.pkl' y 'encoders.pkl' guardados en 'models/'.")
        print("\n=== ¡Proceso finalizado con éxito! ===")

    except Exception as e:
        print(f"\n[ERROR] Ocurrió un fallo en el procesamiento: {e}")