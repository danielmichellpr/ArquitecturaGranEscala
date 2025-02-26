"""
Módulo de preparación de datos para modelos de Machine Learning.

Este módulo proporciona la clase `DataPreparation`, que incluye métodos para:
- División de datos en conjuntos de entrenamiento y validación manteniendo la cronología.
- Manejo de valores nulos mediante imputación de medianas o ceros en variables de retraso.
- Eliminación de columnas innecesarias que no aportan al modelo.
- Manejo de valores infinitos reemplazándolos por valores adecuados.
- Eliminación de variables altamente correlacionadas para evitar redundancia en los datos.
- Normalización de características numéricas con `StandardScaler`.
- Aplicación de una pipeline completa de transformación de datos.

Clases:
    - DataPreparation: Contiene métodos para la preparación estructurada de datos antes de modelado.

Excepciones:
    - ValueError: Se lanza cuando los datos de entrada no son válidos, están vacíos o contienen valores inesperados.
    - KeyError: Se lanza cuando una columna requerida no está presente en el DataFrame.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from logs.logger_config import logger # pylint: disable=C0413

class DataPreparation:
    """
    Clase para la preparación de datos antes de su uso en modelos de machine learning.
    Incluye manejo de valores nulos, eliminación de columnas innecesarias, normalización y filtrado de correlaciones.
    """

    def __init__(self, df: pd.DataFrame, target: str = "item_cnt_month"):
        """
        Inicializa la clase con un DataFrame y el target deseado.
        
        Parámetros:
            - df (pd.DataFrame): DataFrame con los datos preparados por Feature Engineering.
            - target (str): Variable objetivo ('item_cnt_month' para regresión o clasificación).
        
        Excepciones:
            - ValueError: Si df no es un DataFrame o está vacío.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("El argumento df no es un DataFrame válido.")
            raise ValueError("El argumento df debe ser un pandas DataFrame.")
        if df.empty:
            logger.warning("El DataFrame está vacío.")
            raise ValueError("El DataFrame está vacío.")

        self.df = df.copy()
        self.target = target
        self.scaler = StandardScaler()
        self.transformed_columns = None
        logger.info("Inicio del proceso de preparación de datos.")

    def split_data(self, test_size: float = 0.2):
        """
        Divide los datos en conjuntos de entrenamiento y validación manteniendo la cronología de las observaciones.
        
        Parámetros:
            - test_size (float): Proporción del conjunto de validación. Por defecto, 0.2 (20%).
        
        Retorna:
            - Tuple[pd.DataFrame, pd.DataFrame]: DataFrames de entrenamiento y validación.
        
        Excepciones:
            - KeyError: Si la columna 'date_block_num' no existe en el DataFrame.
            - ValueError: Si la proporción de test_size no es válida.
        """
        if "date_block_num" not in self.df.columns:
            logger.error("La columna 'date_block_num' no existe en el DataFrame.")
            raise KeyError("La columna 'date_block_num' no existe en el DataFrame.")
        if not (0 < test_size < 1):
            logger.error("El parámetro 'test_size' debe estar entre 0 y 1.")
            raise ValueError("El parámetro 'test_size' debe estar entre 0 y 1.")

        logger.info("Dividiendo los datos en entrenamiento y validación...")
        date_blocks = self.df["date_block_num"].unique()
        split_index = int(len(date_blocks) * (1 - test_size))
        train_blocks, val_blocks = date_blocks[:split_index], date_blocks[split_index:]

        self.train_df = self.df[self.df["date_block_num"].isin(train_blocks)]
        self.val_df = self.df[self.df["date_block_num"].isin(val_blocks)]

        logger.info("División de datos completada.")
        return self.train_df, self.val_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja los valores faltantes en el DataFrame rellenando con ceros en variables de retraso y la mediana en otras.
        
        Parámetros:
            - df (pd.DataFrame): DataFrame con posibles valores faltantes.
        
        Retorna:
            - pd.DataFrame: DataFrame con valores faltantes tratados.
        
        Excepciones:
            - ValueError: Si el DataFrame está vacío.
        """
        if df.empty:
            logger.warning("El DataFrame está vacío. No se puede manejar valores faltantes.")
            raise ValueError("El DataFrame está vacío.")

        logger.info("Manejando valores faltantes...")
        no_use_columns = ["date_block_num", "shop_id", "item_id", "most_frequent_category_id", "most_frequent_category"]
        lag_features = ["sales_lag_1m", "sales_lag_3m", "sales_lag_6m"]

        for feature in lag_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)
        columns_to_use = [column for column in df.columns if column not in no_use_columns]
        df[columns_to_use] = df[columns_to_use].fillna(df[columns_to_use].median())
        logger.info("Valores faltantes manejados correctamente.")
        return df

    # def remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Elimina columnas innecesarias para el modelo, como identificadores que no aportan información relevante.
        
    #     Parámetros:
    #     df (pd.DataFrame): DataFrame con posibles columnas irrelevantes.
        
    #     Retorna:
    #     pd.DataFrame: DataFrame sin columnas innecesarias.
    #     """
    #     drop_columns = ["shop_id", "item_id"]
    #     existing_columns = [col for col in drop_columns if col in df.columns]

    #     if not existing_columns:
    #         logger.info("No hay columnas innecesarias para eliminar.")
    #     else:
    #         df.drop(columns=existing_columns, inplace=True)
    #         logger.info(f"Columnas eliminadas: {existing_columns}")

    #     return df

    def handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reemplaza valores infinitos por NaN y posteriormente los maneja con la mediana de cada columna.
        
        Parámetros:
            - df (pd.DataFrame): DataFrame con posibles valores infinitos.
        
        Retorna:
            -pd.DataFrame: DataFrame con valores infinitos corregidos.
        """
        if df.empty:
            logger.warning("El DataFrame está vacío. No se pueden manejar valores infinitos.")
            raise ValueError("El DataFrame está vacío.")

        logger.info("Reemplazando valores infinitos...")
        no_use_columns = ["date_block_num", "shop_id", "item_id", "most_frequent_category_id", "most_frequent_category"]
        columns_to_use = [column for column in df.columns if column not in no_use_columns]
        df[columns_to_use] = df[columns_to_use].replace([np.inf, -np.inf], np.nan)
        df[columns_to_use] = df[columns_to_use].fillna(df[columns_to_use].median())
        logger.info("Valores infinitos manejados correctamente.")
        return df

    def remove_highly_correlated_features(self, df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
        """
        Elimina variables altamente correlacionadas por encima de un umbral dado.
        
        Parámetros:
            - df (pd.DataFrame): DataFrame con posibles correlaciones altas.
            - threshold (float): Umbral de correlación para eliminar variables (por defecto, 0.80).
        
        Retorna:
            - pd.DataFrame: DataFrame sin variables altamente correlacionadas.
        
        Excepciones:
            - ValueError: Si el DataFrame está vacío o si el umbral no es válido.
        """
        if df.empty:
            logger.warning("El DataFrame está vacío. No se pueden eliminar variables correlacionadas.")
            raise ValueError("El DataFrame está vacío.")
        if not (0 < threshold < 1):
            logger.error("El umbral de correlación debe estar entre 0 y 1.")
            raise ValueError("El umbral de correlación debe estar entre 0 y 1.")

        logger.info("Eliminando variables altamente correlacionadas...")
        no_use_columns = ["date_block_num", "shop_id", "item_id", "most_frequent_category_id", "most_frequent_category", "item_cnt_month"]
        corr_matrix = df.drop(columns=no_use_columns, errors='ignore').corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        df.drop(columns=to_drop, inplace=True)
        logger.info(f"Variables eliminadas por alta correlación: {to_drop}")
        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Escala características numéricas usando StandardScaler.
        
        Parámetros:
            - df (pd.DataFrame): DataFrame con características a escalar.
            - fit (bool): Si True, ajusta el scaler a los datos; si False, solo transforma.
        
        Retorna:
            - pd.DataFrame: DataFrame con características escaladas.
        
        Excepciones:
            - ValueError: Si el DataFrame está vacío.
        """
        if df.empty:
            logger.warning("El DataFrame está vacío. No se pueden escalar características.")
            raise ValueError("El DataFrame está vacío.")
        logger.info("Escalando características numéricas...")
        no_use_columns = ["date_block_num", "shop_id", "item_id", "most_frequent_category_id", "most_frequent_category", "item_cnt_month"]
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(no_use_columns).tolist()
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        logger.info("Escalamiento de características completado.")
        return df

    def apply_pipeline(self):
        """
        Aplica todas las transformaciones en el conjunto de entrenamiento y validación.
        
        Retorna:
            - None: Modifica los DataFrames internos.
        """
        logger.info("Aplicando pipeline de transformación de datos...")
        self.train_df = self.handle_missing_values(self.train_df)
        # self.train_df = self.remove_unnecessary_columns(self.train_df)
        self.train_df = self.handle_infinite_values(self.train_df)
        self.train_df = self.remove_highly_correlated_features(self.train_df)
        self.train_df = self.scale_features(self.train_df, fit=True)

        self.val_df = self.handle_missing_values(self.val_df)
        # self.val_df = self.remove_unnecessary_columns(self.val_df)
        self.val_df = self.handle_infinite_values(self.val_df)
        self.val_df = self.val_df[self.train_df.columns]
        self.val_df = self.scale_features(self.val_df, fit=False)

        logger.info("Pipeline de transformación aplicado correctamente.")

    def prepare_data(self):
        """
        Ejecuta el proceso completo de preparación de datos.
        
        Retorna:
            - Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - DataFrames de entrenamiento y validación junto con sus etiquetas.
        """
        logger.info("Iniciando proceso de preparación de datos...")
        self.split_data()
        self.apply_pipeline()
        X_train, y_train = self.train_df.drop(columns=[self.target]), self.train_df[self.target]
        X_val, y_val = self.val_df.drop(columns=[self.target]), self.val_df[self.target]
        logger.info("Preparación de datos completada.")
        return X_train, X_val, y_train, y_val
