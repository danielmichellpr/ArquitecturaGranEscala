"""
Módulo para el entrenamiento y evaluación de modelos de predicción de series temporales.

Este módulo proporciona la clase `ModelTraining`, que permite:
- Entrenar y evaluar modelos de series temporales.
- Seleccionar el mejor modelo basado en el RMSE.
- Guardar el modelo entrenado en la carpeta `models/`.

Soporta los siguientes tipos de modelos:
- **Exponential Smoothing** (Suavizamiento exponencial para datos de series temporales).
- **ARIMA** (Modelo autorregresivo integrado de media móvil).
- **Random Forest** (Modelo basado en árboles de decisión).
- **XGBoost** (Regresión basada en gradiente boosting).

Clases:
    - ModelTraining: Maneja el entrenamiento, evaluación y almacenamiento de modelos de predicción.

Excepciones:
    - ValueError: Se lanza si los datos de entrada son inválidos o si ocurre un error durante el entrenamiento del modelo.
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from logs.logger_config import logger # pylint: disable=C0413

class ModelTraining:
    """
    Clase para el entrenamiento y evaluación de modelos de predicción de series temporales.
    
    Esta clase permite entrenar modelos de tipo:
    - Exponential Smoothing
    - ARIMA
    - Random Forest
    - XGBoost
    
    Proporciona métodos para evaluar los modelos y seleccionar el mejor en base al RMSE.
    """

    def __init__(self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, model_type: str = 'xgboost'):
        """
        Inicializa la clase con los datos de entrenamiento y validación.
        
        Parámetros:
        X_train (pd.DataFrame): Variables predictoras de entrenamiento.
        X_val (pd.DataFrame): Variables predictoras de validación.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        y_val (pd.Series): Variable objetivo de validación.
        model_type (str): Tipo de modelo a entrenar ('exponential_smoothing', 'arima', 'random_forest', 'xgboost').
        
        Excepciones:
        ValueError: Si los DataFrames de entrada están vacíos o el tipo de modelo no es válido.
        """
        if X_train.empty or X_val.empty or y_train.empty or y_val.empty:
            logger.error("Los datos de entrenamiento o validación están vacíos.")
            raise ValueError("Los datos de entrenamiento o validación no pueden estar vacíos.")

        if model_type not in ['exponential_smoothing', 'arima', 'random_forest', 'xgboost']:
            logger.error("Tipo de modelo no soportado: %s", model_type)
            raise ValueError("Modelo no soportado. Usa 'exponential_smoothing', 'arima', 'random_forest' o 'xgboost'.")

        no_use_columns = ["date_block_num", "shop_id", "item_id", "most_frequent_category_id", "most_frequent_category", "item_cnt_month"]
        self.X_train, self.X_val = X_train, X_val
        self.X_train_completed = self.X_train.copy()
        self.X_val_completed = self.X_val.copy()
        X_columns = [column for column in self.X_train.columns if column not in no_use_columns]
        self.X_train = self.X_train[X_columns]
        self.X_val = self.X_val[X_columns]
        self.y_train, self.y_val = y_train, y_val
        self.model = None
        self.best_model = None
        self.best_score = float("inf")
        self.model_type = model_type
        logger.info("Inicialización de ModelTraining con modelo %s", self.model_type)

    def train_and_evaluate_model(self):
        """
        Entrena y evalúa el modelo seleccionado y guarda el mejor modelo en la carpeta 'models/'.
        
        Retorna:
        Tuple: Contiene el mejor modelo entrenado, RMSE de entrenamiento y RMSE de validación.
        
        Excepciones:
        ValueError: Si ocurre un error durante el entrenamiento del modelo.
        """
        logger.info("Entrenando modelo: %s", self.model_type)
        
        try:
            if self.model_type == 'exponential_smoothing':
                self.model = ExponentialSmoothing(self.y_train, seasonal='add', seasonal_periods=12).fit()
                train_predictions = self.model.fittedvalues
                val_predictions = self.model.forecast(steps=len(self.y_val))
            
            elif self.model_type == 'arima':
                self.model = ARIMA(self.y_train, order=(5,1,0)).fit()
                train_predictions = self.model.fittedvalues
                val_predictions = self.model.forecast(steps=len(self.y_val))
            
            elif self.model_type == 'random_forest':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(self.X_train, self.y_train)
                train_predictions = self.model.predict(self.X_train)
                val_predictions = self.model.predict(self.X_val)
            
            elif self.model_type == 'xgboost':
                self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                self.model.fit(self.X_train, self.y_train)
                train_predictions = self.model.predict(self.X_train)
                val_predictions = self.model.predict(self.X_val)

            else:
                raise ValueError("Modelo no soportado.")

            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_predictions))
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_predictions))

            logger.info("%s - RMSE en entrenamiento: %.4f", self.model_type, train_rmse)
            logger.info("%s - RMSE en validación: %.4f", self.model_type, val_rmse)

            self.best_score = val_rmse
            self.best_model = self.model

            # Generar predicciones para todo el dataset
            full_predictions = self.best_model.predict(pd.concat([self.X_train, self.X_val], axis=0))
            full_data = pd.concat([self.X_train_completed, self.X_val_completed], axis=0)
            full_data["predicted_item_cnt_month"] = full_predictions
            logger.info("Mejor modelo: %s con RMSE en validación: %.4f", self.model_type, self.best_score)
            return full_data, self.best_model

        except Exception as e:
            logger.error("Error al entrenar el modelo: %s", str(e))
            raise ValueError("Error al entrenar el modelo.") from e
