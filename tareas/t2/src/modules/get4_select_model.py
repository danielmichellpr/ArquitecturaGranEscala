import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class ModelTraining:
    def __init__(self, X_train, X_val, y_train, y_val, model_type='exponential_smoothing'):
        """
        Inicializa la clase con los datos de entrenamiento y validación.
        
        Parámetros:
        X_train, X_val (pd.DataFrame): Variables predictoras.
        y_train, y_val (pd.Series): Variable objetivo.
        model_type (str): Tipo de modelo a entrenar ('exponential_smoothing', 'arima', 'random_forest', 'xgboost').
        """
        self.X_train, self.X_val = X_train, X_val
        X_columns = [column for column in self.X_train.columns if column not in ["date_block_num", "avg_price_per_shop_item", "avg_sales_per_shop_item"]]
        self.X_train = self.X_train[X_columns]
        self.X_val = self.X_val[X_columns]
        self.y_train, self.y_val = y_train, y_val
        self.model = None
        self.best_model = None
        self.best_score = float("inf")
        self.model_type = model_type
    
    def train_and_evaluate_model(self):
        """ Entrena y evalúa el modelo seleccionado. """
        print(f"Entrenando modelo: {self.model_type}...")
        
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
            raise ValueError("Modelo no soportado. Usa 'exponential_smoothing', 'arima', 'random_forest' o 'xgboost'.")
        
        # Calcular RMSE
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_predictions))
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_predictions))
        
        print(f"{self.model_type} - RMSE en entrenamiento: {train_rmse:.4f}")
        print(f"{self.model_type} - RMSE en validación: {val_rmse:.4f}")
        
        self.best_score = val_rmse
        self.best_model = self.model
        
        print(f"Mejor modelo: {self.model_type} con RMSE en validación: {self.best_score:.4f}")
        return self.best_model, train_rmse, val_rmse
