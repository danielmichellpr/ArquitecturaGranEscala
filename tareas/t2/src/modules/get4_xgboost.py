import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import xgboost as xgb

class ModelTraining:
    def __init__(self, X_train, X_val, y_train, y_val):
        """
        Inicializa la clase con los datos de entrenamiento y validación.
        
        Parámetros:
        X_train, X_val (pd.DataFrame): Variables predictoras.
        y_train, y_val (pd.Series): Variable objetivo.
        """
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.best_model = None
        self.best_score = float("inf")
    
    def train_and_evaluate_model(self):
        """ Entrena y evalúa el modelo XGBoost basado en RMSE. """
        print("Entrenando modelo XGBoost...")
        self.model.fit(self.X_train, self.y_train)
        
        # Predicciones en entrenamiento
        train_predictions = self.model.predict(self.X_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_predictions))
        print(f"XGBoost - RMSE en entrenamiento: {train_rmse:.4f}")
        
        # Predicciones en validación
        val_predictions = self.model.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_predictions))
        print(f"XGBoost - RMSE en validación: {val_rmse:.4f}")
        
        self.best_score = val_rmse
        self.best_model = self.model
        
        print(f"Mejor modelo: XGBoost con RMSE en validación: {self.best_score:.4f}")
        return self.best_model, train_rmse, val_rmse
