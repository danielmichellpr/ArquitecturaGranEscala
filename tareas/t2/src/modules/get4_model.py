import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# 
# from sklearn.ensemble import RandomForestRegressor
# import lightgbm as lgb

class ModelTraining:
    def __init__(self, X_train, X_val, y_train, y_val):
        """
        Inicializa la clase con los datos de entrenamiento y validación.
        
        Parámetros:
        X_train, X_val (pd.DataFrame): Variables predictoras.
        y_train, y_val (pd.Series): Variable objetivo.
        """
        self.X_train, self.X_val = X_train, X_val
        X_columns = [column for column in self.X_train.columns if column not in ["date_block_num", "avg_price_per_shop_item", "avg_sales_per_shop_item"]]
        self.X_train = self.X_train[X_columns]
        self.X_val = self.X_val[X_columns]
        self.y_train, self.y_val = y_train, y_val
        self.models = {
            "xgboost": xgb.XGBRegressor(),
            # "random_forest": RandomForestRegressor()
        }
        self.best_model = None
        self.best_score = float("inf")
        self.best_model_name = None


    def train_and_evaluated_model(self):
        """ Entrena y evalúa diferentes modelos, seleccionando el mejor basado en RMSE. """
        for model_name, model in self.models.items():
            print(f"Entrenando modelo: {model_name}")
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val, predictions))
            print(f"{model_name} - RMSE: {rmse:.4f}")
            
            if rmse < self.best_score:
                self.best_score = rmse
                self.best_model = model
                self.best_model_name = model_name
        
        print(f"Mejor modelo: {self.best_model_name} con RMSE: {self.best_score:.4f}")
        return self.best_model

    def train_and_evaluate_models(self):
        """ Entrena y evalúa diferentes modelos, seleccionando el mejor basado en RMSE. """
        for model_name, model in self.models.items():
            print(f"Entrenando modelo: {model_name}")
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val, predictions))
            print(f"{model_name} - RMSE: {rmse:.4f}")
            
            if rmse < self.best_score:
                self.best_score = rmse
                self.best_model = model
                self.best_model_name = model_name
        
        print(f"Mejor modelo: {self.best_model_name} con RMSE: {self.best_score:.4f}")
        return self.best_model
    
    def optimize_hyperparameters(self, param_grids, search_type="grid", cv=3):
        """ Optimiza hiperparámetros del mejor modelo usando GridSearchCV o RandomizedSearchCV. """
        if not self.best_model:
            raise ValueError("No hay modelo entrenado para optimizar.")
        
        param_grid = param_grids.get(self.best_model_name, {})
        
        if search_type == "grid":
            search = GridSearchCV(self.best_model, param_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        else:
            search = RandomizedSearchCV(self.best_model, param_grid, n_iter=10, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42)
        
        search.fit(self.X_train, self.y_train)
        self.best_model = search.best_estimator_
        print(f"Mejor modelo optimizado ({self.best_model_name}): {search.best_params_}")
        return self.best_model
    
    def predict(self, X_val):
        """ Realiza predicciones con el mejor modelo entrenado. """
        if self.best_model:
            return self.best_model.predict(X_val)
        else:
            raise ValueError("El modelo no ha sido entrenado.")

# Uso de la clase con datos preparados
# model_trainer = ModelTraining(X_train, X_val, y_train, y_val)
# model_trainer.train_and_evaluate_models()
# param_grids = {
#     "xgboost": {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]},
#     "lightgbm": {"num_leaves": [31, 50], "learning_rate": [0.01, 0.1]},
#     "random_forest": {"n_estimators": [100, 200], "max_depth": [10, 20]}
# }
# model_trainer.optimize_hyperparameters(param_grids)
# predictions = model_trainer.predict(X_test)
