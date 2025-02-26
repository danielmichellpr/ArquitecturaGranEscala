from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

class ModelOptimizer:
    def __init__(self, model, param_grid, X_train, y_train, search_type="grid", cv=3, n_iter=10):
        """
        Inicializa la clase para optimizar hiperparámetros del modelo.
        
        Parámetros:
        model: Modelo base (ejemplo: XGBoost, Random Forest, LightGBM).
        param_grid (dict): Espacio de búsqueda de hiperparámetros.
        X_train (pd.DataFrame): Conjunto de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        search_type (str): 'grid' para GridSearchCV o 'random' para RandomizedSearchCV.
        cv (int): Número de folds en validación cruzada.
        n_iter (int): Número de iteraciones para RandomizedSearchCV.
        """
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.search_type = search_type
        self.cv = cv
        self.n_iter = n_iter
        self.best_model = None
    
    def optimize(self):
        """ Optimiza el modelo utilizando GridSearchCV o RandomizedSearchCV. """
        if self.search_type == "grid":
            search = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        else:
            search = RandomizedSearchCV(self.model, self.param_grid, n_iter=self.n_iter, cv=self.cv, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42)
        
        search.fit(self.X_train, self.y_train)
        self.best_model = search.best_estimator_
        print(f"Mejores hiperparámetros: {search.best_params_}")
        return self.best_model

    def predict(self, X_test):
        """ Realiza predicciones con el mejor modelo encontrado. """
        if self.best_model:
            return self.best_model.predict(X_test)
        else:
            raise ValueError("No se ha entrenado un modelo optimizado aún.")

# Ejemplo de uso
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2]
# }
# optimizer = ModelOptimizer(xgb.XGBRegressor(), param_grid, X_train, y_train, search_type="grid")
# best_model = optimizer.optimize()
# predictions = optimizer.predict(X_test)
