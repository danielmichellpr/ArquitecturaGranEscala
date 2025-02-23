import pytest
import pandas as pd
import numpy as np
import joblib
import os
from src.modules.get4_select_model import ModelTraining  # Asegúrate de que la ruta sea correcta

@pytest.fixture
def sample_data():
    """Genera datos de prueba para entrenamiento y validación."""
    X_train = pd.DataFrame({
        "feature1": np.random.rand(50),
        "feature2": np.random.rand(50),
        "feature3": np.random.rand(50)
    })
    X_val = pd.DataFrame({
        "feature1": np.random.rand(20),
        "feature2": np.random.rand(20),
        "feature3": np.random.rand(20)
    })
    y_train = pd.Series(np.random.rand(50))
    y_val = pd.Series(np.random.rand(20))
    return X_train, X_val, y_train, y_val

@pytest.fixture
def empty_data():
    """Genera datos vacíos para pruebas."""
    return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

def test_init_valid_data(sample_data):
    """Prueba de inicialización con datos válidos."""
    X_train, X_val, y_train, y_val = sample_data
    mt = ModelTraining(X_train, X_val, y_train, y_val, model_type='random_forest')
    assert isinstance(mt.X_train, pd.DataFrame)
    assert isinstance(mt.y_train, pd.Series)

def test_init_invalid_data(empty_data):
    """Prueba de inicialización con datos vacíos."""
    X_train, X_val, y_train, y_val = empty_data
    with pytest.raises(ValueError):
        ModelTraining(X_train, X_val, y_train, y_val)

def test_train_and_evaluate_random_forest(sample_data):
    """Prueba de entrenamiento y evaluación con Random Forest."""
    X_train, X_val, y_train, y_val = sample_data
    mt = ModelTraining(X_train, X_val, y_train, y_val, model_type='random_forest')
    model, train_rmse, val_rmse = mt.train_and_evaluate_model()
    assert model is not None
    assert train_rmse >= 0
    assert val_rmse >= 0

def test_model_saving_and_loading(sample_data):
    """Prueba de guardado y carga del modelo entrenado."""
    X_train, X_val, y_train, y_val = sample_data
    mt = ModelTraining(X_train, X_val, y_train, y_val, model_type='xgboost')
    mt.train_and_evaluate_model()
    model_path = os.path.join("models", "xgboost_model.pkl")
    assert os.path.exists(model_path)
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None

def test_invalid_model_type(sample_data):
    """Prueba de tipo de modelo inválido."""
    X_train, X_val, y_train, y_val = sample_data
    with pytest.raises(ValueError):
        ModelTraining(X_train, X_val, y_train, y_val, model_type='invalid_model')

if __name__ == "__main__":
    pytest.main()
