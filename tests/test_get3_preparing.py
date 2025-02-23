import pytest
import pandas as pd
import numpy as np
from src.modules.get3_preparing import DataPreparation  # Ajusta la ruta según la estructura

@pytest.fixture
def sample_data():
    """Genera un DataFrame de prueba válido."""
    return pd.DataFrame({
        "date_block_num": [1, 2, 3],
        "shop_id": [1, 2, 3],
        "item_id": [101, 102, 103],
        "item_category_id": [1, 1, 2],
        "item_category_name_es": ["Cat A", "Cat A", "Cat B"],
        "item_price": [10.0, 15.5, 8.0],
        "item_cnt_month": [2, 5, 1]
    })

@pytest.fixture
def empty_data():
    """Genera un DataFrame vacío."""
    return pd.DataFrame()

@pytest.fixture
def invalid_data():
    """Genera un dato inválido (no DataFrame)."""
    return "Esto no es un DataFrame"

@pytest.fixture
def sample_data_with_nulls():
    """Genera un DataFrame con valores nulos."""
    return pd.DataFrame({
        "date_block_num": [1, 2, 3],
        "shop_id": [1, 2, 3],
        "item_id": [101, 102, 103],
        "item_price": [10.0, np.nan, 8.0],
        "item_cnt_month": [2, 5, np.nan]
    })

def test_init_valid_dataframe(sample_data):
    """Prueba de inicialización con un DataFrame válido."""
    dp = DataPreparation(sample_data)
    assert isinstance(dp.df, pd.DataFrame)

def test_init_invalid_dataframe(invalid_data):
    """Prueba de inicialización con un DataFrame inválido."""
    with pytest.raises(ValueError):
        DataPreparation(invalid_data)

def test_init_empty_dataframe(empty_data):
    """Prueba de inicialización con un DataFrame vacío."""
    with pytest.raises(ValueError):
        DataPreparation(empty_data)

def test_handle_missing_values(sample_data_with_nulls):
    """Prueba de manejo de valores nulos."""
    dp = DataPreparation(sample_data_with_nulls)
    df_cleaned = dp.handle_missing_values(dp.df)
    assert df_cleaned.isnull().sum().sum() == 0  # Verifica que no haya valores nulos

def test_handle_infinite_values():
    """Prueba de manejo de valores infinitos."""
    df = pd.DataFrame({"col1": [1, 2, np.inf, -np.inf, 5]})
    dp = DataPreparation(df)
    df_cleaned = dp.handle_infinite_values(dp.df)
    assert not np.isinf(df_cleaned).values.any()  # Verifica que no haya valores infinitos

def test_remove_highly_correlated_features(sample_data):
    """Prueba de eliminación de variables altamente correlacionadas."""
    sample_data["high_corr"] = sample_data["item_price"] * 2  # Agregar una variable artificialmente correlacionada
    dp = DataPreparation(sample_data)
    df_reduced = dp.remove_highly_correlated_features(dp.df, threshold=0.80)
    assert "high_corr" not in df_reduced.columns  # Verifica que la variable haya sido eliminada

def test_scale_features(sample_data):
    """Prueba de escalamiento de características."""
    dp = DataPreparation(sample_data)
    df_scaled = dp.scale_features(dp.df, fit=True)
    assert np.allclose(df_scaled.mean(), 0, atol=1)  # Verifica que los valores tengan media cercana a 0

def test_apply_pipeline(sample_data):
    """Prueba del pipeline completo de preparación de datos."""
    dp = DataPreparation(sample_data)
    dp.split_data()
    dp.apply_pipeline()
    assert dp.train_df.shape[0] > 0  # Verifica que haya datos en el conjunto de entrenamiento
    assert dp.val_df.shape[0] > 0  # Verifica que haya datos en el conjunto de validación
    assert dp.train_df.isnull().sum().sum() == 0  # Verifica que no haya valores nulos

def test_prepare_data(sample_data):
    """Prueba de ejecución completa del proceso de preparación de datos."""
    dp = DataPreparation(sample_data)
    X_train, X_val, y_train, y_val = dp.prepare_data()
    assert X_train.shape[0] > 0 and X_val.shape[0] > 0  # Verifica que haya datos en ambos conjuntos
    assert y_train.shape[0] > 0 and y_val.shape[0] > 0  # Verifica que haya valores objetivo

if __name__ == "__main__":
    pytest.main()