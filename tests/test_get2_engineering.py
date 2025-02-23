import pytest
import pandas as pd
import numpy as np
from src.modules.get2_engineering import FeatureEngineering  # Asegúrate de que la ruta sea correcta

@pytest.fixture
def sample_data():
    """Genera un DataFrame de prueba válido."""
    return pd.DataFrame({
        "date": ["01.01.2021", "02.01.2021", "03.01.2021"],
        "shop_id": [1, 2, 3],
        "item_id": [101, 102, 103],
        "item_category_id": [1, 1, 2],
        "item_category_name_es": ["Cat A", "Cat A", "Cat B"],
        "item_price": [10.0, 15.5, 8.0],
        "item_cnt_day": [2, 5, 1]
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
def missing_date_column():
    """Genera un DataFrame sin la columna 'date'."""
    return pd.DataFrame({
        "shop_id": [1, 2, 3],
        "item_cnt_day": [5, -2, 3]
    })

@pytest.fixture
def null_date_column():
    """Genera un DataFrame con valores nulos en 'date'."""
    return pd.DataFrame({
        "date": [None, None, None],
        "shop_id": [1, 2, 3],
        "item_cnt_day": [5, -2, 3]
    })

def test_init_valid_dataframe(sample_data):
    """Prueba de inicialización con un DataFrame válido."""
    fe = FeatureEngineering(sample_data)
    assert isinstance(fe.df, pd.DataFrame)

def test_init_invalid_dataframe(invalid_data):
    """Prueba de inicialización con un DataFrame inválido."""
    with pytest.raises(ValueError):
        FeatureEngineering(invalid_data)

def test_init_empty_dataframe(empty_data):
    """Prueba de inicialización con un DataFrame vacío."""
    with pytest.raises(ValueError):
        FeatureEngineering(empty_data)

def test_add_time_features(sample_data):
    """Prueba de agregar características temporales."""
    fe = FeatureEngineering(sample_data)
    fe.add_time_features()
    assert all(col in fe.df.columns for col in ["date_block_num", "month", "year", "quarter", "day_of_week", "is_weekend"])

def test_add_time_features_missing_date(missing_date_column):
    """Prueba de agregar características temporales sin la columna 'date'."""
    fe = FeatureEngineering(missing_date_column)
    with pytest.raises(KeyError):
        fe.add_time_features()

def test_aggregate_sales(sample_data):
    """Prueba de agregación de ventas."""
    fe = FeatureEngineering(sample_data)
    df_sales = fe.aggregate_sales()
    assert "item_cnt_month" in df_sales.columns

def test_aggregate_prices(sample_data):
    """Prueba de agregación de precios."""
    fe = FeatureEngineering(sample_data)
    df_prices = fe.aggregate_prices()
    assert "item_price" in df_prices.columns

def test_aggregate_categories(sample_data):
    """Prueba de agregación de categorías incluyendo la moda."""
    fe = FeatureEngineering(sample_data)
    df_categories = fe.aggregate_categories()
    assert all(col in df_categories.columns for col in ["unique_categories", "category_sales_count", "most_frequent_category"])

def test_merge_aggregated_features(sample_data):
    """Prueba de fusión de características agregadas."""
    fe = FeatureEngineering(sample_data)
    df_merged = fe.merge_aggregated_features()
    assert all(col in df_merged.columns for col in ["item_cnt_month", "item_price", "unique_categories", "most_frequent_category"])

def test_apply_feature_engineering(sample_data):
    """Prueba del pipeline completo de ingeniería de características."""
    fe = FeatureEngineering(sample_data)
    df_transformed = fe.apply_feature_engineering()
    assert all(col in df_transformed.columns for col in ["date_block_num", "month", "year", "item_cnt_month", "item_price", "most_frequent_category"])
