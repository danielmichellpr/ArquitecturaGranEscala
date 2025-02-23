import pytest
import pandas as pd
from src.modules.get1_cleaning import SalesCleaner  # Asegúrate de que la ruta sea correcta

@pytest.fixture
def sample_data():
    """Genera un DataFrame de prueba válido."""
    return pd.DataFrame({
        "date": ["01.01.2021", "02.01.2021", "03.01.2021"],
        "item_cnt_day": [5, -2, 3]
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
    return pd.DataFrame({"item_cnt_day": [5, -2, 3]})

@pytest.fixture
def null_date_column():
    """Genera un DataFrame con valores nulos en 'date'."""
    return pd.DataFrame({"date": [None, None, None], "item_cnt_day": [5, -2, 3]})

def test_init_valid_dataframe(sample_data):
    """Prueba de inicialización con un DataFrame válido."""
    cleaner = SalesCleaner(sample_data)
    assert isinstance(cleaner.df, pd.DataFrame)

def test_init_invalid_dataframe(invalid_data):
    """Prueba de inicialización con un DataFrame inválido."""
    with pytest.raises(ValueError):
        SalesCleaner(invalid_data)

def test_init_empty_dataframe(empty_data):
    """Prueba de inicialización con un DataFrame vacío."""
    with pytest.raises(ValueError):
        SalesCleaner(empty_data)

def test_convert_date_valid(sample_data):
    """Prueba de conversión de fecha con un DataFrame válido."""
    cleaner = SalesCleaner(sample_data)
    cleaner.convert_date()
    assert pd.api.types.is_datetime64_any_dtype(cleaner.df["date"])

def test_convert_date_missing_column(missing_date_column):
    """Prueba de conversión de fecha sin la columna 'date'."""
    cleaner = SalesCleaner(missing_date_column)
    with pytest.raises(KeyError):
        cleaner.convert_date()

def test_convert_date_null_values(null_date_column):
    """Prueba de conversión de fecha con valores nulos."""
    cleaner = SalesCleaner(null_date_column)
    with pytest.raises(ValueError):
        cleaner.convert_date()

def test_add_month_year_valid(sample_data):
    """Prueba de generación de la columna 'month_year'."""
    cleaner = SalesCleaner(sample_data)
    cleaner.convert_date()
    cleaner.add_month_year()
    assert "month_year" in cleaner.df.columns

def test_add_month_year_missing_date(missing_date_column):
    """Prueba de 'add_month_year' sin la columna 'date'."""
    cleaner = SalesCleaner(missing_date_column)
    with pytest.raises(KeyError):
        cleaner.add_month_year()

def test_execute_transformations(sample_data):
    """Prueba de ejecución completa de las transformaciones."""
    cleaner = SalesCleaner(sample_data)
    result_df = cleaner.execute_transformations()
    assert "month_year" in result_df.columns
    assert pd.api.types.is_datetime64_any_dtype(result_df["date"])
