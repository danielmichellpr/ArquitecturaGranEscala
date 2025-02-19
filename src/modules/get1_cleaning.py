import pandas as pd
import logging

# Importar logger en lugar de configurarlo aquí
logger = logging.getLogger(__name__)

class SalesCleaner:
    """
    Clase para procesar el DataFrame de ventas 'sales_train' con transformaciones clave.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa la clase con un DataFrame y valida su estructura.
        
        Parámetros:
        df (pd.DataFrame): DataFrame original de ventas.
        
        Excepciones:
        ValueError: Si el argumento no es un DataFrame o está vacío.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("El argumento df no es un DataFrame válido.")
            raise ValueError("El argumento df debe ser un pandas DataFrame.")
        if df.empty:
            logger.warning("El DataFrame está vacío.")
            raise ValueError("El DataFrame está vacío.")
        
        self.df = df.copy()  # Copia para no modificar el original
        logger.info("Inicio del proceso de limpieza de datos.")

    def convert_date(self):
        """
        Convierte la columna 'date' de texto a formato datetime.
        
        Excepciones:
        KeyError: Si la columna 'date' no está en el DataFrame.
        ValueError: Si la conversión de fecha falla debido a un formato incorrecto.
        """
        if "date" not in self.df.columns:
            logger.error("La columna 'date' no existe en el DataFrame.")
            raise KeyError("La columna 'date' no existe en el DataFrame.")
        
        if not pd.api.types.is_datetime64_any_dtype(self.df["date"]):
            try:
                self.df["date"] = pd.to_datetime(self.df["date"], format="%d.%m.%Y")
            except ValueError:
                logger.error("Formato de fecha incorrecto en la columna 'date'. Se esperaba '%d.%m.%Y'.")
                raise ValueError("Formato de fecha incorrecto en la columna 'date'. Se esperaba '%d.%m.%Y'.")
        return self

    def add_month_year(self):
        """
        Crea una nueva columna 'month_year' en formato 'YYYY-MM' para facilitar análisis temporales.
        
        Excepciones:
        KeyError: Si la columna 'date' no está en el DataFrame.
        ValueError: Si la columna 'date' contiene solo valores nulos.
        """
        if "date" not in self.df.columns:
            logger.error("La columna 'date' no existe en el DataFrame.")
            raise KeyError("La columna 'date' no existe en el DataFrame.")
        
        if self.df["date"].isnull().all():
            logger.warning("La columna 'date' contiene solo valores nulos, no se puede generar 'month_year'.")
            raise ValueError("La columna 'date' contiene solo valores nulos.")  
        self.df["month_year"] = self.df["date"].dt.strftime("%Y-%m")

    def execute_transformations(self):
        """
        Ejecuta todas las transformaciones en cadena y retorna el DataFrame procesado.
        
        Retorna:
        pd.DataFrame: DataFrame transformado con columnas 'date' convertida y 'month_year' generada.
        """
        logger.info("Ejecutando transformaciones de datos.")
        result = self.convert_date().add_month_year().df
        logger.info("Finalización del proceso de limpieza de datos.")
        return result

    # def convert_date(self):
    #     """
    #     Convierte la columna 'date' de texto a formato datetime.
    #     """
    #     self.df["date"] = pd.to_datetime(self.df["date"], format="%d.%m.%Y")
    #     return self

    # def fix_negative_sales(self):
    #     """
    #     Convierte valores negativos en 'item_cnt_day' a 0 para evitar problemas con devoluciones.
    #     """
    #     self.df["item_cnt_day"] = self.df["item_cnt_day"].clip(lower=0)
    #     return self

    # def add_month_year(self):
    #     """
    #     Crea una nueva columna 'month_year' en formato 'YYYY-MM' para facilitar análisis temporales.
    #     """
    #     self.df["month_year"] = self.df["date"].dt.strftime("%Y-%m")
    #     # self.df["month_year"] = self.df["date"].dt.to_period("M").dt.to_timestamp()
    #     return self

    # def execute_transformations(self):
    #     """
    #     Ejecuta todas las transformaciones en cadena.
    #     """
    #     return self.convert_date().add_month_year().df
