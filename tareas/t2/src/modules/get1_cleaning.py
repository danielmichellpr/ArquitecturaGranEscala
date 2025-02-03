import pandas as pd

class SalesCleaner:
    """
    Clase para procesar el DataFrame de ventas 'sales_train' con transformaciones clave.
    """

    def __init__(self, df):
        """
        Inicializa la clase con un DataFrame.
        Parámetros:
        df (pd.DataFrame): DataFrame original de ventas.
        """
        self.df = df.copy()  # Copia para no modificar el original

    def convert_date(self):
        """
        Convierte la columna 'date' de texto a formato datetime.
        """
        self.df["date"] = pd.to_datetime(self.df["date"], format="%d.%m.%Y")
        return self

    def add_month_year(self):
        """
        Crea una nueva columna 'month_year' en formato 'YYYY-MM' para facilitar análisis temporales.
        """
        self.df["month_year"] = self.df["date"].dt.strftime("%Y-%m")
        # self.df["month_year"] = self.df["date"].dt.to_period("M").dt.to_timestamp()
        return self

    # def fix_negative_sales(self):
    #     """
    #     Convierte valores negativos en 'item_cnt_day' a 0 para evitar problemas con devoluciones.
    #     """
    #     self.df["item_cnt_day"] = self.df["item_cnt_day"].clip(lower=0)
    #     return self

    def execute_transformations(self):
        """
        Ejecuta todas las transformaciones en cadena.
        """
        return self.convert_date().add_month_year().df
