"""
Módulo para la ingeniería de características en datos de ventas.

Este módulo define la clase `FeatureEngineering`, que proporciona métodos para generar
características a partir de datos de ventas. Las transformaciones incluyen la extracción
de variables temporales, agregaciones de ventas, precios y categorías, y la fusión de 
todas estas características en un solo DataFrame.

Clases:
    - FeatureEngineering: Clase para la ingeniería de características en datos de ventas.

Excepciones:
    - ValueError: Se lanza cuando el DataFrame de entrada es inválido o vacío.
    - KeyError: Se lanza cuando las columnas esperadas no están presentes en el DataFrame.
"""
import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from logs.logger_config import logger # pylint: disable=C0413

class FeatureEngineering:
    """
    Clase para la generación de características (features) en el conjunto de datos de ventas.
    """

    def __init__(self, df: pd.DataFrame, target: str = "item_cnt_month"):
        """
        Inicializa la clase con un DataFrame y la variable objetivo.
        
        Parámetros:
        df (pd.DataFrame): DataFrame con los datos de ventas.
        target (str): Variable objetivo ('item_cnt_month' para regresión o clasificación).
        
        Excepciones:
        ValueError: Si df no es un DataFrame o está vacío.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("El argumento df no es un DataFrame válido.")
            raise ValueError("El argumento df debe ser un pandas DataFrame.")
        if df.empty:
            logger.warning("El DataFrame está vacío.")
            raise ValueError("El DataFrame está vacío.")

        self.df = df.copy()
        self.target = target
        logger.info("Inicio del proceso de ingeniería de características.")

    def add_time_features(self):
        """
        Agrega variables temporales derivadas de la columna 'date'.
        
        Excepciones:
        KeyError: Si la columna 'date' no está presente en el DataFrame.
        ValueError: Si la conversión de fecha falla debido a un formato incorrecto.
        """
        if "date" not in self.df.columns:
            logger.error("La columna 'date' no existe en el DataFrame.")
            raise KeyError("La columna 'date' no existe en el DataFrame.")

        try:
            self.df["date"] = pd.to_datetime(self.df["date"], format="%d.%m.%Y")
        except ValueError as exc:
            logger.error("Formato incorrecto en 'date'. Se esperaba '%d.%m.%Y'.")
            raise ValueError("Formato incorrecto en 'date'. Se esperaba '%d.%m.%Y'.") from exc

        self.df["date_block_num"] = (self.df["date"].dt.year - self.df["date"].dt.year.min()) * 12 + self.df["date"].dt.month
        self.df["month"] = self.df["date"].dt.month
        self.df["year"] = self.df["date"].dt.year
        self.df["quarter"] = self.df["date"].dt.quarter
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["is_weekend"] = self.df["day_of_week"].isin([5, 6]).astype(int)

        logger.info("Variables temporales agregadas correctamente.")
        return self.df

    def aggregate_sales(self):
        """
        Agrega las ventas a nivel mensual agrupando por tienda y producto.
        
        Retorna:
        pd.DataFrame: DataFrame con las ventas agregadas a nivel de mes.
        
        Excepciones:
        KeyError: Si la columna 'item_cnt_day' no está en el DataFrame.
        """
        if "item_cnt_day" not in self.df.columns:
            logger.error("La columna 'item_cnt_day' no existe en el DataFrame.")
            raise KeyError("La columna 'item_cnt_day' no existe en el DataFrame.")

        logger.info("Agregando ventas a nivel mensual...")
        df_sales = (self.df.groupby(
                                    ["date_block_num", "shop_id", "item_id"], as_index=False
                                   )
                           .agg({"item_cnt_day": "sum"})
        )
        df_sales = df_sales.rename(columns={"item_cnt_day": "item_cnt_month"})
        logger.info("Ventas agregadas correctamente a nivel mensual.")
        return df_sales

    def aggregate_prices(self):
        """
        Agrega los precios promedio a nivel mensual agrupando por tienda y producto.
        
        Retorna:
        pd.DataFrame: DataFrame con los precios agregados a nivel de mes.
        
        Excepciones:
        KeyError: Si la columna 'item_price' no está en el DataFrame.
        """
        if "item_price" not in self.df.columns:
            logger.error("La columna 'item_price' no existe en el DataFrame.")
            raise KeyError("La columna 'item_price' no existe en el DataFrame.")

        logger.info("Agregando precios a nivel mensual...")
        df_price = (
                    self.df.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
                           .agg({"item_price": "mean"})
        )
        logger.info("Precios agregados correctamente a nivel mensual.")
        return df_price

    def aggregate_categories(self):
        """
        Agrega información de categorías a nivel mensual con múltiples métricas, incluyendo la moda.
        
        Retorna:
        pd.DataFrame: DataFrame con información agregada de categorías.
        
        Excepciones:
        KeyError: Si la columna 'item_category_id' no está en el DataFrame.
        """
        if "item_category_id" not in self.df.columns:
            logger.error("La columna 'item_category_id' no existe en el DataFrame.")
            raise KeyError("La columna 'item_category_id' no existe en el DataFrame.")

        logger.info("Agregando información de categorías a nivel mensual...")
        df_category = self.df.groupby(["date_block_num", "shop_id", "item_id"], as_index=False).agg({
            "item_category_id": ["nunique", "count", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan],
            "item_category_name_es": [lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan]
        })
        df_category.columns = ["date_block_num",
                               "shop_id",
                               "item_id",
                               "unique_categories",
                               "category_sales_count",
                               "most_frequent_category_id",
                               "most_frequent_category"]
        logger.info("Información de categorías agregada correctamente a nivel mensual.")
        return df_category

    # def merge_aggregated_features(self):
    #     """
    #     Une todas las características agregadas en un único DataFrame.
        
    #     Retorna:
    #     pd.DataFrame: DataFrame con todas las variables agregadas combinadas.
        
    #     Excepciones:
    #     ValueError: Si las funciones de agregación no se han ejecutado previamente.
    #     """
    #     logger.info("Fusionando características agregadas...")
    #     df_sales = self.aggregate_sales()
    #     df_price = self.aggregate_prices()
    #     df_category = self.aggregate_categories()

    #     if df_sales is None or df_price is None or df_category is None:
    #         logger.error("No se han generado correctamente las agregaciones previas.")
    #         raise ValueError("Las funciones de agregación deben ejecutarse antes de la fusión.")

    #     self.df = df_sales.merge(df_price, on=["date_block_num", "shop_id", "item_id"], how="left")
    #     self.df = (
    #                self.df.merge(
    #                              df_category, on=["date_block_num", "shop_id", "item_id"],
    #                              how="left"
    #                             )
    #     )
    #     logger.info("Características fusionadas correctamente.")
    #     return self.df

    def merge_aggregated_features(self):
        """
        Une todas las características agregadas en un único DataFrame.

        Retorna:
        pd.DataFrame: DataFrame con todas las variables agregadas combinadas.

        Excepciones:
        ValueError: Si las funciones de agregación no se han ejecutado previamente.
        """
        logger.info("Fusionando características agregadas...")

        df_sales = self.aggregate_sales()
        df_price = self.aggregate_prices()
        df_category = self.aggregate_categories()

        if df_sales is None or df_price is None or df_category is None:
            logger.error("No se han generado correctamente las agregaciones previas.")
            raise ValueError("Las funciones de agregación deben ejecutarse antes de la fusión.")

        # Asegurar que las columnas coincidan antes de fusionarlas
        self.df = df_sales.merge(df_price, on=["date_block_num", "shop_id", "item_id"], how="left")

        common_columns = set(self.df.columns).intersection(set(df_category.columns))
        if len(common_columns) == 0:
            logger.warning("No hay columnas en común entre df y df_category, verifica la agregación.")

        self.df = self.df.merge(df_category, on=["date_block_num", "shop_id", "item_id"], how="left")
        logger.info("Características fusionadas correctamente.")
        
        return self.df

    def add_sales_features(self):
        """
        Agrega variables de ventas acumuladas y tendencia a nivel mensual.
        
        Retorna:
        pd.DataFrame: DataFrame con las características de ventas agregadas.
        
        Excepciones:
        KeyError: Si la variable objetivo no está en el DataFrame.
        """
        if self.target not in self.df.columns:
            logger.error(f"La variable objetivo '{self.target}' no existe en el DataFrame.")
            raise KeyError(f"La variable objetivo '{self.target}' no existe en el DataFrame.")

        logger.info("Agregando características de ventas...")
        self.df["total_sales"] = (
            self.df.groupby(["shop_id", "item_id"])[self.target].transform("sum")
        )
        self.df["avg_sales_per_shop_item"] = (
            self.df.groupby(["shop_id", "item_id"])[self.target].transform("mean")
        )
        self.df["sales_lag_1m"] = (
            self.df.groupby(["shop_id", "item_id"]).shift(1).fillna(0)[self.target]
        )
        self.df["sales_lag_3m"] = (
            self.df.groupby(["shop_id", "item_id"]).shift(3).fillna(0)[self.target]
        )
        self.df["sales_lag_6m"] = (
            self.df.groupby(["shop_id", "item_id"]).shift(6).fillna(0)[self.target]
        )
        logger.info("Características de ventas agregadas correctamente.")
        return self.df

    def add_price_features(self):
        """
        Agrega variables de precios avanzadas sin perder información de categorías.
        
        Retorna:
        pd.DataFrame: DataFrame con características avanzadas de precios.
        
        Excepciones:
        KeyError: Si la columna 'item_price' no está en el DataFrame.
        """
        if "item_price" not in self.df.columns:
            logger.error("La columna 'item_price' no existe en el DataFrame.")
            raise KeyError("La columna 'item_price' no existe en el DataFrame.")

        logger.info("Agregando características de precios...")
        self.df["avg_price_per_shop_item"] = (
            self.df.groupby(["shop_id", "item_id"])["item_price"].transform("mean")
        )
        self.df["price_deviation"] = (
        (self.df["item_price"] - self.df["avg_price_per_shop_item"]) / self.df["avg_price_per_shop_item"]
        )
        logger.info("Características de precios agregadas correctamente.")
        return self.df

    # def add_category_features(self):
    #     """
    #     Agrega variables relacionadas con la categoría de los productos.
        
    #     Retorna:
    #     pd.DataFrame: DataFrame con características avanzadas de categorías.
        
    #     Excepciones:
    #     KeyError: Si la columna 'item_category_id' no está en el DataFrame.
    #     """
    #     if "item_category_id" not in self.df.columns:
    #         logger.error("La columna 'item_category_id' no existe en el DataFrame.")
    #         raise KeyError("La columna 'item_category_id' no existe en el DataFrame.")

    #     logger.info("Agregando características de categorías...")
    #     self.df["avg_sales_per_category"] = (
    #         self.df.groupby("item_category_id")[self.target].transform("mean")
    #     )
    #     self.df["category_trend_last_3m"] = (
    #         self.df.groupby("item_category_id")[self.target].pct_change(periods=3).fillna(0)
    #     )
    #     self.df["category_sales_rank"] = (
    #         self.df.groupby("item_category_id")[self.target].rank(ascending=False)
    #     )
    #     logger.info("Características de categorías agregadas correctamente.")
    #     return self.df

    def apply_feature_engineering(self):
        """
        Ejecuta todas las funciones de generación de características en orden.
        
        Retorna:
        pd.DataFrame: DataFrame con todas las características generadas.
        """
        logger.info("Ejecutando proceso completo de ingeniería de características...")
        self.add_time_features()
        self.merge_aggregated_features()
        self.add_sales_features()
        self.add_price_features()
        # self.add_category_features()
        logger.info("Proceso de ingeniería de características completado.")
        return self.df
