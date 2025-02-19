import pandas as pd

class FeatureEngineering:
    def __init__(self, df, target="item_cnt_month"):
        """
        Inicializa la clase con un DataFrame y el target deseado a nivel mensual.
        
        Parámetros:
        df (pd.DataFrame): DataFrame con los datos de ventas.
        target (str): Variable objetivo ('item_cnt_month' para regresión o clasificación).
        """
        self.df = df.copy()
        self.target = target
    
    def add_time_features(self):
        """ Agrega variables temporales mejoradas. """
        self.df["date"] = pd.to_datetime(self.df["date"], format="%d.%m.%Y")
        self.df["date_block_num"] = (self.df["date"].dt.year - self.df["date"].dt.year.min()) * 12 + self.df["date"].dt.month
        self.df["month"] = self.df["date"].dt.month
        self.df["year"] = self.df["date"].dt.year
        self.df["quarter"] = self.df["date"].dt.quarter
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["is_weekend"] = self.df["day_of_week"].isin([5, 6]).astype(int)
        return self.df
    
    def aggregate_sales(self):
        """ Crea el dataset de ventas a nivel mensual. """
        return self.df.groupby(["date_block_num", "shop_id", "item_id"], as_index=False).agg({"item_cnt_day": "sum"}).rename(columns={"item_cnt_day": "item_cnt_month"})
    
    def aggregate_prices(self):
        """ Crea el dataset de precios a nivel mensual. """
        return self.df.groupby(["date_block_num", "shop_id", "item_id"], as_index=False).agg({"item_price": "mean"})
    
    def aggregate_categories(self):
            """ Crea el dataset de categorías a nivel mensual con múltiples agregaciones. """
            return self.df.groupby(["date_block_num", "shop_id", "item_id"]).agg({
                "item_category_id": "first",  # Mantiene la categoría principal
                "item_category_id": "nunique",  # Número de categorías distintas por combinación
                "item_category_id": "count"  # Número total de ventas en la categoría
            }).rename(columns={"item_category_id_first": "main_category_id", "item_category_id_nunique": "unique_categories", "item_category_id_count": "category_sales_count"})
        
    
    def merge_aggregated_features(self):
        """ Une todas las características agregadas en un único DataFrame. """
        df_sales = self.aggregate_sales()
        df_price = self.aggregate_prices()
        df_category = self.aggregate_categories()
        
        self.df = df_sales.merge(df_price, on=["date_block_num", "shop_id", "item_id"], how="left")
        self.df = self.df.merge(df_category, on=["date_block_num", "shop_id", "item_id"], how="left")
        return self.df
    
    def add_sales_features(self):
        """ Agrega variables de ventas acumuladas y tendencia a nivel mensual. """
        self.df["total_sales"] = self.df.groupby(["shop_id", "item_id"])[self.target].transform("sum")
        self.df["avg_sales_per_shop_item"] = self.df.groupby(["shop_id", "item_id"])[self.target].transform("mean")
        self.df["sales_lag_1m"] = self.df.groupby(["shop_id", "item_id"]).shift(1).fillna(0)[self.target]
        self.df["sales_lag_3m"] = self.df.groupby(["shop_id", "item_id"]).shift(3).fillna(0)[self.target]
        self.df["sales_lag_6m"] = self.df.groupby(["shop_id", "item_id"]).shift(6).fillna(0)[self.target]
        return self.df
    
    def add_price_features(self):
        """ Agrega variables de precios avanzadas sin perder información de categorías. """
        self.df["avg_price_per_shop_item"] = self.df.groupby(["shop_id", "item_id"])["item_price"].transform("mean")
        self.df["price_deviation"] = (self.df["item_price"] - self.df["avg_price_per_shop_item"]) / self.df["avg_price_per_shop_item"]
        return self.df
    
    def add_category_features(self):
        """ Agrega variables relacionadas con la categoría de los productos. """
        self.df["avg_sales_per_category"] = self.df.groupby("item_category_id")[self.target].transform("mean")
        self.df["category_trend_last_3m"] = self.df.groupby("item_category_id")[self.target].pct_change(periods=3).fillna(0)
        self.df["category_sales_rank"] = self.df.groupby("item_category_id")[self.target].rank(ascending=False)
        return self.df
    
    def apply_feature_engineering(self):
        """ Ejecuta todas las funciones de generación de características. """
        self.add_time_features()
        self.merge_aggregated_features()
        self.add_sales_features()
        self.add_price_features()
        self.add_category_features()
        return self.df

# Uso de la clase con el DataFrame train
# train = FeatureEngineering(train, target="item_cnt_month").apply_feature_engineering()
