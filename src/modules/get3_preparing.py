import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class DataPreparation:
    def __init__(self, df, target="item_cnt_month"):
        """
        Inicializa la clase con un DataFrame y el target deseado.
        
        Parámetros:
        df (pd.DataFrame): DataFrame con los datos preparados por Feature Engineering.
        target (str): Variable objetivo ('item_cnt_month' para regresión o clasificación).
        """
        self.df = df.copy()
        self.target = target
        self.scaler = StandardScaler()
        self.transformed_columns = None
    
    def split_data(self, test_size=0.2):
        """ Divide los datos en entrenamiento y validación manteniendo la cronología. """
        date_blocks = self.df["date_block_num"].unique()
        split_index = int(len(date_blocks) * (1 - test_size))
        train_blocks, val_blocks = date_blocks[:split_index], date_blocks[split_index:]
        
        self.train_df = self.df[self.df["date_block_num"].isin(train_blocks)]
        self.val_df = self.df[self.df["date_block_num"].isin(val_blocks)]
        
        return self.train_df, self.val_df
    
    def handle_missing_values(self, df):
        """ Maneja valores faltantes en el DataFrame. """
        lag_features = ["sales_lag_1m", "sales_lag_3m", "sales_lag_6m"]
        df[lag_features] = df[lag_features].fillna(0)
        df.fillna(df.median(), inplace=True)
        return df
    
    def remove_unnecessary_columns(self, df):
        """ Elimina columnas innecesarias para el modelo. """
        drop_columns = ["shop_id", "item_id", "item_category_id"]
        df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
        return df
    
    def handle_infinite_values(self, df):
        """ Reemplaza valores infinitos por NaN y luego los maneja. """
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(), inplace=True)
        return df
    
    def remove_highly_correlated_features(self, df, threshold=0.80):
        """ Elimina variables altamente correlacionadas. """
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Graficar la matriz de correlación antes de eliminar variables (solo en entrenamiento)
        plt.figure(figsize=(9, 5))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.1f')
        plt.title("Matriz de Correlación de Variables Antes de la Eliminación")
        plt.show()
        
        df.drop(columns=to_drop, inplace=True)
        
        # Graficar la matriz de correlación después de eliminar variables (solo en entrenamiento)
        corr_matrix_after = df.corr().abs()
        plt.figure(figsize=(9, 5))
        sns.heatmap(corr_matrix_after, cmap='coolwarm', annot=True, fmt='.1f')
        plt.title("Matriz de Correlación de Variables Después de la Eliminación")
        plt.show()
        
        return df
    
    def scale_features(self, df, fit=False):
        """ Escala características numéricas con StandardScaler. """
        if self.transformed_columns is None:
            self.transformed_columns = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col != self.target]
        
        if fit:
            df[self.transformed_columns] = self.scaler.fit_transform(df[self.transformed_columns])
        else:
            df[self.transformed_columns] = self.scaler.transform(df[self.transformed_columns])
        
        return df
    
    def apply_pipeline(self):
        """ Aplica las transformaciones en train y luego en val. """
        # Transformaciones en el conjunto de entrenamiento
        self.train_df = self.handle_missing_values(self.train_df)
        self.train_df = self.remove_unnecessary_columns(self.train_df)
        self.train_df = self.handle_infinite_values(self.train_df)
        self.train_df = self.remove_highly_correlated_features(self.train_df)
        self.train_df = self.scale_features(self.train_df, fit=True)
        
        # Aplicar transformaciones en validación (sin ajuste)
        self.val_df = self.handle_missing_values(self.val_df)
        self.val_df = self.remove_unnecessary_columns(self.val_df)
        self.val_df = self.handle_infinite_values(self.val_df)
        self.val_df = self.scale_features(self.val_df, fit=False)
        
        X_train, y_train = self.train_df.drop(columns=[self.target]), self.train_df[self.target]
        X_val, y_val = self.val_df.drop(columns=[self.target]), self.val_df[self.target]
        
        return X_train, X_val, y_train, y_val
    
    def prepare_data(self):
        """ Ejecuta el proceso de preparación de datos. """
        self.split_data()
        return self.apply_pipeline()

# Uso de la clase con el DataFrame train
# data_prep = DataPreparation(train, target="item_cnt_month")
# X_train, X_val, y_train, y_val = data_prep.prepare_data()
