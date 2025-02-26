"""
Módulo que implementa un pipeline E2E para la predicción de ventas.

Este módulo define la clase `SalesPredictionPipeline`, que permite realizar el preprocesamiento de datos,
la ingeniería de características y el entrenamiento de modelos de Machine Learning de manera estructurada
y reutilizable. Se basa en un enfoque modular que permite cargar y reutilizar datos previamente procesados
o ejecutar nuevamente los pasos si el usuario lo requiere.

Características:
    - **Carga dinámica de datos**: Los datos se procesan solo si no existen archivos previos o si el usuario lo solicita.
    - **Parámetros configurables**: Se usa un archivo `config.yaml` para definir rutas, modelos, hiperparámetros y flags de ejecución.
    - **Limpieza de datos**: Transformación de datos brutos (`SalesCleaner`).
    - **Ingeniería de características**: Generación de variables (`FeatureEngineering`).
    - **Preparación de datos**: Eliminación de valores atípicos y escalado (`DataPreparation`).
    - **Entrenamiento de modelos**: Compatibilidad con modelos `Random Forest`, `XGBoost`, `ARIMA` y `Exponential Smoothing`.

Clases:
    - SalesPredictionPipeline: Ejecuta el pipeline de predicción de ventas.

Excepciones:
    - FileNotFoundError: Si los archivos de datos no existen.
    - ValueError: Si los datos de entrada son inválidos.
    - Exception: Manejo de errores en cada paso del pipeline.

Ejemplo de uso:
    >>> pipeline = SalesPredictionPipeline(force_cleaning=True, force_engineering=True, force_training=True)
    >>> pipeline.run_pipeline()
"""

import os
import yaml
import pandas as pd
from src.modules.data_utils import DataProcessor
from src.modules.eda_utils import detect_outliers
from src.modules.get1_cleaning import SalesCleaner
from src.modules.get2_engineering import FeatureEngineering
from src.modules.get3_preparing import DataPreparation
from src.modules.get4_select_model import ModelTraining
from logs.logger_config import logger
import joblib

class SalesPredictionPipeline:
    """
    Clase que ejecuta el pipeline E2E de predicción de ventas, verificando si los datos han sido procesados previamente
    y permitiendo la recarga de archivos en caso de que ya existan.
    """
    
    def __init__(self, config_path="./config.yaml", force_cleaning=False, force_engineering=False, force_training=False):
        """
        Inicializa la clase cargando la configuración desde un archivo YAML y verificando las banderas de ejecución.
        
        Parámetros:
        - config_path (str): Ruta al archivo de configuración YAML.
        - force_cleaning (bool): Si es True, fuerza la ejecución de la limpieza de datos.
        - force_engineering (bool): Si es True, fuerza la ejecución de la ingeniería de características.
        - force_training (bool): Si es True, fuerza la ejecución del entrenamiento del modelo.
        """
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            logger.error("Error al cargar el archivo de configuración: %s", str(e))
            raise
        
        self.force_cleaning = self.config["flags"]["force_cleaning"]
        self.force_engineering = self.config["flags"]["force_engineering"]
        self.force_training = self.config["flags"]["force_training"]
        
        self.data_path = self.config["data"]["raw_path"]
        self.results_path = self.config["data"]["results_path"]
        self.models_path = self.config["data"]["models_path"]
        
        self.cleaned_data_path = self.config["output"]["cleaned_data_filename"]
        self.engineered_data_path = self.config["output"]["engineered_data_filename"]
        self.predictions_data_path = self.config["output"]["predictions_data_filename"]
        self.model_type = self.config["model"]["selected"]
        self.model_filename = os.path.join(self.models_path, f"{self.model_type}_model.pkl")
    
    def data_cleaning(self):
        """Ejecuta la limpieza de datos si no existe el archivo o si se fuerza la ejecución."""
        try:
            if not os.path.exists(self.cleaned_data_path) or self.force_cleaning:
                logger.info("Ejecutando limpieza de datos...")
                sales_train = pd.read_csv(os.path.join(self.data_path, "sales_train.csv"))
                shops = pd.read_csv(os.path.join(self.data_path, "shops.csv"))
                items = pd.read_csv(os.path.join(self.data_path, "items.csv"))
                item_categories = pd.read_csv(os.path.join(self.data_path, "item_categories.csv"))
                
                sales_train = SalesCleaner(sales_train).execute_transformations()
                shops = DataProcessor.translate_column(shops, "shop_name")
                item_categories = DataProcessor.translate_column(item_categories, "item_category_name")
                
                train = sales_train.merge(shops, on="shop_id", how="left")
                train = train.merge(items, on="item_id", how="left")
                train = train.merge(item_categories, on="item_category_id", how="left")
                
                _, filtered_data_ip = detect_outliers(
                    train,
                    column=self.config["preprocessing"]["detect_outliers"]["column"],
                    iqr_factor=self.config["preprocessing"]["detect_outliers"]["iqr_factor"],
                    z_threshold=self.config["preprocessing"]["detect_outliers"]["z_threshold"],
                    lower_percentile=self.config["preprocessing"]["detect_outliers"]["lower_percentile"],
                    upper_percentile=self.config["preprocessing"]["detect_outliers"]["upper_percentile"]
                )
                train = filtered_data_ip["Z-Score"].reset_index(drop=True)
                train.to_csv(self.cleaned_data_path, index=False)
                logger.info("Limpieza de datos completada y guardada en %s", self.cleaned_data_path)
            else:
                logger.info("Cargando datos limpios desde archivo %s", self.cleaned_data_path)
                train = pd.read_csv(self.cleaned_data_path)
            return train
        except Exception as e:
            logger.error("Error durante la limpieza de datos: %s", str(e))
            raise
    
    def feature_engineering(self, train):
        """Ejecuta la ingeniería de características si no existe el archivo o si se fuerza la ejecución."""
        try:
            if not os.path.exists(self.engineered_data_path) or self.force_engineering:
                logger.info("Ejecutando ingeniería de características...")
                train["date"] = pd.to_datetime(train["date"], format="%Y-%m-%d")
                train_model = FeatureEngineering(train, target="item_cnt_month").apply_feature_engineering()
                train_model.to_csv(self.engineered_data_path, index=False)
                logger.info("Ingeniería de características completada y guardada en %s", self.engineered_data_path)
            else:
                logger.info("Cargando datos con ingeniería de características desde archivo %s", self.engineered_data_path)
                train_model = pd.read_csv(self.engineered_data_path)
            return train_model
        except Exception as e:
            logger.error("Error durante la ingeniería de características: %s", str(e))
            raise
    
    def apply_modeling(self, train_model):
        """Ejecuta el entrenamiento del modelo si no existe el archivo de predicciones o si se fuerza la ejecución."""
        try:
            if not os.path.exists(self.predictions_data_path) or not os.path.exists(self.model_filename) or self.force_training:
                logger.info("Ejecutando entrenamiento del modelo...")
                X_train, X_val, y_train, y_val = DataPreparation(train_model, target="item_cnt_month").prepare_data()
                # Obtener las columnas utilizadas en el modelo
                no_use_columns = ["date_block_num", "shop_id", "item_id", "most_frequent_category_id", "most_frequent_category", "item_cnt_month"]
                model_features = X_train[[col for col in X_train if col not in no_use_columns]].columns.tolist()
                
                model_trainer = ModelTraining(X_train, X_val, y_train, y_val, self.config["model"]["selected"])
                full_data, best_model = model_trainer.train_and_evaluate_model()
                full_data = pd.concat([full_data,pd.concat([y_train, y_val])],axis=1)
                full_data.to_csv(self.predictions_data_path, index=False)
                os.makedirs(self.models_path, exist_ok=True)
                joblib.dump(best_model, self.model_filename)
                
                # Guardar las columnas utilizadas en el modelo
                feature_list_path = os.path.join(self.models_path, "model_features.txt")
                with open(feature_list_path, "w") as file:
                    file.write("\n".join(model_features))
                
                logger.info("Modelo guardado en: %s", self.model_filename)
                logger.info("Lista de variables utilizadas guardada en: %s", feature_list_path)
            else:
                logger.info("Cargando modelo previamente entrenado...")
                best_model = joblib.load(self.model_filename)
                full_data = pd.read_csv(self.predictions_data_path)
                # Cargar las columnas utilizadas en el modelo
                feature_list_path = os.path.join(self.models_path, "model_features.txt")
                if os.path.exists(feature_list_path):
                    with open(feature_list_path, "r") as file:
                        model_features = file.read().splitlines()
                    logger.info("Variables utilizadas en el modelo: %s", model_features)
                else:
                    model_features = None
            
            # return full_data, best_model, model_features
        except Exception as e:
            logger.error("Error durante el entrenamiento del modelo: %s", str(e))
            raise
    
    
    def run_pipeline(self):
        """Ejecuta el pipeline completo paso a paso, respetando si los archivos ya existen."""
        train = self.data_cleaning()
        train_model = self.feature_engineering(train)
        #full_data, best_model, model_features = 
        self.apply_modeling(train_model)
        logger.info("Proceso Finalizado")

#Ejecutar el pipeline con parámetros
if __name__ == "__main__":
   pipeline = SalesPredictionPipeline(force_cleaning=False, force_engineering=False, force_training=False)
   pipeline.run_pipeline()
