import sys
import os
# Agrega la raíz del proyecto a sys.path para que Python encuentre `main.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import SalesPredictionPipeline
from logs.logger_config import logger
import pandas as pd

if __name__ == "__main__":
    logger.info("Iniciando entrenamiento del modelo...")
    pipeline = SalesPredictionPipeline(force_cleaning=False, 
                                       force_engineering=False, force_training=True)
    
    # Leer los datos con ingeniería de características
    engineered_data = pd.read_csv(pipeline.engineered_data_path)

    pipeline.apply_modeling(engineered_data)
    logger.info("Entrenamiento del modelo finalizado.")
