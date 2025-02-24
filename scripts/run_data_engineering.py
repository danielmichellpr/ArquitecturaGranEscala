import sys
import os
# Agrega la raíz del proyecto a sys.path para que Python encuentre `main.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import SalesPredictionPipeline
from logs.logger_config import logger
import pandas as pd

if __name__ == "__main__":
    logger.info("Iniciando ingeniería de características...")
    pipeline = SalesPredictionPipeline(force_cleaning=False, 
                                       force_engineering=True, force_training=False)
    
    # Leer los datos limpios generados en la fase anterior
    cleaned_data = pd.read_csv(pipeline.cleaned_data_path)
    
    pipeline.feature_engineering(cleaned_data)
    logger.info("Ingeniería de características finalizada.")
