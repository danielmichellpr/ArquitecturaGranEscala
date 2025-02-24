import sys
import os
# Agrega la raíz del proyecto a sys.path para que Python encuentre `main.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import SalesPredictionPipeline
from logs.logger_config import logger

if __name__ == "__main__":
    logger.info("Iniciando limpieza de datos...")
    pipeline = SalesPredictionPipeline(force_cleaning=True, force_engineering=False, force_training=False)
    pipeline.data_cleaning()
    logger.info("Limpieza de datos finalipzada.")
