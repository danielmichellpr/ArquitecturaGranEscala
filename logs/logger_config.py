import logging
import os

# Obtener la ruta absoluta del directorio del proyecto
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Esto apunta a `logs/`
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # Esto apunta a la raíz del proyecto

# Ruta fija para guardar los logs en `logs/pipeline.log`
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")  # Archivo de logs

# Configuración del logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

# Crear y exportar el logger
logger = logging.getLogger("SALES")
