# ArquitecturaGranEscala

**Repositorio de la clase Arquitectura de Datos - Gran Escala de la Maestría en  Ciencia de Datos**

# Proyecto de Predicción de Ventas

Este proyecto implementa un pipeline E2E de predicción de ventas utilizando técnicas de Machine Learning y Series Temporales.
El código está diseñado de manera modular y permite la ejecución controlada mediante banderas y configuración YAML.

## Estructura del Proyecto
```
📂 tu_proyecto/
 ├── 📄 README.md                  # Documentación del proyecto
 ├── 📄 app.py                      # Aplicación principal (si aplica)
 ├── 📄 config.yaml                  # Archivo de configuración paramétrico
 ├── 📂 data/                        # Datos utilizados en el pipeline
 │    ├── 📂 output/                 # Datos procesados en diferentes etapas
 │    │    ├── data_cleaning.csv      # Datos después de la limpieza
 │    │    ├── data_engineering.csv   # Datos después de la ingeniería de características
 │    │    ├── data_predictions.csv   # Predicciones generadas
 │    ├── 📂 raw/                     # Datos originales sin procesar
 │    │    ├── item_categories.csv
 │    │    ├── items.csv
 │    │    ├── sales_train.csv
 │    │    ├── sample_submission.csv
 │    │    ├── shops.csv
 │    │    ├── test.csv
 ├── 📄 environment.yml               # Configuración de entorno
 ├── 📂 logs/                         # Registros del pipeline
 │    ├── logger_config.py            # Configuración del logger
 │    ├── pipeline.log                 # Archivo de logs generados durante la ejecución
 ├── 📄 main.py                        # Archivo principal que ejecuta todo el pipeline
 ├── 📂 models/                        # Modelos entrenados y características utilizadas
 │    ├── model_features.txt          # Lista de variables utilizadas en el modelo
 │    ├── xgboost_model.pkl           # Modelo entrenado
 ├── 📂 notebooks/                     # Notebooks de análisis y pruebas
 │    ├── main.ipynb
 │    ├── main_all.ipynb
 │    ├── main_testing.ipynb
 ├── 📂 src/                           # Código fuente
 │    ├── 📂 modules/                  # Módulos del pipeline
 │    │    ├── data_utils.py           # Procesamiento de datos
 │    │    ├── eda_utils.py            # Exploración de datos
 │    │    ├── get1_cleaning.py        # Limpieza de datos
 │    │    ├── get2_engineering.py     # Ingeniería de características
 │    │    ├── get3_preparing.py       # Preparación de datos
 │    │    ├── get4_select_model.py    # Selección y entrenamiento de modelo
 │    │    ├── others_modules.py       # Otros módulos adicionales
 ├── 📂 tests/                         # Pruebas unitarias con pytest
 │    ├── test_get1_cleaning.py        # Prueba para limpieza de datos
 │    ├── test_get2_engineering.py     # Prueba para ingeniería de características
 │    ├── test_get3_preparing.py       # Prueba para preparación de datos
 │    ├── test_get4_select_model.py    # Prueba para selección y entrenamiento de modelo
```

## Flujo del Proyecto

1️⃣ Carga de Configuración (config.yaml) → Define rutas, modelos y parámetros de ejecución.

2️⃣ Carga de Datos (data/raw/) → Se leen los archivos de ventas, tiendas y productos.

3️⃣ Limpieza de Datos (SalesCleaner) → Se transforman y limpian los datos crudos.

4️⃣ Traducción de Datos (DataProcessor) → Traduce nombres de tiendas y categorías.

5️⃣ Fusión de Datos → Se combinan los archivos de ventas, tiendas, productos y categorías.

6️⃣ Eliminación de Outliers (detect_outliers) → Se detectan valores atípicos en precios.

7️⃣ Ingeniería de Características (FeatureEngineering) → Se generan nuevas variables para el modelo.

8️⃣ Preparación de Datos (DataPreparation) → Normalización y eliminación de correlaciones.

9️⃣ Entrenamiento del Modelo (ModelTraining) → Se entrena un modelo y se guarda en models/.

🔟 Generación de Predicciones → Se aplican las predicciones al conjunto de prueba y se guardan en data/results/

## Configuración con YAML (config.yaml)
```yaml
data:
  raw_path: "../data/raw/"
  results_path: "../data/results/"
  models_path: "../models/"

preprocessing:
  detect_outliers:
    column: "item_price"
    iqr_factor: 1.5
    z_threshold: 3
    lower_percentile: 0.01
    upper_percentile: 0.99

model:
  types:
    - "random_forest"
    - "xgboost"
    - "arima"
    - "exponential_smoothing"
  selected: "xgboost"

output:
  cleaned_data_filename: "../data/data_cleaning.csv"
  engineered_data_filename: "../data/data_engineering.csv"
  predictions_data_filename: "../data/data_predictions.csv"
  model_features_filename: "../models/model_features.txt"

flags:
  force_cleaning: false
  force_engineering: false
  force_training: false
```

## Cómo Ejecutar el Proyecto

```python
python main.py
```

Esto generará:

✅ `data_predictions.csv`→ Predicciones para todo el dataset procesado.

✅ `models/xgboost_model.pkl` → Modelo entrenado guardado.

## Ejecución de Pruebas Unitarias

`pytest tests/`

Si solo deseas ejecutar un test en particular:

`pytest tests/test_model_training.py`