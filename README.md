# ArquitecturaGranEscala

**Repositorio de la clase Arquitectura de Datos - Gran Escala de la Maestría en  Ciencia de Datos**

# Proyecto de Predicción de Ventas

Este proyecto implementa un pipeline de predicción de ventas utilizando técnicas de Machine Learning y Series Temporales.
El código está parametrizado con un archivo YAML para permitir configuraciones dinámicas sin modificar el código.

## Estructura del Proyecto

📂 tu_proyecto/
 ├── 📂 src/                     # Código fuente
 │    ├── 📂 modules/            # Módulos del pipeline
 │    │    ├── data_utils.py     # Procesamiento de datos
 │    │    ├── eda_utils.py      # Exploración de datos
 │    │    ├── get1_cleaning.py  # Limpieza de datos
 │    │    ├── get2_engineering.py  # Ingeniería de características
 │    │    ├── get3_preparing.py  # Preparación de datos
 │    │    ├── get4_select_model.py  # Selección y entrenamiento de modelo
 │    │    ├── model_training.py  # Entrenamiento del modelo
 │
 ├── 📂 tests/                    # Pruebas unitarias con pytest
 │    ├── test_data_preparation.py
 │    ├── test_feature_engineering.py
 │    ├── test_model_training.py
 │
 ├── 📂 logs/                     # Registros del pipeline
 ├── 📂 data/                     # Datos utilizados en el pipeline
 │    ├── 📂 raw/                 # Datos originales sin procesar
 │    ├── 📂 results/             # Resultados finales del modelo
 │
 ├── 📂 models/                   # Modelos entrenados
 │    ├── random_forest_model.pkl
 │    ├── full_predictions.csv
 │
 ├── config.yaml                  # Archivo de configuración paramétrico
 ├── README.md                     # Documentación del proyecto
 ├── main.py                       # Ejecución del pipeline completo

## Flujo del Proyecto

1️⃣ Carga de Configuración (config.yaml) → Define rutas, hiperparámetros y tipo de modelo.

2️⃣ Carga de Datos (data/raw/) → Se leen los archivos de ventas, tiendas y productos.

3️⃣ Limpieza de Datos (SalesCleaner) → Se transforman y limpian los datos crudos.

4️⃣ Traducción de Datos (DataProcessor) → Traduce nombres de tiendas y categorías.

5️⃣ Fusión de Datos → Se combinan los archivos de ventas, tiendas, productos y categorías.

6️⃣ Eliminación de Outliers (detect_outliers) → Se detectan valores atípicos en precios.

7️⃣ Ingeniería de Características (FeatureEngineering) → Se generan nuevas variables para el modelo.

8️⃣ Preparación de Datos (DataPreparation) → Normalización y eliminación de correlaciones.

9️⃣ Entrenamiento del Modelo (ModelTraining) → Se entrena un modelo y se guarda en models/.

🔟 Generación de Predicciones → Se aplican las predicciones al conjunto de prueba.


## Cómo Ejecutar el Proyecto

1️⃣ Instalar dependencias

pip install -r requirements.txt

2️⃣ Ejecutar el pipeline completo

python main.py

Esto generará:

✅ submission.csv → Predicciones para el conjunto de prueba.

✅ full_predictions.csv → Predicciones para todo el dataset procesado.

✅ models/random_forest_model.pkl → Modelo entrenado guardado.

## Ejecución de Pruebas Unitarias

Las pruebas unitarias verifican cada módulo de procesamiento.

pytest tests/

Si solo deseas ejecutar un test en particular:

pytest tests/test_model_training.py

🔥 Mejoras Futuras

🚀 Optimizar hiperparámetros con GridSearchCV.
🚀 Implementar monitoreo de modelo en producción.