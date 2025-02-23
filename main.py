import pandas as pd
import sys
import os
import yaml

# Cargar configuración desde YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Agrega la ruta de src al sys.path
# sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from src.modules.data_utils import DataProcessor
from src.modules.eda_utils import detect_outliers
from src.modules.get1_cleaning import SalesCleaner
from src.modules.get2_engineering import FeatureEngineering
from src.modules.get3_preparing import DataPreparation
from src.modules.get4_select_model import ModelTraining

# Definir la ruta de los datos
data_path = config["data"]["raw_path"]
results_path = config["data"]["results_path"]
submission_filename = config["output"]["submission_filename"]
full_predictions_filename = config["output"]["full_predictions_filename"]

# Cargar los datasets
sales_train = pd.read_csv(os.path.join(data_path, "sales_train.csv"))
shops = pd.read_csv(os.path.join(data_path, "shops.csv"))
items = pd.read_csv(os.path.join(data_path, "items.csv"))
item_categories = pd.read_csv(os.path.join(data_path, "item_categories.csv"))
test = pd.read_csv(os.path.join(data_path, "test.csv"))

# Limpieza de datos
sales_train = SalesCleaner(sales_train).execute_transformations()
shops = DataProcessor.translate_column(shops, "shop_name")
item_categories = DataProcessor.translate_column(item_categories, "item_category_name")

# Fusionar los datos
train = sales_train.merge(shops, on="shop_id", how="left")
train = train.merge(items, on="item_id", how="left")
train = train.merge(item_categories, on="item_category_id", how="left")

# Detectar y eliminar valores atípicos
outlier_params = config["preprocessing"]["detect_outliers"]
summary_df_ip, filtered_data_ip = detect_outliers(
    train,
    column=outlier_params["column"],
    iqr_factor=outlier_params["iqr_factor"],
    z_threshold=outlier_params["z_threshold"],
    lower_percentile=outlier_params["lower_percentile"],
    upper_percentile=outlier_params["upper_percentile"]
)
train = filtered_data_ip["Z-Score"].reset_index(drop=True)
train["date"] = pd.to_datetime(train["date"], format="%Y-%m-%d")

# Aplicar ingeniería de características
t = train.copy()
train_model = FeatureEngineering(t, target="item_cnt_month").apply_feature_engineering()

# Preparación de datos
data_prep = DataPreparation(train_model, target="item_cnt_month")
X_train, X_val, y_train, y_val = data_prep.prepare_data()

# Entrenar el modelo
model_trainer = ModelTraining(X_train, X_val, y_train, y_val, "random_forest")
full_data, best_model, train_rmse, val_rmse = model_trainer.train_and_evaluate_model()

# Aplicar el modelo a los datos de test
test_model = FeatureEngineering(test, target="item_cnt_month").apply_feature_engineering()
data_prep_test = DataPreparation(test_model, target="item_cnt_month")
X_test, _, _, _ = data_prep_test.prepare_data()

# Realizar predicciones
predictions = best_model.predict(X_test)

# Guardar las predicciones en test
submission = pd.DataFrame({"ID": test["ID"], "item_cnt_month": predictions})
submission.to_csv(os.path.join(results_path, submission_filename), index=False)

# Guardar las predicciones con los datos completos
full_data.to_csv(os.path.join(results_path, full_predictions_filename), index=False)

print(f"Proceso E2E completado. Predicciones guardadas en '{submission_filename}' y '{full_predictions_filename}'.")
