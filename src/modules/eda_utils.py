"""
Módulo de análisis exploratorio de datos (EDA).

Este módulo proporciona funciones para:
- Graficar distribuciones de variables numéricas con histogramas y boxplots.
- Detectar valores atípicos en una columna mediante diferentes métodos.
- Identificar los productos o tiendas más vendidos.

Funciones:
    - plot_distribution: Genera histogramas y boxplots para visualizar distribuciones de datos.
    - detect_outliers: Detecta valores atípicos usando IQR, Z-Score y Percentiles.
    - top_sellers: Encuentra los productos o tiendas más vendidos y los grafica.

Excepciones:
    - KeyError: Se lanza cuando las columnas necesarias no están presentes en el DataFrame.
    - ValueError: Se lanza cuando los datos no son numéricos o hay inconsistencias.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from logs.logger_config import logger  # pylint: disable=C0413

def plot_distribution(df: pd.DataFrame, column: str, title: str = None, clip_min: float = None, clip_max: float = None, figsize: tuple = (12, 5), color: str = 'royalblue'):
    """
    Grafica la distribución de una columna junto con su boxplot.
    
    Parámetros:
    df (pd.DataFrame): DataFrame de datos.
    column (str): Nombre de la columna a graficar.
    title (str): Título opcional del gráfico.
    clip_min (float): Límite inferior opcional para recortar valores extremos.
    clip_max (float): Límite superior opcional para recortar valores extremos.
    figsize (tuple): Tamaño del gráfico.
    color (str): Color del histograma.
    
    Excepciones:
    KeyError: Si la columna no existe en el DataFrame.
    """
    if column not in df.columns:
        logger.error("La columna '%s' no existe en el DataFrame.", column)
        raise KeyError(f"La columna '{column}' no existe en el DataFrame.")

    logger.info("Generando distribución de la columna '%s'...",column)
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1]})

    # Recortar valores extremos si se especifican
    data = df[column]
    if clip_min is not None and clip_max is not None:
        data = data.clip(clip_min, clip_max)

    # Cálculo de estadísticas
    mean = data.mean()
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    # Histograma con KDE (curva de densidad)
    sns.histplot(data, bins=50, kde=True, color=color, ax=axes[0])
    axes[0].axvline(mean, color="blue", linestyle="--", label=f"Media: {mean:.2f}")
    axes[0].axvline(median, color="red", linestyle="-.", label=f"Mediana: {median:.2f}")
    axes[0].axvline(q1, color="green", linestyle=":", label=f"Q1: {q1:.2f}")
    axes[0].axvline(q3, color="purple", linestyle=":", label=f"Q3: {q3:.2f}")

    axes[0].set_title(title if title else f"Distribución de {column}")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Gráfico de caja (Boxplot)
    sns.boxplot(x=data, color=color, ax=axes[1])
    axes[1].set_xlabel(column)
    axes[1].set_title("Boxplot")

    plt.tight_layout()
    plt.show()
    logger.info("Distribución de '%s' generada correctamente.",column)

def detect_outliers(df: pd.DataFrame, column: str, iqr_factor: float = 1.5, z_threshold: float = 3, lower_percentile: float = 0.01, upper_percentile: float = 0.99):
    """
    Detecta valores atípicos en una columna utilizando tres métodos: IQR, Z-Score y Percentiles.
    Retorna un DataFrame resumen comparativo y un diccionario con los DataFrames filtrados.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    column (str): Nombre de la columna a analizar.
    iqr_factor (float): Factor para determinar el umbral de outliers con IQR (default = 1.5).
    z_threshold (float): Umbral para considerar valores atípicos con Z-Score (default = 3).
    lower_percentile (float): Percentil inferior para detección de outliers (default = 1%).
    upper_percentile (float): Percentil superior para detección de outliers (default = 99%).

    Retorna:
    - summary_df (pd.DataFrame): Resumen de los métodos de detección de outliers.
    - filtered_data (dict): Diccionario con los DataFrames sin outliers según cada método.

    Excepciones:
    KeyError: Si la columna no existe en el DataFrame.
    ValueError: Si la columna no contiene datos numéricos.
    """
    if column not in df.columns:
        logger.error("La columna '%s' no existe en el DataFrame.", column)
        raise KeyError(f"La columna '{column}' no existe en el DataFrame.")
    if not np.issubdtype(df[column].dtype, np.number):
        logger.error("La columna '%s' no contiene datos numéricos.", column)
        raise ValueError(f"La columna '{column}' no contiene datos numéricos.")

    logger.info("Detectando outliers en la columna '%s'...",column)
    data = df[column].dropna()
    total_data = len(data)

    # MÉTODO 1: IQR
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound_iqr = q1 - (iqr_factor * iqr)
    upper_bound_iqr = q3 + (iqr_factor * iqr)
    outliers_iqr = data[(data < lower_bound_iqr) | (data > upper_bound_iqr)]
    filtered_iqr = df[~df[column].isin(outliers_iqr)]

    # MÉTODO 2: Z-Score
    z_scores = stats.zscore(data)
    outliers_z = data[np.abs(z_scores) > z_threshold]
    filtered_z = df[~df[column].isin(outliers_z)]

    # MÉTODO 3: Percentiles
    lower_bound_perc = data.quantile(lower_percentile)
    upper_bound_perc = data.quantile(upper_percentile)
    outliers_perc = data[(data < lower_bound_perc) | (data > upper_bound_perc)]
    filtered_perc = df[~df[column].isin(outliers_perc)]

    # Construcción del DataFrame de resumen
    summary_df = pd.DataFrame({
        "Método": ["IQR", "Z-Score", "Percentiles"],
        "Límite Inferior": [lower_bound_iqr, data.mean() - (z_threshold * data.std()), lower_bound_perc],
        "Límite Superior": [upper_bound_iqr, data.mean() + (z_threshold * data.std()), upper_bound_perc],
        "Datos Outliers": [len(outliers_iqr), len(outliers_z), len(outliers_perc)],
        "Porcentaje Outliers (%)": [(len(outliers_iqr) / total_data) * 100, (len(outliers_z) / total_data) * 100, (len(outliers_perc) / total_data) * 100]
    })

    # Diccionario con los DataFrames filtrados
    filtered_data = {
        "IQR": filtered_iqr,
        "Z-Score": filtered_z,
        "Percentiles": filtered_perc
    }

    logger.info("Detección de outliers en '%s' completada.",column)
    return summary_df, filtered_data

def top_sellers(df: pd.DataFrame, group_by: str, figsize: tuple = (12, 5), top_n: int = 10) -> pd.DataFrame:
    """
    Identifica los elementos más vendidos en un DataFrame.
    
    Parámetros:
    df (pd.DataFrame): DataFrame con datos de ventas.
    group_by (str): Columna de agrupación ('item_id', 'shop_id', etc.).
    top_n (int): Número de elementos a mostrar.
    
    Retorna:
    pd.DataFrame: DataFrame con los elementos más vendidos.
    
    Excepciones:
    KeyError: Si la columna de agrupación no existe en el DataFrame.
    """
    if group_by not in df.columns:
        logger.error("La columna '%s' no existe en el DataFrame.", group_by)
        raise KeyError(f"La columna '{group_by}' no existe en el DataFrame.")

    logger.info("Identificando los %d elementos más vendidos por '%s'...", top_n, group_by)
    top_selling_df = df.groupby(group_by)["item_cnt_day"].sum().reset_index()
    top_selling_df = top_selling_df.sort_values(by="item_cnt_day", ascending=False).head(top_n)

    plt.figure(figsize=figsize)
    sns.barplot(data=top_selling_df, x=group_by, y="item_cnt_day", palette="tab10")
    plt.title(f"Top {top_n} más vendidos por '{group_by}'")
    plt.xlabel(group_by.replace("_", " ").title())
    plt.ylabel("Cantidad de productos vendidos")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--")
    plt.show()

    logger.info("Gráfico de los top vendedores generado correctamente.")
    return top_selling_df
