import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_distribution(df, column, title=None, clip_min=None, clip_max=None, figsize=(12, 5), color='royalblue'):
    """
    Grafica la distribución de una columna junto con su boxplot en una disposición horizontal (histograma a la izquierda, boxplot a la derecha).
    Muestra la media, mediana y cuartiles en la visualización.

    Parámetros:
    df (pd.DataFrame): DataFrame de datos.
    column (str): Nombre de la columna a graficar.
    title (str): Título opcional del gráfico.
    clip_min (float): Límite inferior opcional para recortar valores extremos.
    clip_max (float): Límite superior opcional para recortar valores extremos.
    figsize (tuple): Tamaño del gráfico.
    color (str): Color del histograma.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1]})

    # Recortar valores extremos si se especifican
    data = df[column]
    if clip_min is not None and clip_max is not None:
        data = data.clip(clip_min, clip_max)

    # Cálculo de estadísticas
    mean = data.mean()
    median = data.median()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    # Histograma con KDE (curva de densidad) a la izquierda
    sns.histplot(data, bins=50, kde=True, color=color, ax=axes[0])
    axes[0].axvline(mean, color="blue", linestyle="--", label=f"Media: {mean:.2f}")
    axes[0].axvline(median, color="red", linestyle="-.", label=f"Mediana: {median:.2f}")
    axes[0].axvline(Q1, color="green", linestyle=":", label=f"Q1: {Q1:.2f}")
    axes[0].axvline(Q3, color="purple", linestyle=":", label=f"Q3: {Q3:.2f}")

    axes[0].set_title(title if title else f"Distribución de {column}")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Gráfico de caja (Boxplot) a la derecha
    sns.boxplot(x=data, color=color, ax=axes[1])
    axes[1].set_xlabel(column)
    axes[1].set_title("Boxplot")

    plt.tight_layout()
    plt.show()

def detect_outliers(df, column, iqr_factor=1.5, z_threshold=3, lower_percentile=0.01, upper_percentile=0.99):
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
    """

    # Copia de los datos
    data = df[column].dropna()
    total_data = len(data)

    # MÉTODO 1: IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound_iqr = Q1 - (iqr_factor * IQR)
    upper_bound_iqr = Q3 + (iqr_factor * IQR)
    outliers_iqr = data[(data < lower_bound_iqr) | (data > upper_bound_iqr)]
    remaining_iqr = total_data - len(outliers_iqr)
    percentage_iqr = (len(outliers_iqr) / total_data) * 100
    filtered_iqr = df[~df[column].isin(outliers_iqr)]

    # MÉTODO 2: Z-Score
    z_scores = stats.zscore(data)
    outliers_z = data[np.abs(z_scores) > z_threshold]
    lower_bound_z = data.mean() - (z_threshold * data.std())
    upper_bound_z = data.mean() + (z_threshold * data.std())
    remaining_z = total_data - len(outliers_z)
    percentage_z = (len(outliers_z) / total_data) * 100
    filtered_z = df[~df[column].isin(outliers_z)]

    # MÉTODO 3: Percentiles
    lower_bound_perc = data.quantile(lower_percentile)
    upper_bound_perc = data.quantile(upper_percentile)
    outliers_perc = data[(data < lower_bound_perc) | (data > upper_bound_perc)]
    remaining_perc = total_data - len(outliers_perc)
    percentage_perc = (len(outliers_perc) / total_data) * 100
    filtered_perc = df[~df[column].isin(outliers_perc)]

    # Construcción del DataFrame de resumen
    summary_df = pd.DataFrame({
        "Método": ["IQR", "Z-Score", "Percentiles"],
        "Límite Inferior": [lower_bound_iqr, lower_bound_z, lower_bound_perc],
        "Límite Superior": [upper_bound_iqr, upper_bound_z, upper_bound_perc],
        "Datos Outliers": [len(outliers_iqr), len(outliers_z), len(outliers_perc)],
        "Datos Sin Outliers": [remaining_iqr, remaining_z, remaining_perc],
        "Porcentaje Outliers (%)": [percentage_iqr, percentage_z, percentage_perc]
    })

    # Diccionario con los DataFrames filtrados
    filtered_data = {
        "IQR": filtered_iqr,
        "Z-Score": filtered_z,
        "Percentiles": filtered_perc
    }

    return summary_df, filtered_data

def top_sellers(df, group_by, figsize=(12, 5), top_n=10):
    """
    Identifica los elementos más vendidos en un DataFrame.
    
    Parámetros:
    df (pd.DataFrame): DataFrame con datos de ventas.
    group_by (str): Columna de agrupación ('item_id', 'shop_id', etc.).
    top_n (int): Número de elementos a mostrar.
    
    Retorna:
    - DataFrame con los elementos más vendidos.
    """
    if group_by not in df.columns:
        raise ValueError(f"La columna '{group_by}' no existe en el DataFrame.")
    
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
    
    return top_selling_df
