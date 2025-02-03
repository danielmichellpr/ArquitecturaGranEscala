# data_utils.py
import pandas as pd
from deep_translator import GoogleTranslator

class DataProcessor:
    """
    Clase utilitaria para procesamiento de datos en diferentes DataFrames.
    """

    @staticmethod
    def completitud(df):
        """
        Calcula la cantidad de valores nulos y el porcentaje de completitud para cada variable en un DataFrame.
        
        Parámetros:
        df (pd.DataFrame): DataFrame a analizar.
        
        Retorna:
        pd.DataFrame: DataFrame con columnas ["variable", "num_nulos", "completitud"]
        """
        # Calcular valores nulos y porcentaje de completitud
        num_nulos = df.isnull().sum()
        total_filas = len(df)
        completitud = (1 - num_nulos / total_filas) * 100  # Porcentaje de datos no nulos
        
        # Crear DataFrame de salida
        df_completitud = pd.DataFrame({
            "variable": df.columns,
            "num_nulos": num_nulos.values,
            "completitud": completitud.values
        })        
        return df_completitud.sort_values(by="num_nulos", ascending=False)

    @staticmethod
    def translate_column(df, column_name, source_lang="ru", target_lang="es"):
        """
        Traduce los valores de una columna en un DataFrame utilizando Google Translator.

        Parámetros:
        df (pd.DataFrame): DataFrame a procesar.
        column_name (str): Nombre de la columna a traducir.
        source_lang (str): Idioma de origen (por defecto, ruso 'ru').
        target_lang (str): Idioma destino (por defecto, español 'es').

        Retorna:
        pd.DataFrame: DataFrame con una nueva columna traducida.
        """
        translator = GoogleTranslator(source=source_lang, target=target_lang)

        def safe_translate(text):
            """ Traduce un texto, pero maneja errores si el traductor falla. """
            if isinstance(text, str) and text.strip():  # Evita traducir valores vacíos o nulos
                try:
                    return translator.translate(text)
                except Exception:
                    return text  # En caso de error, devuelve el texto original
            return text

        # Crear nueva columna con la traducción
        translated_col = column_name +"_"+target_lang
        df[translated_col] = df[column_name].apply(safe_translate)

        return df
