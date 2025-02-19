# data_utils.py
import pandas as pd
from deep_translator import GoogleTranslator

class DataProcessor:
    """
    Clase utilitaria para procesamiento de datos en diferentes DataFrames.
    """

    @staticmethod
    def completitud(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la cantidad de valores nulos y el porcentaje de completitud para cada variable en un DataFrame.
        
        Parámetros:
        df (pd.DataFrame): DataFrame a analizar.
        
        Retorna:
        pd.DataFrame: DataFrame con columnas ["variable", "num_nulos", "completitud"]
        """
        # Verificar que el input sea un DataFrame válido
        if not isinstance(df, pd.DataFrame):
            raise ValueError("El argumento df debe ser un pandas DataFrame.")
        
        if df.empty:
            raise ValueError("El DataFrame está vacío.")
        
        # Calcular valores nulos y porcentaje de completitud
        num_nulos = df.isnull().sum()
        total_filas = len(df)
        completitud = (1 - num_nulos / total_filas) * 100  # Porcentaje de datos no nulos
        
        # Crear DataFrame de salida
        df_completitud = pd.DataFrame({
            "variable": df.columns,
            "num_nulos": num_nulos.values,
            "completitud": completitud.values
        }).sort_values(by="num_nulos", ascending=False)
        
        return df_completitud

    @staticmethod
    def translate_column(df: pd.DataFrame, column_name: str, source_lang: str = "ru", target_lang: str = "es") -> pd.DataFrame:
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
        # Verificar que el input sea un DataFrame válido
        if not isinstance(df, pd.DataFrame):
            raise ValueError("El argumento df debe ser un pandas DataFrame.")
        
        if column_name not in df.columns:
            raise ValueError(f"La columna '{column_name}' no existe en el DataFrame.")
        
        if df[column_name].isnull().all():
            raise ValueError(f"La columna '{column_name}' no tiene valores válidos para traducir.")
        
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        
        def safe_translate(text):
            """ Traduce un texto, pero maneja errores específicos si el traductor falla. """
            if isinstance(text, str) and text.strip():  # Evita traducir valores vacíos o nulos
                try:
                    return translator.translate(text)
                except exceptions.NotValidPayload:
                    return text  # Entrada inválida para la API
                except exceptions.TooManyRequests:
                    return text  # Límite de solicitudes alcanzado
                except exceptions.ElementNotFoundInAPIResponse:
                    return text  # Elemento no encontrado en respuesta API
                except exceptions.RequestError:
                    return text  # Error en la solicitud
            return text
        
        # Crear nueva columna con la traducción
        translated_col = column_name + "_" + target_lang
        df[translated_col] = df[column_name].astype(str).apply(safe_translate)
        
        return df

    # @staticmethod
    # def translate_column(df, column_name, source_lang="ru", target_lang="es"):
    #     """
    #     Traduce los valores de una columna en un DataFrame utilizando Google Translator.

    #     Parámetros:
    #     df (pd.DataFrame): DataFrame a procesar.
    #     column_name (str): Nombre de la columna a traducir.
    #     source_lang (str): Idioma de origen (por defecto, ruso 'ru').
    #     target_lang (str): Idioma destino (por defecto, español 'es').

    #     Retorna:
    #     pd.DataFrame: DataFrame con una nueva columna traducida.
    #     """
    #     translator = GoogleTranslator(source=source_lang, target=target_lang)

    #     def safe_translate(text):
    #         """ Traduce un texto, pero maneja errores si el traductor falla. """
    #         if isinstance(text, str) and text.strip():  # Evita traducir valores vacíos o nulos
    #             try:
    #                 return translator.translate(text)
    #             except Exception:
    #                 return text  # En caso de error, devuelve el texto original
    #         return text

    #     # Crear nueva columna con la traducción
    #     translated_col = column_name +"_"+target_lang
    #     df[translated_col] = df[column_name].apply(safe_translate)

    #     return df
