import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Configuración de la aplicación
st.set_page_config(page_title="Predicción de Ventas", layout="wide")

# Título
st.title("Dashboard de Predicción de Ventas")

# Carga de datos
st.sidebar.header("Subir archivo CSV")
uploaded_file = st.sidebar.file_uploader("Carga un archivo CSV", type=["csv"])

if uploaded_file:
    file_name = uploaded_file.name
    st.sidebar.write(f"Archivo cargado: {file_name}")
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Archivo cargado correctamente")
    
    # Vista previa de los datos
    st.subheader("Vista previa del DataFrame")
    st.dataframe(df.head())
    
    # Estadísticas descriptivas
    st.subheader("Estadísticas descriptivas")
    st.write(df.describe())
    
    # Filtrado por Train/Test
    flag_option = st.sidebar.selectbox("Seleccionar dataset:", ["Train", "Test", "Todos"])
    
    if flag_option != "Todos":
        df = df[df['flag'] == flag_option]
    
    # Gráfico de comparación entre predicción y realidad
    st.subheader("Comparación de ventas reales vs predichas")
    fig = px.scatter(df, x='item_cnt_month', y='y_predict', color='flag',
                     title='Ventas reales vs Predichas', labels={"item_cnt_month": "Ventas Reales", "y_predict": "Ventas Predichas"})
    st.plotly_chart(fig)
    
    # Cálculo de métricas de error
    mae = mean_absolute_error(df['item_cnt_month'], df['y_predict'])
    rmse = np.sqrt(mean_squared_error(df['item_cnt_month'], df['y_predict']))
    r2 = r2_score(df['item_cnt_month'], df['y_predict'])
    
    # Mostrar métricas
    st.subheader("Métricas de evaluación del modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", round(mae, 2))
    col2.metric("RMSE", round(rmse, 2))
    col3.metric("R²", round(r2, 2))
    
    # Filtro por tienda y producto
    st.sidebar.subheader("Filtrar por tienda y producto")
    tiendas = df['shop_id'].unique()
    productos = df['item_id'].unique()
    
    tienda_seleccionada = st.sidebar.selectbox("Seleccionar Tienda", tiendas)
    producto_seleccionado = st.sidebar.selectbox("Seleccionar Producto", productos)
    
    df_filtrado = df[(df['shop_id'] == tienda_seleccionada) & (df['item_id'] == producto_seleccionado)]
    
    if not df_filtrado.empty:
        st.subheader(f"Datos para Tienda {tienda_seleccionada} y Producto {producto_seleccionado}")
        st.dataframe(df_filtrado)
    else:
        st.warning("No hay datos para la selección actual.")
