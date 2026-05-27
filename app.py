import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Pronóstico de Demanda - XGBoost",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado para mejorar la interfaz de usuario (UI Premium)
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_ml_model():
    """Carga de manera segura el modelo entrenado de XGBoost."""
    model_path = "modelo_xgboost_ventas.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, None
        except Exception as e:
            return None, f"Error al deserializar el archivo pickle: {str(e)}"
    else:
        return None, f"No se encontró el archivo '{model_path}' en el directorio actual. Por favor súbelo o colócalo en la misma carpeta de este script."

model, model_error = load_ml_model()

# --- DEFINICIÓN DE VARIABLES ---
# Lista de variables que el usuario declaró para el modelo
DEFAULT_FEATURES = [
    'Vendedor', 'Fec Factura', 'Nombre del solicitante', 'Ventas totales', 
    'Descripción de material (producto)', 'Cantidad facturada', 'Ofc. Venta', 
    'Período contable', 'Pobl. Destino', 'Canal de Distribución', 
    'Hora facturación', 'Departamento', 'Mes'
]

# Intentamos extraer las características con las que se entrenó el modelo
features_to_use = DEFAULT_FEATURES
if model is not None:
    # Verificamos si el modelo expone las variables esperadas
    if hasattr(model, 'feature_names_in_'):
        features_to_use = list(model.feature_names_in_)
    elif hasattr(model, 'feature_names'): # XGBoost nativo
        if model.feature_names is not None:
            features_to_use = model.feature_names

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/583/583279.png", width=80)
    st.title("Demand Forecast App")
    st.markdown("### Modelo Predictivo XGBoost")
    st.markdown("---")
    
    # Indicador de estado del modelo
    if model is not None:
        st.success("✅ Modelo cargado correctamente")
        # Mostrar metadatos del modelo si es posible
        with st.expander("Detalles del Modelo"):
            st.write(f"**Tipo:** {type(model).__name__}")
            st.write(f"**Variables esperadas ({len(features_to_use)}):**")
            st.caption(", ".join(features_to_use))
    else:
        st.error("❌ Modelo no disponible")
        st.info("Sube el archivo 'modelo_xgboost_ventas.pkl' en el mismo directorio de esta app para habilitar las predicciones.")
        if model_error:
            st.error(model_error)

    st.markdown("---")
    st.markdown("### Configuración de Datos")
    # Opción para forzar categorías si XGBoost nativo es usado
    enable_categorical = st.checkbox("Habilitar soporte nativo de categóricos", value=True, 
                                     help="Habilita que XGBoost interprete las columnas de texto como categorías nativas.")

# --- CUERPO PRINCIPAL ---
st.title("📈 Sistema Inteligente de Pronóstico de Demanda")
st.markdown("Optimiza tu cadena de suministro, inventario y ventas utilizando algoritmos predictivos basados en Machine Learning (XGBoost).")

# Separador visual
st.markdown("<br>", unsafe_allow_html=True)

# Si el modelo no está cargado, mostramos una pantalla de advertencia y detenemos la app temporalmente
if model is None:
    st.warning("⚠️ Esperando por la carga correcta del archivo de modelo `.pkl` en el servidor...")
    st.stop()

# Dividimos la aplicación en Pestañas (Tabs) para mejorar la usabilidad
tab_manual, tab_masivo, tab_analisis = st.tabs([
    "🎯 Predicción Individual", 
    "📂 Predicción Masiva", 
    "📊 Análisis de Pronósticos"
])

# --- PESTAÑA 1: PREDICCIÓN INDIVIDUAL (FORMULARIO MANUAL) ---
with tab_manual:
    st.header("Entrada de Datos Manual")
    st.markdown("Rellena los campos a continuación para estimar el valor objetivo (Demanda esperada).")
    
    # Creamos un formulario dinámico basado en las variables del modelo
    with st.form("manual_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        form_data = {}
        
        # Iteramos dinámicamente sobre las variables requeridas
        for i, feat in enumerate(features_to_use):
            # Distribuimos equitativamente entre las 3 columnas
            target_col = col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3)
            
            with target_col:
                # Modificamos los controles según el nombre o naturaleza esperada de la variable
                if 'Mes' in feat:
                    form_data[feat] = st.slider("Mes de Pronóstico", 1, 12, datetime.now().month, key=feat)
                elif 'Fec' in feat or 'Fecha' in feat:
                    selected_date = st.date_input("Fecha Factura", datetime.now(), key=feat)
                    # Convertir a string o unix timestamp dependiendo de lo que espere el modelo
                    form_data[feat] = selected_date.strftime("%Y-%m-%d")
                elif 'Cantidad' in feat or 'Ventas' in feat or 'totales' in feat:
                    form_data[feat] = st.number_input(f"{feat}", min_value=0.0, value=100.0, step=1.0, key=feat)
                elif 'Período' in feat or 'contable' in feat:
                    form_data[feat] = st.number_input(f"{feat} (Año/Período)", min_value=2000, max_value=2100, value=2026, key=feat)
                elif 'Hora' in feat:
                    # Input de hora simple o numérico
                    form_data[feat] = st.number_input(f"{feat} (Hora 0-23)", min_value=0, max_value=23, value=12, key=feat)
                else:
                    # Por defecto, entrada de texto para categóricos
                    form_data[feat] = st.text_input(f"{feat}", value="Ejemplo", key=feat)

        # Botón de envío del formulario
        submit_button = st.form_submit_with_name("Realizar Predicción")

    if submit_button:
        # Convertir entrada del formulario a DataFrame
        input_df = pd.DataFrame([form_data])
        
        # Procesamiento básico si se habilitan categorías
        if enable_categorical:
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    input_df[col] = input_df[col].astype('category')
        
        # Mostrar el vector de entrada antes de predecir (para depuración)
        with st.expander("Ver vector de características procesado"):
            st.dataframe(input_df)
            
        try:
            # Predicción
            prediction = model.predict(input_df)[0]
            
            # Formatear el resultado estéticamente
            st.markdown("---")
            st.subheader("🎉 Resultado del Pronóstico")
            
            # Usamos columnas para mostrar un indicador destacado
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric(
                    label="Valor Predicho (Demanda/Ventas)", 
                    value=f"{prediction:,.2f}",
                    delta="XGBoost Estimator"
                )
            with res_col2:
                st.info(
                    "💡 **Sugerencia:** Este resultado corresponde al escenario configurado en el panel de entrada. "
                    "Puedes ajustar las variables (por ejemplo, subir ventas estimadas, cambiar mes de temporada, o modificar el vendedor) "
                    "y volver a calcular para simular diferentes escenarios de negocio."
                )
                
        except Exception as e:
            st.error(f"Error al realizar la predicción: {str(e)}")
            st.warning(
                "Este error suele ocurrir si el modelo se entrenó esperando tipos de datos específicos (numéricos en lugar de texto). "
                "Intenta verificar el preprocesamiento de variables categóricas o utiliza la predicción masiva cargando un archivo con tus datos estructurados."
            )

# --- PESTAÑA 2: PREDICCIÓN MASIVA (SUBIR ARCHIVO) ---
with tab_masivo:
    st.header("Predicciones por Lote (Carga de Archivos)")
    st.markdown(
        "Sube un archivo de Excel o CSV que contenga las columnas requeridas para obtener predicciones automatizadas sobre múltiples registros."
    )
    
    # Sección para descargar plantilla de ejemplo
    st.markdown("### 📥 Descargar Plantilla")
    # Crear un DataFrame vacío con las columnas requeridas
    template_df = pd.DataFrame(columns=features_to_use)
    # Crear ejemplo de fila para guiar al usuario
    example_row = {}
    for col in features_to_use:
        if 'Mes' in col: example_row[col] = 5
        elif 'Fec' in col: example_row[col] = "2026-05-15"
        elif 'Cantidad' in col or 'Ventas' in col: example_row[col] = 150.0
        elif 'Período' in col: example_row[col] = 2026
        elif 'Hora' in col: example_row[col] = 14
        else: example_row[col] = "Valor_Ejemplo"
    
    template_df = pd.concat([template_df, pd.DataFrame([example_row])], ignore_index=True)
    
    # Botón para descargar la plantilla de datos
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar Plantilla CSV",
        data=csv_template,
        file_name="plantilla_pronosticos_ventas.csv",
        mime="text/csv",
        help="Utiliza este archivo como estructura base llenándolo con tus propios datos."
    )
    
    st.markdown("---")
    st.markdown("### 📤 Cargar tus Datos")
    
    uploaded_file = st.file_uploader("Elige tu archivo CSV o Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Identificar extensión y cargar datos
            if uploaded_file.name.endswith('.csv'):
                data_df = pd.read_csv(uploaded_file)
            else:
                data_df = pd.read_excel(uploaded_file)
                
            st.success("✅ Archivo cargado correctamente.")
            
            # Previsualización de los datos originales
            st.subheader("Previsualización de Datos Cargados")
            st.dataframe(data_df.head(10))
            
            # Validación de columnas obligatorias
            missing_cols = [col for col in features_to_use if col not in data_df.columns]
            
            if len(missing_cols) > 0:
                st.error(f"Faltan las siguientes columnas requeridas en tu archivo: {missing_cols}")
                st.info("Por favor, ajusta tu archivo para que coincida exactamente con la lista de columnas de la plantilla.")
            else:
                st.success("¡Estructura de columnas validada con éxito!")
                
                # Botón para activar el pipeline de predicción
                if st.button("Ejecutar Pronósticos en Lote"):
                    with st.spinner("Procesando datos y generando predicciones..."):
                        # Filtrar y ordenar las columnas exactamente como el modelo las espera
                        prediction_input = data_df[features_to_use].copy()
                        
                        # Conversión a categóricas si está habilitado
                        if enable_categorical:
                            for col in prediction_input.columns:
                                if prediction_input[col].dtype == 'object':
                                    prediction_input[col] = prediction_input[col].astype('category')
                        
                        # Ejecutar predicción masiva
                        preds = model.predict(prediction_input)
                        
                        # Agregar las predicciones al DataFrame original
                        result_df = data_df.copy()
                        result_df['Pronóstico_Demanda'] = preds
                        
                        # Guardar el resultado en la sesión de streamlit para la pestaña 3
                        st.session_state['predicted_data'] = result_df
                        
                        # Mostrar resultados obtenidos
                        st.subheader("Resultados Calculados")
                        st.dataframe(result_df.head(10))
                        
                        # Botones para descargar resultados
                        csv_results = result_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="📥 Descargar Reporte en CSV",
                            data=csv_results,
                            file_name=f"reporte_pronosticos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.balloons()
                        
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {str(e)}")

# --- PESTAÑA 3: ANÁLISIS DE PRONÓSTICOS ---
with tab_analisis:
    st.header("Panel de Análisis e Insights Gráficos")
    
    # Comprobar si existen datos predichos en sesión
    if 'predicted_data' not in st.session_state:
        st.info("💡 Por favor, realiza una **Predicción Masiva** en la pestaña anterior para desbloquear el análisis visual de los datos.")
    else:
        df = st.session_state['predicted_data']
        
        # Ofrecer métricas agregadas globales
        total_predicted = df['Pronóstico_Demanda'].sum()
        avg_predicted = df['Pronóstico_Demanda'].mean()
        max_predicted = df['Pronóstico_Demanda'].max()
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Suma Total Pronosticada", f"{total_predicted:,.2f}")
        with col_m2:
            st.metric("Demanda Promedio por Fila", f"{avg_predicted:,.2f}")
        with col_m3:
            st.metric("Demanda Máxima Única", f"{max_predicted:,.2f}")
            
        st.markdown("---")
        
        # Gráficas dinámicas según variables existentes
        st.subheader("Visualizaciones Analíticas")
        
        # Graficar Demanda por Mes
        if 'Mes' in df.columns:
            st.markdown("#### Pronóstico Mensual")
            monthly_data = df.groupby('Mes')['Pronóstico_Demanda'].sum().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=monthly_data, x='Mes', y='Pronóstico_Demanda', ax=ax, palette="Blues_d")
            ax.set_title("Demanda Estimada por Mes")
            ax.set_xlabel("Mes del Año")
            ax.set_ylabel("Demanda Pronosticada")
            st.pyplot(fig)
            
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            # Gráfica por Vendedor si existe
            vendedor_col = [c for c in df.columns if 'Vendedor' in c]
            if len(vendedor_col) > 0:
                v_col = vendedor_col[0]
                st.markdown(f"#### Top Vendedores por Demanda Estimada")
                vendedor_data = df.groupby(v_col)['Pronóstico_Demanda'].sum().reset_index()
                vendedor_data = vendedor_data.sort_values(by='Pronóstico_Demanda', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=vendedor_data, y=v_col, x='Pronóstico_Demanda', ax=ax, palette="viridis")
                ax.set_title("Demanda Esperada por Vendedor (Top 10)")
                st.pyplot(fig)
                
        with col_g2:
            # Gráfica por Producto / Descripción si existe
            prod_col = [c for c in df.columns if 'material' in c or 'producto' in c or 'Producto' in c]
            if len(prod_col) > 0:
                p_col = prod_col[0]
                st.markdown(f"#### Productos más Demandados (Pronosticado)")
                prod_data = df.groupby(p_col)['Pronóstico_Demanda'].sum().reset_index()
                prod_data = prod_data.sort_values(by='Pronóstico_Demanda', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=prod_data, y=p_col, x='Pronóstico_Demanda', ax=ax, palette="rocket")
                ax.set_title("Demanda Esperada por Producto (Top 10)")
                st.pyplot(fig)
                
        # Tabla resumen descargable organizada
        st.subheader("Tabla de Datos con Pronósticos Completos")
        st.dataframe(df)
