import streamlit as st
import joblib
import pickle
import numpy as np
import pytz
from datetime import datetime
import pandas as pd

import psycopg2
# Fetch variables
USER = "postgres.czmstzemfgnnkbqrrjwg" #os.getenv("user")
PASSWORD = "ujh6534dgd5e5"# os.getenv("password")
HOST = "aws-1-us-west-2.pooler.supabase.com" #os.getenv("host")
PORT = "6543" #os.getenv("port")
DBNAME = "postgres" #os.getenv("dbname")

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# Selección de zona horaria
timezones = ['UTC', 'America/New_York', 'America/Los_Angeles', 'Europe/London', 'Europe/Paris', 'Asia/Tokyo', 'America/Mexico_City']
selected_timezone = st.selectbox("Selecciona tu zona horaria:", timezones)
tz = pytz.timezone(selected_timezone)
# Connect to the database
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current Time:", result)
    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

except Exception as e:
    st.write(str(e))



# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'models/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

# Inicializar log de predicciones
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    st.write(result)
    
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Estandarizar
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")
        
        # Mostrar todas las probabilidades
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")
        
        # Guardar en log
        timestamp = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')
        log_entry = {
            'Fecha y Hora': timestamp,
            'Longitud del Sépalo': sepal_length,
            'Ancho del Sépalo': sepal_width,
            'Longitud del Pétalo': petal_length,
            'Ancho del Pétalo': petal_width,
            'Especie Predicha': predicted_species,
            'Confianza': f"{max(probabilities):.1%}",
            'Probabilidades': {species: f"{prob:.1%}" for species, prob in zip(target_names, probabilities)}
        }
        st.session_state.prediction_log.append(log_entry)
