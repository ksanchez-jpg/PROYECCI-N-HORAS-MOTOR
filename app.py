import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="📈 Proyección de Horas Motor", page_icon="⛽", layout="wide")
st.title("📊 Proyección de Horas Motor 2026")

# Subir archivo CSV
uploaded_file = st.file_uploader("📂 Sube el archivo CSV con los datos históricos", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Datos cargados:")
    st.dataframe(df)

    # Verificar que las columnas necesarias existan
    if "Mes" in df.columns and "Horas_Motor" in df.columns:
        # Convertir Mes a índice numérico si no lo es
        if not np.issubdtype(df["Mes"].dtype, np.number):
            df["Mes_Num"] = np.arange(1, len(df) + 1)
        else:
            df["Mes_Num"] = df["Mes"]

        # Entrenar regresión lineal
        X = df[["Mes_Num"]]
        y = df["Horas_Motor"]
        model = LinearRegression()
        model.fit(X, y)

        # Proyección hasta diciembre 2026
        future_months = np.arange(1, len(df) + (16 - len(df)) + 1)  # Ajustar horizonte según datos
        future_pred = model.predict(future_months.reshape(-1, 1))

        # Construir DataFrame proyectado
        df_future = pd.DataFrame({
            "Mes_Num": future_months,
            "Horas_Motor_Pred": future_pred
        })

        st.subheader("📈 Proyección de Horas Motor hasta 2026:")
        st.line_chart(df_future.set_index("Mes_Num"))

        st.subheader("📥 Descargar proyección")
        csv = df_future.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", csv, "proyeccion_horas_motor.csv", "text/csv")

    else:
        st.error("❌ El archivo debe contener las columnas 'Mes' y 'Horas_Motor'.")
