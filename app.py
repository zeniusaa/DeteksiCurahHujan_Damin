# =======================================
#  IMPORT LIBRARY
# =======================================
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# =======================================
#  LOAD MODEL DAN OBJEK PENDUKUNG
# =======================================
pickle_in = open('Tubes_Damin.pkl', 'rb')
model = pickle.load(pickle_in)
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_stress = pickle.load(open('label_encoder_stress.pkl', 'rb'))
le_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))

# =======================================
#  FUNGSI PREDIKSI STRES
# =======================================
def predict_stress(gender, age, screen_time, sleep_duration, physical_activity):
    df_input = pd.DataFrame({
        'Gender': [le_gender.transform([gender])[0]],
        'Age': [age],
        'Screen Time (hrs/day)': [screen_time],
        'Sleep Duration (hrs)': [sleep_duration],
        'Physical Activity (hrs/week)': [physical_activity],
        'Screen Time (log)': [np.log1p(screen_time)],
        'Physical Activity (sqrt)': [np.sqrt(physical_activity)]
    })

    X_scaled = scaler.transform(df_input)
    pred = model.predict(X_scaled)
    pred_label = le_stress.inverse_transform(pred)[0]
    return pred_label

# =======================================
#  APLIKASI STREAMLIT UTAMA
# =======================================
def main():
    st.set_page_config(page_title="Student Stress Predictor", layout="centered")
    st.title("Student Stress Level Classification App")

    st.markdown("""
    <div style='background-color:#2C6E91; padding:10px; border-radius:10px'>
    <h3 style='color:white; text-align:center'>Cek Tingkat Stres Anda Selama Pembelajaran Online</h3>
    </div>
    """, unsafe_allow_html=True)

    st.header("Input Data Mahasiswa")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Usia", 15, 25, 20)
    screen_time = st.slider("Durasi Screen Time (jam/hari)", 1.0, 12.0, 6.0)
    sleep_duration = st.slider("Durasi Tidur (jam)", 3.0, 10.0, 6.5)
    physical_activity = st.slider("Aktivitas Fisik (jam/minggu)", 0.0, 10.0, 3.0)

    if st.button("Prediksi Tingkat Stres"):
        result = predict_stress(gender, age, screen_time, sleep_duration, physical_activity)
        st.success(f"Prediksi Tingkat Stres Anda: {result}")

    st.markdown("---")
    st.header("Akurasi Model")

    # Hitung akurasi berdasarkan test set statis (opsional bisa diubah)
    df = pd.read_csv("Dataset/Student Mental Health Analysis.csv")
    df.drop(columns=['Name'], inplace=True)
    df['Gender'] = df['Gender'].str.strip().str.capitalize()
    df['Stress Level'] = df['Stress Level'].str.strip().str.capitalize()
    df['Gender'] = le_gender.transform(df['Gender'])
    df['Stress Level'] = le_stress.transform(df['Stress Level'])
    df['Screen Time (log)'] = np.log1p(df['Screen Time (hrs/day)'])
    df['Physical Activity (sqrt)'] = np.sqrt(df['Physical Activity (hrs/week)'])

    X = df.drop("Stress Level", axis=1)
    y = df["Stress Level"]
    X_scaled = scaler.transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi Random Forest", value=f"{acc*100:.2f}%")

    st.caption("Model menggunakan RandomForestClassifier dengan preprocessing StandardScaler, LabelEncoder, dan SMOTE untuk balancing.")

#  Eksekusi utama
if __name__ == '__main__':
    main()
