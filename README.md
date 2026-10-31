# Prediksi Tingkat Stres Mahasiswa (Damin)

Aplikasi **prediksi tingkat stres mahasiswa** selama pembelajaran daring berbasis Machine Learning — Tugas Besar mata kuliah Data Mining.
Model dilatih dengan **Random Forest** + penanganan data tidak seimbang (**SMOTE**), disajikan sebagai web app interaktif dengan **Streamlit**.

> Nama repo `DeteksiCurahHujan_Damin` adalah nama lama; isi proyek ini adalah prediksi tingkat stres, bukan curah hujan.

## Fitur

- Prediksi tingkat stres (Low, Medium, High) dari gender, usia, screen time, durasi tidur, dan aktivitas fisik
- Penanganan data imbalance dengan SMOTE
- Clustering pola hidup dengan PCA dan mutual information untuk melihat fitur paling berpengaruh
- Visualisasi data (matplotlib, seaborn)
- Web app interaktif (Streamlit)

## Hasil

- Akurasi data uji: 61,15%
- Rata-rata validasi silang 5-fold: 60,85%

## Tech Stack

- Python
- Streamlit
- scikit-learn (Random Forest, StandardScaler, LabelEncoder)
- imbalanced-learn (SMOTE)
- pandas, numpy, matplotlib, seaborn

## Cara Menjalankan

```bash
pip install streamlit scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
streamlit run app.py
```

## Struktur Project

| File | Keterangan |
|------|------------|
| `app.py` | Web app Streamlit (load model & prediksi) |
| `Tubes_Damin.ipynb` | Notebook training & eksperimen |
| `Tubes_Damin.pkl`, `scaler.pkl`, `label_encoder_*.pkl` | Model & objek pendukung (pickle) |
| `Dataset/` | Data latih (Student Mental Health Analysis) |

## Tim

- Raka Zeniusa Barron (220102070)
- Shofa Farhah Tanzilah (220102080)
- Arina Sabilana (220102016)
