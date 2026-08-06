# Deteksi Curah Hujan (Damin)

Aplikasi **deteksi/prediksi curah hujan** berbasis Machine Learning — Tugas Besar.
Model dilatih dengan **Random Forest** + penanganan data tidak seimbang (**SMOTE**), disajikan sebagai web app interaktif dengan **Streamlit**.

## Fitur

- Prediksi curah hujan dari input fitur
- Penanganan data imbalance dengan SMOTE
- Visualisasi data (matplotlib, seaborn)
- Web app interaktif (Streamlit)

## Tech Stack

- Python
- Streamlit
- scikit-learn (Random Forest, SMOTE)
- pandas, numpy, matplotlib, seaborn
- imbalanced-learn

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur Project

| File | Keterangan |
|------|------------|
| `app.py` | Web app Streamlit (load model & prediksi) |
| `Tubes_Damin.ipynb` | Notebook training & eksperimen |
| `Tubes_Damin.pkl`, `scaler.pkl`, `label_encoder_*.pkl` | Model & objek pendukung (pickle) |
| `Dataset/` | Data latih |
