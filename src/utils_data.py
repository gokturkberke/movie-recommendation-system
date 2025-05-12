# src/utils_data.py
import streamlit as st
import pandas as pd
import os
import re # clean_text için eklendi
from surprise import dump # load_trained_surprise_model için eklendi

@st.cache_data
def load_cleaned_data(filename, data_path='cleaned_data'):
    """Loads a CSV file from the cleaned_data directory with error handling."""
    file_path = os.path.join(data_path, filename)
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            st.warning(f"Warning: {filename} is empty.")
        return df
    except FileNotFoundError:
        st.error(f"ERROR: File not found: {file_path}. Please ensure the file exists.")
        return pd.DataFrame() # Return empty DataFrame on error
    except pd.errors.EmptyDataError:
        st.error(f"ERROR: No data: {filename} is empty or corrupted.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        st.error(f"ERROR: An unexpected error occurred while loading {filename}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

@st.cache_data
def load_movies(data_path='cleaned_data'):
    return load_cleaned_data('movies_clean.csv', data_path)

@st.cache_data
def load_ratings(data_path='cleaned_data'):
    return load_cleaned_data('ratings_clean.csv', data_path)

@st.cache_resource # Model gibi kaynaklar için cache_resource daha uygun
def load_trained_surprise_model(model_filename="svd_trained_model.pkl"):
    # Bu fonksiyonun içindeki path'ler app.py'nin konumuna göreydi,
    # utils_data.py'nin konumuna göre düzeltilmesi gerekebilir.
    # Şimdilik app.py ile aynı src klasöründe olduğu için çalışacaktır.
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Bu utils_data.py'nin olduğu src klasörü olacak
    cleaned_data_dir = os.path.join(script_dir, '..', 'cleaned_data') # src'nin bir üstündeki cleaned_data
    model_path = os.path.join(cleaned_data_dir, model_filename)

    print(f"Önceden eğitilmiş Surprise modeli yükleniyor: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"HATA: Kayıtlı model dosyası bulunamadı: {model_path}. "
                 f"Lütfen önce train_save_model.py script'ini çalıştırın.")
        return None

    try:
        loaded_object = dump.load(model_path)
        model = loaded_object[1]
        print("Önceden eğitilmiş model başarıyla yüklendi.")
        return model
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {e}")
        return None

@st.cache_data
def load_tags(data_path='cleaned_data'):
    return load_cleaned_data('tags_clean.csv', data_path)

# Centralized text cleaning function (consistent with preprocess_dataset.py)
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    # Remove year in parentheses (e.g., "(1995)") if present
    text = re.sub(r'\s*\(\d{4}\)', '', text) # Yıl parantezlerini eşleştirmek için \( ve \)
# ...
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # \s zaten boşluk karakterlerini ifade eder
    # Normalize whitespace (replace multiple spaces with a single space and strip)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
