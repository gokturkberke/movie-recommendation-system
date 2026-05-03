import os
import sys
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise import dump

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLEANED_DATA_DIR, SVD_MODEL_PATH  # noqa: E402

CLEANED_DATA_PATH = CLEANED_DATA_DIR
MODEL_SAVE_PATH = SVD_MODEL_PATH

def load_ratings_data_for_training(data_path):
    ratings_file_path = os.path.join(data_path, 'ratings_clean.csv')
    print(f"Temizlenmiş reyting verisi yükleniyor: {ratings_file_path}")
    try:
        return pd.read_csv(ratings_file_path)
    except FileNotFoundError:
        print(f"HATA: {ratings_file_path} bulunamadı. Lütfen preprocess_dataset.py script'ini çalıştırdığınızdan emin olun.")
        return None

def train_and_save_surprise_model(ratings_df, model_output_path):
    if ratings_df is None or ratings_df.empty:
        print("Reyting verisi boş veya yüklenemedi. Model eğitimi iptal edildi.")
        return

    print("Surprise veri seti hazırlanıyor...")
    reader = Reader(rating_scale=(0.5, 5.0)) # Reyting ölçeğinize göre ayarlayın
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

    print("Tam eğitim seti (full trainset) oluşturuluyor...")
    full_trainset = data.build_full_trainset() # Tüm veriyi kullanarak trainset oluştur

    print("SVD modeli eğitiliyor... (Bu işlem veri büyüklüğüne göre zaman alabilir)")
    algo = SVD() # SVD algoritmasını kullanıyoruz
    algo.fit(full_trainset) # Modeli tüm trainset üzerinde eğit

    print(f"Eğitilmiş model şuraya kaydediliyor: {model_output_path}")
    # Modeli kaydet (sadece algoritmayı, tahminlere gerek yok)
    dump.dump(model_output_path, algo=algo)
    print("Model başarıyla kaydedildi!")

if __name__ == '__main__':
    print("Model Eğitim ve Kaydetme Script'i Başlatıldı.")
    # Temizlenmiş reyting verisini yükle
    ratings_data = load_ratings_data_for_training(CLEANED_DATA_PATH)

    # Modeli eğit ve kaydet
    train_and_save_surprise_model(ratings_data, MODEL_SAVE_PATH)
    print("Model Eğitim ve Kaydetme Script'i Tamamlandı.")
