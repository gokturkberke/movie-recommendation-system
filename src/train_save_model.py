import pandas as pd
import os
from surprise import Dataset, Reader, SVD
from surprise import dump # Model kaydetmek/yüklemek için

# Bu script'in (train_save_model.py) bulunduğu dizin (src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# cleaned_data klasörünün yolu (src klasörünün bir üst dizininde olduğunu varsayıyoruz)
CLEANED_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'cleaned_data'))
# Kaydedilecek modelin tam yolu (cleaned_data klasörü içine)
MODEL_SAVE_PATH = os.path.join(CLEANED_DATA_PATH, 'svd_trained_model.pkl')

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
