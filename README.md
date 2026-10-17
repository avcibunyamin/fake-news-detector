# Yapay Zeka Destekli Sahte Haber Tespit Sistemi

Haber metinlerini analiz ederek **gerçek** veya **sahte** olarak sınıflandıran web uygulaması.

## Özellikler

- Metin giriş alanı veya `.txt` dosyası yükleme
- Otomatik NLP ön işleme (temizleme, stopword kaldırma)
- TF-IDF + Logistic Regression tabanlı sınıflandırma
- Güven skoru ile sonuç gösterimi
- Sade ve kullanıcı dostu Streamlit arayüzü

## Kurulum

```bash
git clone https://github.com/avcibunyamin/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

## Veri Seti

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) Kaggle'dan indirilip `data/` klasörüne yerleştirilmelidir:

```
data/
├── Fake.csv
└── True.csv
```

## Kullanım

**1. Modeli eğit:**
```bash
python train.py
```

**2. Uygulamayı başlat:**
```bash
streamlit run app.py
```

## Proje Yapısı

```
fake-news-detector/
├── data/           # Veri seti (Kaggle'dan indirilmeli)
├── model/          # Eğitilmiş model dosyaları
├── preprocess.py   # Metin ön işleme sınıfı
├── train.py        # Model eğitim scripti
├── predict.py      # Tahmin sınıfı
├── app.py          # Streamlit arayüzü
├── requirements.txt
└── proposal.md     # Proje teklifi
```

## Model Performansı

| Metrik | Değer |
|---|---|
| Accuracy | ~%95 |
| Precision | ~%95 |
| Recall | ~%95 |
| F1-Score | ~%95 |

> Not: Değerler train.py çalıştırıldıktan sonra güncellenir.

## Teknolojiler

- Python 3.10+
- scikit-learn — TF-IDF, Logistic Regression
- NLTK — metin ön işleme
- Streamlit — web arayüzü
