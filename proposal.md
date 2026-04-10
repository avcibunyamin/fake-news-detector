# Proje Teklifi: Yapay Zeka Destekli Sahte Haber Tespit Sistemi

**Öğrenci:** Bünyamin AVCI
**Tarih:** Nisan 2026

---

## Hedef

Bu projede, doğal dil işleme (NLP) ve makine öğrenimi yöntemleri kullanılarak haber
metinlerini otomatik olarak **gerçek** veya **sahte** olarak sınıflandırabilen bir sistem
geliştirilecektir. Son yıllarda sosyal medya ve internet üzerinden yayılan yanlış bilgilerin
toplumsal etkileri göz önünde bulundurulduğunda, bu tür bir sistemin pratik değeri oldukça
yüksektir. Sistemin çıktıları, kullanıcı dostu bir Streamlit web arayüzü aracılığıyla gerçek
zamanlı olarak sunulacaktır.

---

## Veri Seti

Projede Kaggle üzerinde yayımlanmış olan **Fake and Real News Dataset** kullanılacaktır.
Bu veri seti iki ayrı CSV dosyasından oluşmaktadır:

- `Fake.csv` — 23.000'den fazla sahte haber makalesi
- `True.csv` — 21.000'den fazla gerçek haber makalesi

Her kayıt; haber başlığı (`title`), içerik (`text`), konu (`subject`) ve yayın tarihi
(`date`) alanlarını içermektedir. Etiketler ikili yapıdadır: `0 = Sahte`, `1 = Gerçek`.

Veri kaynağı: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Yöntem

### Ön İşleme
Ham metinler aşağıdaki adımlardan geçirilecektir:
- Küçük harfe çevirme
- URL, sayı ve noktalama işaretlerinin kaldırılması
- İngilizce stopword'lerin çıkarılması

### Vektörizasyon
Temizlenmiş metinler, **TF-IDF (Term Frequency–Inverse Document Frequency)** yöntemiyle
sayısal vektörlere dönüştürülecektir. Unigram ve bigram kombinasyonu (ngram_range=(1,2))
kullanılarak bağlamsal bilgi korunacaktır.

### Model
Sınıflandırma için **Logistic Regression** algoritması tercih edilecektir. Bu tercih;
yorumlanabilirlik, eğitim hızı ve TF-IDF ile yüksek uyumluluk gibi avantajlara
dayanmaktadır.

### Değerlendirme
Model performansı aşağıdaki metrikler ile ölçülecektir:
- Accuracy, Precision, Recall, F1-Score

---

## Beklenen Sonuç

Literatürdeki benzer çalışmalar incelendiğinde TF-IDF + Logistic Regression kombinasyonunun
bu veri setinde %95 civarında doğruluk sağladığı görülmektedir. Proje sonunda kullanıcıların
herhangi bir haber metnini yapıştırarak saniyeler içinde güvenilirlik tahmini alabildiği,
işlevsel ve sade bir web uygulaması teslim edilecektir.

---

## Teknoloji Yığını

| Katman | Araç / Kütüphane |
|---|---|
| Programlama dili | Python 3.10+ |
| NLP / ML | scikit-learn, NLTK |
| Arayüz | Streamlit |
| Versiyon kontrolü | Git / GitHub |

---

## Referanslar

- Ahmed, H. et al. (2018). *Detecting opinion spams and fake news using text classification.*
- Kaggle – Fake and Real News Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- scikit-learn Documentation: https://scikit-learn.org
