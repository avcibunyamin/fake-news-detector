"""
Model eğitim modülü.
TF-IDF vektörizasyonu ve Logistic Regression ile sahte haber tespiti modeli eğitir.
"""

import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from preprocess import TextPreprocessor


class FakeNewsTrainer:
    """
    Sahte haber tespit modeli eğiten sınıf.

    TF-IDF vektörizasyonu ile Logistic Regression modelini birleştirir.
    Eğitilmiş model ve vektörizer disk'e kaydedilir.
    """

    def __init__(self, model_dir: str = "model"):
        """
        Args:
            model_dir: Model dosyalarının kaydedileceği dizin.
        """
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        os.makedirs(model_dir, exist_ok=True)

    def load_dataset(self, fake_path: str, real_path: str) -> pd.DataFrame:
        """
        Fake ve Real CSV dosyalarını yükler ve birleştirir.

        Args:
            fake_path: Sahte haberler CSV dosyası yolu.
            real_path: Gerçek haberler CSV dosyası yolu.

        Returns:
            Birleştirilmiş DataFrame (text, label sütunları).
        """
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)

        fake_df["label"] = 0  # 0 = Sahte
        real_df["label"] = 1  # 1 = Gerçek

        # Başlık ve içeriği birleştir
        for df in [fake_df, real_df]:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

        combined = pd.concat([fake_df[["text", "label"]], real_df[["text", "label"]]])
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Toplam örnek: {len(combined)}")
        print(f"Sahte: {(combined['label'] == 0).sum()} | Gerçek: {(combined['label'] == 1).sum()}")
        return combined

    def train(self, fake_path: str, real_path: str, test_size: float = 0.2):
        """
        Modeli eğitir ve değerlendirme metriklerini yazdırır.

        Args:
            fake_path: Sahte haberler CSV yolu.
            real_path: Gerçek haberler CSV yolu.
            test_size: Test seti oranı (varsayılan: 0.2).
        """
        df = self.load_dataset(fake_path, real_path)

        print("Ön işleme uygulanıyor...")
        df["clean_text"] = self.preprocessor.clean_batch(df["text"].tolist())

        X = df["clean_text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print("TF-IDF vektörizasyonu...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        print("Model eğitimi başlıyor...")
        self.model.fit(X_train_vec, y_train)

        y_pred = self.model.predict(X_test_vec)
        self._print_metrics(y_test, y_pred)
        self._save()

    def _print_metrics(self, y_true, y_pred):
        """Değerlendirme metriklerini ekrana yazdırır."""
        print("\n========== Model Performansı ==========")
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
        print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")
        print("\nDetaylı Rapor:")
        print(classification_report(y_true, y_pred, target_names=["Sahte", "Gerçek"]))

    def _save(self):
        """Eğitilmiş model ve vektörizeri disk'e kaydeder."""
        with open(os.path.join(self.model_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Model '{self.model_dir}/' dizinine kaydedildi.")


if __name__ == "__main__":
    trainer = FakeNewsTrainer()
    trainer.train(
        fake_path="data/Fake.csv",
        real_path="data/True.csv",
    )
