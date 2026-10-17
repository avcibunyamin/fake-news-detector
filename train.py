"""
Model training module.
Trains a fake news classifier with TF-IDF vectorization and Logistic Regression.
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
    Trains the fake news classifier.

    Combines TF-IDF vectorization with a Logistic Regression model.
    The trained model and vectorizer are saved to disk.
    """

    def __init__(self, model_dir: str = "model"):
        """
        Args:
            model_dir: Directory where the model files are saved.
        """
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        os.makedirs(model_dir, exist_ok=True)

    def load_dataset(self, fake_path: str, real_path: str) -> pd.DataFrame:
        """
        Loads and merges the fake and real news CSV files.

        Args:
            fake_path: Path to the fake news CSV file.
            real_path: Path to the real news CSV file.

        Returns:
            Merged DataFrame with text and label columns.
        """
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)

        fake_df["label"] = 0  # 0 = fake
        real_df["label"] = 1  # 1 = real

        # Combine title and body
        for df in [fake_df, real_df]:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

        combined = pd.concat([fake_df[["text", "label"]], real_df[["text", "label"]]])
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Total samples: {len(combined)}")
        print(f"Fake: {(combined['label'] == 0).sum()} | Real: {(combined['label'] == 1).sum()}")
        return combined

    def train(self, fake_path: str, real_path: str, test_size: float = 0.2):
        """
        Trains the model and prints evaluation metrics.

        Args:
            fake_path: Path to the fake news CSV.
            real_path: Path to the real news CSV.
            test_size: Test set ratio (default: 0.2).
        """
        df = self.load_dataset(fake_path, real_path)

        print("Preprocessing...")
        df["clean_text"] = self.preprocessor.clean_batch(df["text"].tolist())

        X = df["clean_text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print("Vectorizing with TF-IDF...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        print("Training the model...")
        self.model.fit(X_train_vec, y_train)

        y_pred = self.model.predict(X_test_vec)
        self._print_metrics(y_test, y_pred)
        self._save()

    def _print_metrics(self, y_true, y_pred):
        """Prints evaluation metrics."""
        print("\n========== Model Performance ==========")
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
        print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")
        print("\nDetailed report:")
        print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

    def _save(self):
        """Saves the trained model and vectorizer to disk."""
        with open(os.path.join(self.model_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Model saved to '{self.model_dir}/'.")


if __name__ == "__main__":
    trainer = FakeNewsTrainer()
    trainer.train(
        fake_path="data/Fake.csv",
        real_path="data/True.csv",
    )
