"""
Prediction module.
Loads the trained model and classifies new texts.
"""

import os
import pickle

from preprocess import TextPreprocessor


class FakeNewsPredictor:
    """
    Classifies news articles with the trained model.

    Loads model.pkl and vectorizer.pkl from the model/ directory.
    """

    LABELS = {0: "Fake News", 1: "Real News"}

    def __init__(self, model_dir: str = "model"):
        """
        Args:
            model_dir: Directory that contains the model files.
        """
        model_path = os.path.join(model_dir, "model.pkl")
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                "Model files not found. Please train the model first with train.py."
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        self.preprocessor = TextPreprocessor()

    def predict(self, text: str) -> dict:
        """
        Predicts the label of a single text.

        Args:
            text: Raw news text.

        Returns:
            label: Predicted label ("Fake News" / "Real News")
            is_fake: True if the text is classified as fake
            confidence: Model confidence as a percentage (0-100)
        """
        cleaned = self.preprocessor.clean(text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        confidence = self.model.predict_proba(vectorized)[0][prediction]

        return {
            "label": self.LABELS[prediction],
            "is_fake": prediction == 0,
            "confidence": round(float(confidence) * 100, 2),
        }
