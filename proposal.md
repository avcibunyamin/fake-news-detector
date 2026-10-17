# Project Proposal: AI-Powered Fake News Detection System

**Date:** April 2026

---

## Goal

This project builds a system that uses natural language processing (NLP) and machine learning
to automatically classify news articles as **real** or **fake**. Given the social impact of
misinformation spreading through social media and the internet, such a system has clear
practical value. Results are served in real time through a user-friendly Streamlit web
interface.

---

## Dataset

The project uses the **Fake and Real News Dataset** published on Kaggle.
It consists of two CSV files:

- `Fake.csv` — more than 23,000 fake news articles
- `True.csv` — more than 21,000 real news articles

Each record has a title (`title`), body (`text`), subject (`subject`) and publication date
(`date`). Labels are binary: `0 = Fake`, `1 = Real`.

Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Method

### Preprocessing
Raw texts go through the following steps:
- Lowercasing
- Removing URLs, numbers and punctuation
- Removing English stopwords

### Vectorization
Cleaned texts are converted into numeric vectors with **TF-IDF (Term Frequency–Inverse
Document Frequency)**. A combination of unigrams and bigrams (`ngram_range=(1, 2)`) keeps
some contextual information.

### Model
**Logistic Regression** is used for classification because it is interpretable, fast to
train and works well with TF-IDF features.

### Evaluation
Model performance is measured with:
- Accuracy, Precision, Recall, F1-Score

---

## Expected Outcome

Similar studies report around 95% accuracy for TF-IDF + Logistic Regression on this dataset.
The deliverable is a simple, working web app where users paste any news article and get a
reliability prediction within seconds.

---

## Tech Stack

| Layer | Tool / Library |
|---|---|
| Language | Python 3.10+ |
| NLP / ML | scikit-learn, NLTK |
| Interface | Streamlit |
| Version control | Git / GitHub |

---

## References

- Ahmed, H. et al. (2018). *Detecting opinion spams and fake news using text classification.*
- Kaggle – Fake and Real News Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- scikit-learn Documentation: https://scikit-learn.org
