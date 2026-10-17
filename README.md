# AI-Powered Fake News Detection System

A web app that classifies news articles as **real** or **fake**.

## Features

- Paste text or upload a `.txt` file
- Automatic NLP preprocessing (cleaning, stopword removal)
- TF-IDF + Logistic Regression classifier
- Result shown with a confidence score
- Simple Streamlit interface

## Installation

```bash
git clone https://github.com/avcibunyamin/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

## Dataset

Download the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place it in `data/`:

```
data/
├── Fake.csv
└── True.csv
```

## Usage

**1. Train the model:**
```bash
python train.py
```

**2. Start the app:**
```bash
streamlit run app.py
```

## Project Structure

```
fake-news-detector/
├── data/           # Dataset (download from Kaggle)
├── model/          # Trained model files
├── preprocess.py   # Text preprocessing
├── train.py        # Model training script
├── predict.py      # Prediction class
├── app.py          # Streamlit interface
├── requirements.txt
└── proposal.md     # Project proposal
```

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | ~95% |
| Precision | ~95% |
| Recall | ~95% |
| F1-Score | ~95% |

> Note: `train.py` prints the actual values after training.

## Tech Stack

- Python 3.10+
- scikit-learn — TF-IDF, Logistic Regression
- NLTK — text preprocessing
- Streamlit — web interface
