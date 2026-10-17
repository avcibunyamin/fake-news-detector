"""
Streamlit web interface.
Takes a news article from the user and classifies it as fake or real.
"""

import streamlit as st
from predict import FakeNewsPredictor

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="centered",
)

# Header
st.title("🔍 Fake News Detection System")
st.markdown("Enter a news article or upload a file and let the AI analyze it.")
st.divider()


@st.cache_resource
def load_predictor():
    """Loads the model once and caches it."""
    return FakeNewsPredictor()


try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Input method
input_method = st.radio("Input method:", ["Enter Text", "Upload File"], horizontal=True)

text_input = ""

if input_method == "Enter Text":
    text_input = st.text_area(
        "Paste the news article here:",
        height=200,
        placeholder="Enter the headline and body of the article...",
    )
else:
    uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
    if uploaded_file is not None:
        text_input = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded text:", value=text_input, height=200, disabled=True)

st.divider()

# Analyze button
if st.button("Analyze", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text or upload a file.")
    else:
        with st.spinner("Analyzing..."):
            result = predictor.predict(text_input)

        # Result card
        if result["is_fake"]:
            st.error(f"### Result: {result['label']}")
            st.metric("Confidence", f"{result['confidence']}%")
            st.markdown(
                "> The AI classified this article as **fake**. "
                "Please verify it with reliable sources."
            )
        else:
            st.success(f"### Result: {result['label']}")
            st.metric("Confidence", f"{result['confidence']}%")
            st.markdown(
                "> The AI classified this article as **real**."
            )

st.divider()
st.caption("AI-Powered Fake News Detection System | TF-IDF + Logistic Regression")
