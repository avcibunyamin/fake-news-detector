"""
Streamlit web arayüzü.
Kullanıcıdan haber metni alır ve sahte/gerçek sınıflandırması yapar.
"""

import streamlit as st
from predict import FakeNewsPredictor

# Sayfa yapılandırması
st.set_page_config(
    page_title="Sahte Haber Tespit Sistemi",
    page_icon="🔍",
    layout="centered",
)

# Başlık
st.title("🔍 Sahte Haber Tespit Sistemi")
st.markdown("Bir haber metni girin veya dosya yükleyin — yapay zeka analiz etsin.")
st.divider()


@st.cache_resource
def load_predictor():
    """Modeli bir kez yükler ve önbelleğe alır."""
    return FakeNewsPredictor()


try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Giriş seçimi
input_method = st.radio("Giriş yöntemi:", ["Metin Gir", "Dosya Yükle"], horizontal=True)

text_input = ""

if input_method == "Metin Gir":
    text_input = st.text_area(
        "Haber metnini buraya yapıştırın:",
        height=200,
        placeholder="Haber başlığı ve içeriğini buraya girin...",
    )
else:
    uploaded_file = st.file_uploader("Metin dosyası yükleyin (.txt)", type=["txt"])
    if uploaded_file is not None:
        text_input = uploaded_file.read().decode("utf-8")
        st.text_area("Yüklenen metin:", value=text_input, height=200, disabled=True)

st.divider()

# Analiz butonu
if st.button("Analiz Et", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Lütfen bir metin girin veya dosya yükleyin.")
    else:
        with st.spinner("Analiz ediliyor..."):
            result = predictor.predict(text_input)

        # Sonuç kartı
        if result["is_fake"]:
            st.error(f"### Sonuç: {result['label']}")
            st.metric("Güven Skoru", f"%{result['confidence']}")
            st.markdown(
                "> Bu haber yapay zeka tarafından **sahte** olarak sınıflandırıldı. "
                "Lütfen güvenilir kaynaklardan doğrulayın."
            )
        else:
            st.success(f"### Sonuç: {result['label']}")
            st.metric("Güven Skoru", f"%{result['confidence']}")
            st.markdown(
                "> Bu haber yapay zeka tarafından **gerçek** olarak sınıflandırıldı."
            )

st.divider()
st.caption("Yapay Zeka Destekli Sahte Haber Tespit Sistemi | TF-IDF + Logistic Regression")
