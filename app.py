import streamlit as st
import pickle
import re
from pathlib import Path
from typing import Tuple

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

@st.cache_resource(show_spinner=True)
def load_artifacts() -> Tuple[object, object]:
    m_path = MODELS_DIR / "model.pkl"
    v_path = MODELS_DIR / "vectorizer.pkl"
    with m_path.open("rb") as fm:
        m = pickle.load(fm)
    with v_path.open("rb") as fv:
        v = pickle.load(fv)
    return m, v

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check if it is Fake or Real.")

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("Artifacts failed to load.")
    st.caption(str(e))
    st.stop()

user_input = st.text_area("Enter News Content Here:")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        pred = model.predict(vectorized)[0]
        if pred == 0:
            st.error("⚠ This News is FAKE")
        else:
            st.success("✅ This News is REAL")
    else:
        st.warning("Please enter some text.")
