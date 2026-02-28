import streamlit as st
import pickle
import re

# Load model and vectorizer
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function (MUST match training cleaning)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check if it is Fake or Real.")

user_input = st.text_area("Enter News Content Here:")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 0:
            st.error("⚠ This News is FAKE")
        else:
            st.success("✅ This News is REAL")
    else:
        st.warning("Please enter some text.")