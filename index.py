import streamlit as st
import pickle
from words import clean_words

st.set_page_config(
    page_title="Spam Message Classifier", 
    page_icon="📩", 
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

try:
    tfidf, model = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts missing! Please run `python mainmodel.py` first.")
    st.stop()

st.title("📩 SMS Spam Detector")
st.write("Enter an SMS or email message below to check if it's **Spam** or **Legitimate (Ham)**.")

user_input = st.text_area(
    "Message Content:", 
    height=150, 
    placeholder="Type or paste message here..."
)

if st.button("Check Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message before checking.")
    else:
        cleaned_text = clean_words(user_input)
        vectorized_text = tfidf.transform([cleaned_text]).toarray()
        prediction = model.predict(vectorized_text)[0]
        prob = model.predict_proba(vectorized_text)[0]
        confidence = round(prob[1] * 100, 2)
        
        st.write("### Result:")
        if prediction == 1:
            st.error(f"🚨 **SPAM DETECTED** ({confidence}% confidence)")
        else:
            st.success(f"✅ **NOT SPAM / HAM** ({100 - confidence}% confidence)")