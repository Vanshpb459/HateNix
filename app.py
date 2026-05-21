import streamlit as st
import os
import sys
from flask import Flask
app = Flask(__name__)
# Add HateNix directory to path to import HateSpeechModel
sys.path.append(os.path.join(os.path.dirname(__file__), 'HateNix'))
from HateNix import HateSpeechModel

st.set_page_config(page_title="HateNix - AI Moderation", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    # Only use RoBERTa model since local pkl is not present
    return HateSpeechModel()

st.title("🛡️ HateNix: Hate Speech Detection")
st.markdown("Analyze text to detect hate speech and offensive language using AI.")

model = load_model()

text_input = st.text_area("Enter text to analyze:", height=150, placeholder="Paste a comment or text here...")

if st.button("Analyze Text", type="primary"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            results = model.predict([text_input])
            pred, score = results[0]
            
            st.subheader("Result")
            if pred == 1:
                st.error(f"🚨 Hate Speech Detected! (Confidence: {score:.1%})")
            else:
                st.success(f"✅ Safe Text (Confidence of being safe: {1-score:.1%})")
            
            st.progress(score if pred == 1 else 1-score)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.caption("Powered by RoBERTa and HateNix AI")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
