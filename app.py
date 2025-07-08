import streamlit as st
from transformers import pipeline

# Set Streamlit page configuration
st.set_page_config(page_title="Text Summarizer", page_icon="📝")

# Title and description
st.title("📝 Text Summarizer using BART")
st.write("Paste any long text below and click 'Summarize' to get a short, meaningful summary using a pre-trained transformer model.")

# Load the summarization model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# Input text area
text = st.text_area("Enter the text to summarize:", height=200)

# Button to trigger summarization
if st.button("Summarize"):
    if not text.strip():
        st.warning("⚠️ Please enter some text above.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
            st.subheader("📋 Summary Output:")
            st.success(summary[0]['summary_text'])
