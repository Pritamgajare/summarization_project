import streamlit as st
from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit interface
st.title("Text Summarization App")
st.write("Enter the text you want to summarize:")

# Input text
input_text = st.text_area("Input Text", height=300)

# Summarize the text when button is pressed
if st.button("Summarize"):
    if input_text:
        # Use model to generate summary
        summary = summarizer(input_text, max_length=200, min_length=50, do_sample=False)
        st.subheader("Summary")
        st.write(summary[0]['summary_text'])
    else:
        st.error("Please enter some text to summarize!")
