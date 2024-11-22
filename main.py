import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the tokenizer and model from your files
model_path = "./t5_model"  # Set the path to the directory containing the T5 model
tokenizer_path = "./t5_tokenizer"  # Set the path to the directory containing the T5 tokenizer

tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Streamlit interface
st.title("Text Summarization App")
st.write("Enter the text you want to summarize:")

# Input text
input_text = st.text_area("Input Text", height=300)

# Summarize the text when button is pressed
if st.button("Summarize"):
    if input_text:
        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate the summary using the model
        summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

        # Decode and display the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("Summary")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize!")
