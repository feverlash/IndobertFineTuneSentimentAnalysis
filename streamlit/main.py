import streamlit as st
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

# Load the model and tokenizer from Hugging Face
model_name = "feverlash/Indonesian-SentimentAnalysis-Model"

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Sentiment mapping
sentiment_mapping = {
    "positive": 1,
    "negative": 0,
    "neutral": 2
}

def predict(text):
    """
    Function to predict the sentiment of a given text.
    """
    inputs = tokenizer(
        text,
        return_tensors="tf",  # TensorFlow tensors
        truncation=True,      # Truncate if text is too long
        padding="max_length", # Pad to the maximum length
        max_length=128        # Maximum sequence length
    )
    
    # Predict using the model
    outputs = model(inputs)
    logits = outputs.logits
    
    # Calculate probabilities
    probabilities = tf.nn.softmax(logits).numpy()
    
    # Get the predicted index
    predicted_index = int(tf.argmax(probabilities, axis=1).numpy()[0])  # Convert to integer
    
    # Map index to sentiment label
    index_to_sentiment = {v: k for k, v in sentiment_mapping.items()}
    predicted_label = index_to_sentiment.get(predicted_index, "unknown")
    
    # Confidence of the prediction
    confidence = probabilities[0][predicted_index]
    
    return predicted_label, confidence

def run():
    st.title('Identifikasi Sentimen dari Suatu Kalimat')
    st.write("Masukkan teks untuk mengidentifikasi apakah sentimen tersebut positif, negatif, atau netral.")

    # Input text from the user
    sentence = st.text_input('Masukkan Teks Anda')
    
    if st.button('Prediksi'):
        if sentence.strip():
            predicted_label, confidence = predict(sentence)
            st.write(f"**Prediksi Sentimen:** {predicted_label}")
            st.write(f"**Confidence:** {confidence:.2f}")
        else:
            st.write("Silakan masukkan teks terlebih dahulu!")

if __name__ == '__main__':
    run()
