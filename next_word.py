import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences  # ✅ fixed import

# --- Fix for loading models across TF/Keras versions ---
try:
    from keras.src.legacy.saving import load_model
except ImportError:
    from tensorflow.keras.models import load_model

# ----------------- Load Tokenizer -----------------
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# ----------------- Predict Next Word -----------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# ----------------- Streamlit App -----------------
st.title("Next-Word Predictor using GRU")
st.write("Load a trained GRU model + tokenizer and predict the next word in your sequence.")

# File uploaders
model_path = st.file_uploader("📂 Model path (.h5 or .keras)", type=["h5", "keras"])
tokenizer_path = st.file_uploader("📂 Tokenizer path (.pkl / .pickle)", type=["pkl", "pickle"])

if model_path and tokenizer_path:
    try:
        # Save uploaded files temporarily
        with open("uploaded_model.h5", "wb") as f:
            f.write(model_path.getbuffer())
        with open("uploaded_tokenizer.pkl", "wb") as f:
            f.write(tokenizer_path.getbuffer())

        # Load model & tokenizer
        model = load_model("uploaded_model.h5")
        tokenizer = load_tokenizer("uploaded_tokenizer.pkl")

        st.success("✅ Model & Tokenizer loaded successfully!")

        # Input text
        input_text = st.text_input("✍️ Enter your input text:")

        if st.button("🔮 Predict Next Word"):
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            st.write(f"**Next Word Prediction:** {next_word}")

    except Exception as e:
        st.error(f"❌ Failed to load model/tokenizer: {str(e)}")