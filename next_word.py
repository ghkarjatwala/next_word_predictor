import streamlit as st
import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
st.title("📖 Next Word Predictor ")

st.write("This app trains a GRU model on Shakespeare’s *Hamlet* and predicts the next word.")

@st.cache_resource
def train_gru_model():
    # Download Hamlet
    nltk.download('gutenberg')
    from nltk.corpus import gutenberg
    text = gutenberg.raw('shakespeare-hamlet.txt').lower()

    # Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    # Input sequences
    input_sequences = []
    for line in text.split("\n"):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_seq = token_list[: i + 1]
            input_sequences.append(n_gram_seq)

    max_sequence_len = max(len(x) for x in input_sequences)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre"))

    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Build GRU model
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
    model.add(GRU(150, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(100))
    model.add(Dense(total_words, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train small (to keep fast)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

    return model, tokenizer, max_sequence_len


model, tokenizer, max_sequence_len = train_gru_model()

# ---------------------------
# Prediction Function
# ---------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1) :]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# ---------------------------
# Streamlit UI
# ---------------------------
input_text = st.text_input("✍️ Enter a phrase:")

if st.button("🔮 Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.success(f"**Next word prediction:** {next_word}")
