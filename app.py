
# Import Required Libraries
import streamlit as st
import numpy as np
import re
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model


# Set Page Configuration
st.set_page_config(
    page_title="McDonald's Review Rating Prediction",
    page_icon="🍔",
    layout="centered"
)


# Define File Paths
BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "final_ann_model.keras"
vectorizer_path = BASE_DIR / "tfidf_vectorizer.pkl"


# Load ANN Model
@st.cache_resource
def load_ann_model():
    return load_model(model_path)


# Load TF-IDF Vectorizer
@st.cache_resource
def load_tfidf():
    return joblib.load(vectorizer_path)


# Load Saved Files
model = load_ann_model()
tfidf = load_tfidf()


# Text Cleaning Function
def clean_text(text):

    # Convert Text into Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML Tags
    text = re.sub(r'<.*?>', '', text)

    # Remove Special Characters and Numbers
    text = re.sub(r"[^a-z'\s]", ' ', text)

    # Remove Extra Spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Streamlit Application
st.title("🍔 McDonald's Customer Review Rating Prediction")

st.write(
    "Predict customer ratings using TF-IDF and Optuna Tuned ANN."
)

st.divider()


# User Input
review = st.text_area(
    "Enter Customer Review",
    placeholder="The food was delicious and the service was excellent.",
    height=150
)


# Prediction Button
if st.button("Predict Rating", type="primary"):

    if review.strip() == "":
        st.warning("Please enter a customer review.")

    else:

        # Clean Review
        cleaned_review = clean_text(review)

        if cleaned_review == "":
            st.warning("Please enter a valid review.")

        else:

            # Convert Text into TF-IDF Features
            review_tfidf = tfidf.transform([cleaned_review])

            # Convert Sparse Matrix into Dense Array
            review_tfidf = review_tfidf.toarray().astype(np.float32)

            # Predict Rating
            prediction = model.predict(review_tfidf, verbose=0)

            # Get Predicted Class
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Convert Class into Star Rating
            predicted_rating = predicted_class + 1

            # Get Prediction Confidence
            confidence = np.max(prediction) * 100

            # Display Prediction
            st.success(
                f"Predicted Rating: {predicted_rating} / 5 Stars"
            )

            st.subheader("Predicted Rating")
            st.write("⭐" * predicted_rating)

            st.write(
                f"Prediction Confidence: {confidence:.2f}%"
            )

            # Display Rating Probabilities
            st.subheader("Rating Probability Distribution")

            for i, probability in enumerate(prediction[0]):
                st.write(
                    f"{i + 1} Star: {probability * 100:.2f}%"
                )
