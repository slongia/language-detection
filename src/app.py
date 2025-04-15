from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import re
import os
import logging
import config  # Import the config module

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# --- Resource Loading ---
def load_resources():
    """Loads the model, vectorizer, and fitted label encoder."""
    try:
        model_path = os.path.join(
            BASE_DIR, "..", "models", "language_detection_model.pkl"
        )
        vectorizer_path = os.path.join(
            BASE_DIR, "..", "models", "language_detection_vectorizer.pkl"
        )
        data_path = os.path.join(
            BASE_DIR, "..", "data", "raw", "language_detection.csv"
        )

        with (
            open(model_path, "rb") as f_model,
            open(vectorizer_path, "rb") as f_vectorizer,
        ):
            model = pickle.load(f_model)
            vectorizer = pickle.load(f_vectorizer)

        # Load data just to fit the label encoder
        df = pd.read_csv(data_path)
        if "Language" not in df.columns:
            raise KeyError(
                "Column 'Language' not found in the dataset for LabelEncoder fitting."
            )
        le = LabelEncoder()
        le.fit(df["Language"])  # Fit the encoder

        logging.info("Model, vectorizer, and label encoder loaded successfully.")
        return model, vectorizer, le
    except FileNotFoundError as e:
        logging.error(f"Error loading resources: {e}. Please check file paths.")
        raise  # Reraise after logging to stop the app if resources can't load
    except KeyError as e:
        logging.error(f"Error loading resources: {e}.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during resource loading: {e}")
        raise


model, vectorizer, le = load_resources()  # Load resources at startup

app = Flask(__name__)


# --- Text Preprocessing ---
def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    # Remove punctuation, special characters, numbers, and newlines
    text = re.sub(r'[!@#$(),\\n"%^*?\\:;~`0-9\\[\\]]', "", text)
    text = text.lower()
    return text


# --- Routes ---
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text", "").strip()
        if not text:
            return render_template(
                "home.html", pred="Error: Input text cannot be empty."
            )

        processed_text = preprocess_text(text)
        dat = [processed_text]

        vect = vectorizer.transform(dat).toarray()

        my_pre = model.predict(vect)

        predicted_language = le.inverse_transform(my_pre)[0]  # Get the first element

        return render_template(
            "home.html", pred=f"The predicted language is: '{predicted_language}'"
        )

    except KeyError as e:
        logging.warning(f"Missing form field in request: {e}")
        return render_template(
            "home.html", pred="Error: Missing 'text' field in the request."
        )
    except ValueError as e:
        logging.error(f"ValueError during prediction: {e}")
        return render_template(
            "home.html", pred="Error: Invalid input causing prediction issue."
        )
    except Exception as e:
        # Log unexpected errors
        logging.exception(
            "An unexpected error occurred during prediction."
        )  # Log full traceback
        return render_template(
            "home.html", pred="An unexpected error occurred. Please try again later."
        )


if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", config.DEFAULT_FLASK_RUN_HOST)
    port = int(os.environ.get("FLASK_RUN_PORT", config.DEFAULT_FLASK_RUN_PORT))

    debug_str = os.environ.get("FLASK_DEBUG", str(config.DEFAULT_FLASK_DEBUG))
    debug = debug_str.lower() in ("true", "1", "t")

    app.run(debug=debug, host=host, port=port)
