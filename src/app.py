from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import re
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

le = LabelEncoder()
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # load the dataset
        df = pd.read_csv(
            os.path.join(BASE_DIR, "..", "data", "raw", "language_detection.csv")
        )
        y = df["Language"]

        # label encoding
        y = le.fit_transform(y)

        # load model and vectorizer
        model = pickle.load(
            open(
                os.path.join(BASE_DIR, "..", "models", "language_detection_model.pkl"),
                "rb",
            )
        )
        vectorizer = pickle.load(
            open(
                os.path.join(
                    BASE_DIR, "..", "models", "language_detection_vectorizer.pkl"
                ),
                "rb",
            )
        )

        if request.method == "POST":
            text = request.form["text"]
            if not text.strip():
                return render_template(
                    "home.html", pred="Error: Input text cannot be empty."
                )

            # preprocessing the text
            text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', "", text)
            text = re.sub(r"[[]]", "", text)
            text = text.lower()
            dat = [text]

            # creating the vector
            vect = vectorizer.transform(dat).toarray()

            # making the prediction
            my_pre = model.predict(vect)
            my_pre = le.inverse_transform(my_pre)

        return render_template("home.html", pred=f"The above text is in '{my_pre[0]}'")
    except FileNotFoundError as e:
        return render_template(
            "home.html", pred="Error: Required file not found. Please check the setup."
        )
    except KeyError as e:
        return render_template(
            "home.html", pred="Error: Missing required column in the dataset."
        )
    except ValueError as e:
        return render_template(
            "home.html", pred="Error: Invalid input or model/vectorizer issue."
        )
    except Exception as e:
        return render_template(
            "home.html", pred=f"An unexpected error occurred: {str(e)}"
        )


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=True, host="127.0.0.1", port=5001)
