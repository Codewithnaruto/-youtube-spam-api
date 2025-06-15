from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and authors
model = joblib.load("model.pkl")
known_authors = joblib.load("known_authors.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    comment = data.get("comment")
    author = data.get("author")

    # Spam prediction
    prediction = model.predict([comment])[0]
    prediction_label = "spam" if prediction == 1 else "not spam"

    # Author check
    author_status = "authorized" if author in known_authors else "unauthorized"

    return jsonify(
        {"comment_prediction": prediction_label, "author_status": author_status}
    )


if __name__ == "__main__":
    app.run(debug=True)
