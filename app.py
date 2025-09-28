
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "API lista"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features or not isinstance(features, list):
            return jsonify({"error": "Debe proporcionar 'features' como lista"}), 400

        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
