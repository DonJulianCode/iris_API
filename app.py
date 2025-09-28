# app.py - API Flask mejorada que devuelve prediction, probabilities y confidence
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

# Intentar cargar el modelo al iniciar
try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"Modelo cargado desde: {MODEL_PATH}")
    model_loaded = True
except Exception as e:
    logging.exception("No se pudo cargar el modelo:")
    model = None
    model_loaded = False

# Nombre de clases (orden esperado según dataset Iris)
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API lista",
        "model_loaded": model_loaded
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    # Verificar modelo cargado
    if not model_loaded:
        return jsonify({"error": "Modelo no disponible en el servidor."}), 500

    # Parsear JSON
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSON inválido o faltante en el body."}), 400

    # Validar campo 'features'
    if not data or "features" not in data:
        return jsonify({"error": "Debe proporcionar el campo 'features' en el body (lista de números)."}), 400

    features = data["features"]
    if not isinstance(features, (list, tuple)):
        return jsonify({"error": "'features' debe ser una lista de números."}), 400

    # Determinar número de características esperado (si el modelo lo tiene)
    expected = getattr(model, "n_features_in_", None)
    if expected is None:
        expected = 4  # fallback razonable para Iris

    if len(features) != expected:
        return jsonify({"error": f"'features' debe tener {expected} elementos. Recibidos: {len(features)}"}), 400

    # Convertir a numpy array float
    try:
        arr = np.array(features, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({"error": "No se pudieron convertir las features a numeros (float)."}), 400

    # Predicción
    try:
        pred = int(model.predict(arr)[0])
    except Exception:
        logging.exception("Error al predecir:")
        return jsonify({"error": "Error interno al calcular la predicción."}), 500

    # Probabilidades y confianza (si el modelo soporta predict_proba)
    probabilities = None
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(arr)[0].astype(float).tolist()
            confidence = float(max(probabilities)) * 100.0
        except Exception:
            logging.exception("Error al calcular predict_proba (se omitirá):")
            probabilities = None
            confidence = None

    # Nombre de clase seguro
    class_name = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else str(pred)

    response = {
        "prediction": pred,
        "class_name": class_name,
        "probabilities": probabilities,   # lista de floats 0..1 o null
        "confidence": round(confidence, 2) if confidence is not None else None  # porcentaje (0..100) o null
    }

    return jsonify(response), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
