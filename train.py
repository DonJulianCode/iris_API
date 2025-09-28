
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Cargar dataset Iris
data = load_iris()
X, y = data.data, data.target

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Guardar modelo
joblib.dump(model, "model.pkl")
print("Modelo entrenado y guardado como model.pkl")
