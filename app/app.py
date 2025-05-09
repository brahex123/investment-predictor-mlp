import gradio as gr
import joblib
import numpy as np
import tensorflow as tf

# 📦 Charger le modèle et le scaler
model = tf.keras.models.load_model("model/mlp_final_model.h5")
y_scaler = joblib.load("model/y_scaler.pkl")

# 🧾 Liste des variables utilisées à l'entraînement (remplace si besoin)
features = [
    "PIB réel",
    "Inflation",
    "Taux de chômage",
    "Population urbaine",
    "Flux d'IDE",
    "Commerce extérieur (% PIB)",
    "Croissance PIB",
    "Population totale",
    "PIB courant"
]

# 🧮 Fonction de prédiction
def predict_investment(*inputs):
    # Convertir les inputs en tableau 2D
    X = np.array([inputs])
    # Faire la prédiction (y normalisé)
    y_scaled = model.predict(X)
    # Inverser la normalisation
    y_pred = y_scaler.inverse_transform(y_scaled)
    return f"📈 Prédiction de l'investissement : {y_pred[0][0]:.2f} % du PIB"

# 🎛️ Interface Gradio
inputs = [gr.Number(label=feature) for feature in features]
interface = gr.Interface(
    fn=predict_investment,
    inputs=inputs,
    outputs="text",
    title="🔍 Prédiction de l'investissement (% PIB)",
    description="Entrez les variables économiques pour obtenir une prédiction basée sur le modèle MLP optimisé."
)

# 🚀 Lancer l'app
if __name__ == "__main__":
    interface.launch()

