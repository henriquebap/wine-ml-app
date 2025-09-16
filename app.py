import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestRegressor
import gradio as gr

# Configurações - troque pelo seu usuário e dataset repo
HF_DATASET_REPO = "henriquebap/wine-ml-dataset"
CSV_FILENAME = "WineQT.csv"

model = None
feature_cols = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
]


def load_data():
    csv_path = hf_hub_download(repo_id=HF_DATASET_REPO, repo_type="dataset", filename=CSV_FILENAME)
    df = pd.read_csv(csv_path)
    # Garantir colunas esperadas; remover colunas extras
    cols = set(feature_cols + ["quality"])
    df = df[[c for c in df.columns if c in cols]]
    return df.dropna()

def train():
    global model
    df = load_data()
    X = df[feature_cols]
    y = df["quality"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return f"Modelo treinado com {len(df)} linhas."

def predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    if model is None:
        train()
    x = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]], columns=feature_cols)
    pred = float(model.predict(x)[0])
    return f"{pred:.2f} (arredondado: {int(round(pred))})"

with gr.Blocks(title="Wine Quality - MVP") as demo:
    gr.Markdown("## 🍷 Wine Quality - MVP (RandomForest)")
    status = gr.Textbox(label="Status do Treino", interactive=False)
    btn_train = gr.Button("Treinar/Re-treinar modelo do CSV do HuggingFace")
    btn_train.click(fn=train, outputs=status)

    gr.Markdown("### Fazer predição")
    with gr.Row():
        fixed_acidity = gr.Number(value=7.5, label="fixed acidity")
        volatile_acidity = gr.Number(value=0.5, label="volatile acidity")
        citric_acid = gr.Number(value=0.25, label="citric acid")
        residual_sugar = gr.Number(value=2.0, label="residual sugar")
        chlorides = gr.Number(value=0.08, label="chlorides")
        free_sd = gr.Number(value=15.0, label="free sulfur dioxide")
        total_sd = gr.Number(value=50.0, label="total sulfur dioxide")
        density = gr.Number(value=0.996, label="density")
        pH = gr.Number(value=3.3, label="pH")
        sulphates = gr.Number(value=0.6, label="sulphates")
        alcohol = gr.Number(value=10.0, label="alcohol")
    out = gr.Textbox(label="Qualidade prevista")

    btn_pred = gr.Button("Prever")
    btn_pred.click(
        predict,
        inputs=[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                free_sd, total_sd, density, pH, sulphates, alcohol],
        outputs=out
    )

    # Treina automático ao iniciar (rápido)
    status.value = train()

demo.launch()