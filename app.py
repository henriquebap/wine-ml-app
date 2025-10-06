import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestRegressor
import gradio as gr
from pathlib import Path
import joblib

# Configurações - troque pelo seu usuário e dataset repo
HF_DATASET_REPO = "henriquebap/wine-ml-dataset"
CSV_FILENAME = "WineQT.csv"
MODEL_PATH = Path("data/models/wine_quality_regressor.joblib")

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


def load_final_model():
    global model, feature_cols
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            loaded_model = bundle.get("model", None)
            meta = bundle.get("metadata", {})
            feats = meta.get("features")
            if isinstance(feats, list) and len(feats) > 0:
                feature_cols = feats
            if loaded_model is not None:
                model = loaded_model
                return "Modelo final carregado do disco."
        except Exception as e:
            return f"Falha ao carregar modelo salvo: {e}"
    return None

def train():
    global model
    df = load_data()
    X = df[feature_cols]
    y = df["quality"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return f"Modelo treinado com {len(df)} linhas."


def load_or_train():
    """Tenta carregar o modelo final salvo; se não existir, treina um baseline do CSV."""
    msg = load_final_model()
    if msg:
        return msg
    return train()

def predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    if model is None:
        load_or_train()
    x = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]], columns=feature_cols)
    pred = float(model.predict(x)[0])
    return f"{pred:.2f} (arredondado: {int(round(pred))})"


def predict_batch(file: gr.File | None):
    if model is None:
        load_or_train()
    if file is None:
        return "Arquivo CSV não fornecido.", None
    try:
        df_in = pd.read_csv(file.name)
    except Exception as e:
        return f"Falha ao ler CSV: {e}", None
    missing = [c for c in feature_cols if c not in df_in.columns]
    if missing:
        return f"CSV faltando colunas: {missing}", None
    preds = model.predict(df_in[feature_cols])
    out = df_in.copy()
    out["pred_quality"] = preds
    return f"OK - {len(out)} linhas processadas.", out


def model_info():
    if model is None:
        load_or_train()
    info = {
        "features": feature_cols,
    }
    try:
        import numpy as _np
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            s = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
            info["feature_importance_top"] = s.head(10).to_dict()
    except Exception:
        pass
    return info

with gr.Blocks(title="Wine Quality - MVP") as demo:
    gr.Markdown("## 🍷 Wine Quality - MVP (Modelo Final + Fallback de Treino)")
    status = gr.Textbox(label="Status", interactive=False)
    with gr.Row():
        btn_load = gr.Button("Carregar modelo final / Treinar")
        btn_info = gr.Button("Info do modelo")
    btn_load.click(fn=load_or_train, outputs=status)

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

    gr.Markdown("### Predição em lote (CSV)")
    csv_in = gr.File(label="CSV com colunas de features", file_types=[".csv"])
    msg, df_out = gr.Textbox(label="Mensagem"), gr.Dataframe(label="Resultado")
    btn_batch = gr.Button("Processar CSV")
    btn_batch.click(predict_batch, inputs=csv_in, outputs=[msg, df_out])

    info_out = gr.JSON(label="Detalhes do modelo")
    btn_info.click(model_info, outputs=info_out)

    # Carrega ao iniciar (ou treina se ausente)
    status.value = load_or_train()

if __name__ == "__main__":
    demo.launch()