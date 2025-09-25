---
title: Wine Ml App
emoji: ⚡
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false
short_description: MVP-wine-ml-app
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Setup local

1. Criar venv e instalar dependências:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure o token do Hugging Face (se necessário para datasets privados):

```
export HF_TOKEN=seu_token
export HF_DATASET_REPO=henriquebap/wine-ml-dataset
export HF_DATASET_FILENAME=WineQT.csv
```

3. Estrutura sugerida:

```
📁 data/
├── raw/
├── processed/
└── models/

📁 src/
├── data_ingestion.py
└── data_processing.py

📓 notebooks/
├── 01_exploratory_data_analysis.ipynb
├── 02_statistical_analysis.ipynb
└── 03_data_visualization.ipynb
```

## Notebooks

- 01 EDA: inspeção inicial, estatísticas, distribuições e correlações
- 02 Estatística: normalidade, ANOVA, outliers
- 03 Visualização: histogramas, boxplots, scatter, heatmap

## App Gradio

O `app.py` treina um `RandomForestRegressor` usando o CSV do HF e permite inferência manual.

## Próximos passos

- Comparação de modelos, tuning e avaliação em notebooks 04-07
- Opcional: dashboard (Streamlit) e API (FastAPI)
