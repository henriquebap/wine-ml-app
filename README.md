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

# Wine ML App

Este projeto é desenvolvido em Python e Jupyter Notebooks que realiza uma analise em um Dataset de qualidade de vinhos do [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset). A aplicação passa por algumas fase para avaliar e comparar os diferentes tipos de vinhos.

## Tecnologias utilizadas

- Python
- [Jupyter Notebooks](https://docs.jupyter.org/en/latest/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- [Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference)
- [Pandas](https://pandas.pydata.org/docs/)
- [Gradio](https://www.gradio.app/docs)
- [Matplotlib](https://matplotlib.org/stable/index.html)

## Setup local

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/henriquebap/wine-ml-app.git
   cd wine-ml-app
2. **Crie um ambiente virtual (opcional, mas recomendado):**
- Linux
   ```bash
   python -m venv .venv
   source .venv/bin/activate
- Windows
   ```bash
   python -m venv .venv
   source .venv\Scripts\activate
3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
4. **Configure o token do Hugging Face (se necessário para datasets privados):**
   ```bash
   export HF_TOKEN=seu_token
   export HF_DATASET_REPO=henriquebap/wine-ml-dataset
   export HF_DATASET_FILENAME=WineQT.csv

5. **Estrutura sugerida:**
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
