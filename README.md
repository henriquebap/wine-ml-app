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

- 01 EDA: inspeção inicial, estatísticas, distribuições e correlações (alvo numérico `quality` com classes auxiliares apenas para visualização)
- 02 Estatística: testes não-paramétricos (Kruskal), correlação Spearman/Kendall
- 03 Visualização: histogramas, boxplots, scatter, heatmap
- 04-07 Modelagem: regressão em `quality` usando dados processados (`df_capped.csv`) e `selected_features` exportado de 01

## Dicionário das features do dataset

| Feature               | Significado (PT-BR)                                                                                                  |
|-----------------------|----------------------------------------------------------------------------------------------------------------------|
| fixed acidity         | Ácidos fixos (não voláteis) presentes no vinho; não evaporam facilmente e influenciam a acidez total e o sabor.     |
| volatile acidity      | Quantidade de ácido acético; em níveis altos pode gerar aroma/sabor avinagrado desagradável.                        |
| citric acid           | Ácido cítrico em pequenas quantidades; adiciona sensação de “frescor” e pode realçar o sabor do vinho.              |
| residual sugar        | Açúcar remanescente após a fermentação; é raro < 1 g/L; > 45 g/L geralmente classifica o vinho como doce.           |
| chlorides             | Teor de cloretos (sal) no vinho.                                                                                    |
| free sulfur dioxide   | SO₂ livre em equilíbrio entre forma molecular (gás dissolvido) e bissulfito; inibe microrganismos e oxidação.       |
| total sulfur dioxide  | SO₂ total (livre + combinado); em baixas concentrações é pouco perceptível; acima de ~50 ppm torna-se evidente.     |
| density               | Densidade do vinho; costuma ser próxima à da água, variando conforme teor de álcool (↓ densidade) e açúcar (↑).     |
| pH                    | Mede acidez/basicidade (0 = muito ácido, 14 = muito básico); vinhos geralmente ficam entre pH 3 e 4.                |
| sulphates             | Sulfatos usados como aditivos; podem contribuir para os níveis de SO₂, atuando como antimicrobianos/antioxidantes.  |
| alcohol               | Teor alcoólico (% de álcool em volume) do vinho.                                                                    |

## App Gradio

O `app.py` tenta carregar o modelo final salvo em `data/models/wine_quality_regressor.joblib` (exportado no notebook 07). Se não existir, faz fallback para treinar rapidamente um `RandomForestRegressor` usando o CSV do Hugging Face e permite inferência manual.

### Executar localmente

1. Crie e ative o ambiente
   ```bash
   python -m venv .venv311
   source .venv311/bin/activate
   pip install -r requirements.txt
   ```
2. (Opcional) Defina variáveis de ambiente para Hugging Face
   ```bash
   export HF_TOKEN=seu_token
   export HF_DATASET_REPO=henriquebap/wine-ml-dataset
   export HF_DATASET_FILENAME=WineQT.csv
   export HF_PROCESSED_REPO=henriquebap/wine-ml-processed
   ```
3. Execute os notebooks na ordem 01→07 para gerar `data/processed/df_capped.csv`, `reports/eda/selected_features.csv` e exportar o modelo em `data/models/`.
4. Rode o app
   ```bash
   python app.py
   ```

### Checklist de consistência

- [x] Objetivo alinhado: regressão sobre `quality` (classes apenas para visualização)
- [x] Dados processados unificados: notebooks 02-07 usam `df_capped.csv` (Hub/local)
- [x] Seleção de features: carrega `reports/eda/selected_features.csv` com fallback para numéricas
- [x] Export de modelo final: `data/models/wine_quality_regressor.joblib`
- [x] App conectado ao modelo final com fallback de treino

## Próximos passos

- Comparação de modelos, tuning e avaliação em notebooks 04-07
- Opcional: dashboard (Streamlit) e API (FastAPI)
