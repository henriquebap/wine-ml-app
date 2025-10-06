# 📊 ANÁLISE COMPLETA E ROBUSTA DO PROJETO WINE ML

**Data:** 06 de Outubro de 2025  
**Projeto:** Wine Quality ML - Tech Challenge Fase 3  
**Objetivo:** Identificar inconsistências, problemas metodológicos e propor correções

---

## 🎯 SUMÁRIO EXECUTIVO

### ⚠️ PROBLEMAS CRÍTICOS IDENTIFICADOS

1. **Inconsistência Fundamental de Objetivos** (CRÍTICO)
2. **Data Leakage Severo** (CRÍTICO)
3. **Desconexão no Fluxo de Dados** (CRÍTICO)
4. **Métricas Inadequadas para o Problema** (ALTO)
5. **Validação Cruzada Inadequada** (ALTO)
6. **Preprocessamento Inconsistente** (ALTO)
7. **Desconexão entre Notebooks e Aplicação** (MÉDIO)
8. **Falta de Rastreabilidade** (MÉDIO)

---

## 📋 ANÁLISE DETALHADA POR PROBLEMA

### 1. ⛔ INCONSISTÊNCIA FUNDAMENTAL DE OBJETIVOS (CRÍTICO)

#### 🔍 Descrição do Problema

O projeto apresenta uma **contradição fundamental** na definição do problema de Machine Learning:

**Notebook 01 (EDA):**
```python
# Define explicitamente um problema de CLASSIFICAÇÃO
df['quality_class'] = df['quality'].apply(
    lambda x: 'Baixa (3-4)' if x <= 4 else 'Média (5-6)' if x <= 6 else 'Alta (7-8)'
)
```

- **Objetivo declarado:** Classificação multi-classe (3 categorias)
- **Classes:** Baixa (3-4), Média (5-6), Alta (7-8)
- **Justificativa:** "Mais interpretável para o negócio"
- **Análise estatística:** Kruskal-Wallis, Spearman ordinal

**Notebooks 04-07 (Modelos):**
```python
# Fazem REGRESSÃO, predizendo quality numérica
y = df["quality"]  # valores de 3 a 8
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RF": RandomForestRegressor(n_estimators=300, random_state=42),
}
```

- **Objetivo real:** Regressão contínua
- **Target:** `quality` numérica (3-8)
- **Métricas:** RMSE, MAPE

#### 💥 Impacto

1. **Incoerência científica:** Análise exploratória não se alinha com modelagem
2. **Perda de insights:** Análise estatística de classes não é aproveitada
3. **Métrica enganosa:** RMSE não reflete o desempenho real para classificação
4. **Avaliação inadequada:** Para avaliadores, falta clareza sobre o objetivo real

#### ✅ Recomendação

**ESCOLHER UMA ABORDAGEM:**

**Opção A - CLASSIFICAÇÃO (Recomendada):**
```python
# Motivos: Problema de negócio mais interpretável, decisões discretas
# - Vinhos são categorizados em faixas de qualidade
# - Tomada de decisão é discreta (aceitar/rejeitar lote)
# - Imbalance problem pode ser tratado com class_weight

# Modelagem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score

y = df['quality_class']
model = RandomForestClassifier(class_weight='balanced', random_state=42)
```

**Opção B - REGRESSÃO (Alternativa):**
```python
# Motivos: Aproveita granularidade dos scores (3-8)
# - Pode ser arredondado para predição categórica
# - Ordinalidade é preservada naturalmente

# Ajustar todo EDA para regressão:
# - Remover análise de classes
# - Focar em correlação linear (Pearson)
# - Métricas: RMSE, MAE, R²
```

**DECISÃO OBRIGATÓRIA:** Definir e documentar claramente no README.md

---

### 2. 🚨 DATA LEAKAGE SEVERO (CRÍTICO)

#### 🔍 Descrição do Problema

O projeto apresenta **vazamento de informação** do conjunto de teste para o treino em múltiplos pontos:

**Problema 1: Preprocessamento no Dataset Completo**

```python
# notebooks/04_baseline_models.ipynb (e outros)
df = load_wine_dataframe(repo_id=HF_REPO, filename=FILENAME)
pre = DataPreprocessor(feature_columns=FEATURES, target_column="quality")
df_p = pre.fit_transform(df)  # ❌ FIT no dataset completo!
X = df_p[FEATURES]
y = df_p["quality"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**O que acontece:**
1. `DataPreprocessor.fit_transform()` calcula:
   - Quantis de outliers (0.01, 0.99) em TODO o dataset
   - StandardScaler fitted em TODO o dataset
2. Depois divide em treino/teste
3. **Resultado:** O modelo "vê" estatísticas do conjunto de teste durante treino

**Problema 2: Clipping de Outliers com Dados de Teste**

```python
# src/data_processing.py
def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
    q_low, q_high = self.outlier_clip_quantiles  # (0.01, 0.99)
    features = self._get_feature_columns(df)
    for col in features:
        low = df[col].quantile(q_low)   # ❌ Usa TODO o dataframe
        high = df[col].quantile(q_high) # ❌ Usa TODO o dataframe
        df[col] = df[col].clip(lower=low, upper=high)
    return df
```

**Problema 3: Transformações no Notebook 01**

```python
# notebooks/01_exploratory_data_analysis.ipynb
# Cria df_log e df_capped em TODO o dataset
df_log = df.copy()
for col in high_skew:
    df_log[col] = np.log1p(df_log[col].clip(lower=0))

# Capping por CLASSE usando todo o dataset
df_capped = df_log.copy()
for col in cap_cols:
    for cls in order:
        mask = df_capped[target_col] == cls
        Q1, Q3 = df_capped.loc[mask, col].quantile(0.25), df_capped.loc[mask, col].quantile(0.75)
        # ❌ Calcula IQR com dados de treino E teste juntos
```

#### 💥 Impacto

1. **Métricas infladas:** Desempenho reportado é **otimista demais**
2. **Generalização comprometida:** Modelo não vai performar em produção
3. **Avaliação inválida:** Resultados não são confiáveis cientificamente
4. **Validação cruzada comprometida:** O CV também sofre do mesmo problema

#### 📊 Estimativa de Impacto

- **RMSE reportado:** ~0.70
- **RMSE real (sem leakage):** Provavelmente ~0.75-0.80
- **Diferença:** 7-14% de degradação esperada

#### ✅ Recomendação

**FLUXO CORRETO:**

```python
# 1. Carregar dados brutos
df_raw = load_wine_dataframe(repo_id=HF_REPO, filename=FILENAME)
df = df_raw.drop_duplicates().reset_index(drop=True)

# 2. SPLIT PRIMEIRO (antes de qualquer transformação)
X_raw = df[FEATURES]
y = df[target_column]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)

# 3. FIT preprocessor SOMENTE em treino
preprocessor = DataPreprocessor(
    feature_columns=FEATURES,
    outlier_clip_quantiles=(0.01, 0.99),
    scale_features=True
)
preprocessor.fit(pd.DataFrame(X_train_raw))  # ✅ FIT apenas em treino

# 4. TRANSFORM em treino e teste separadamente
X_train = preprocessor.transform(pd.DataFrame(X_train_raw))
X_test = preprocessor.transform(pd.DataFrame(X_test_raw))

# 5. Treinar modelo
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**IMPORTANTE:** O mesmo vale para validação cruzada!

```python
# Para CV, usar Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', DataPreprocessorSklearn()),  # Wrapper para sklearn
    ('model', RandomForestRegressor(random_state=42))
])

# Agora o CV faz fit/transform correto em cada fold
scores = cross_val_score(pipeline, X_raw, y, cv=5, scoring='neg_root_mean_squared_error')
```

---

### 3. 🔀 DESCONEXÃO NO FLUXO DE DADOS (CRÍTICO)

#### 🔍 Descrição do Problema

Existem **TRÊS versões diferentes dos dados** sendo usadas de forma inconsistente:

**Versão 1: Dados Brutos (`WineQT.csv`)**
- **Fonte:** `henriquebap/wine-ml-dataset`
- **Usado em:** Notebooks 04-07, app.py
- **Características:** 
  - 1143 linhas (com duplicatas)
  - 1018 linhas (sem duplicatas)
  - Features em escala original

**Versão 2: Dados Transformados (`df_log.csv`)**
- **Fonte:** Gerado no Notebook 01
- **Transformações:** log1p nas features com alto skew
- **Usado em:** Potencialmente em notebooks 02-03 (não explícito)

**Versão 3: Dados Capped (`df_capped.csv` → `processed/full.csv`)**
- **Fonte:** Gerado no Notebook 01, hospedado em `henriquebap/wine-ml-processed`
- **Transformações:** 
  - log1p nas features com alto skew
  - Winsorização (capping IQR) **por classe**
  - Features têm valores transformados (ex: fixed acidity = 2.128232 ao invés de ~8.4)
- **Usado em:** Notebooks 02 (Statistical Analysis), 03 (Visualization)

**Evidência do Problema:**

```python
# Notebook 02 - Statistical Analysis
df_path = hf_hub_download(
    repo_id='henriquebap/wine-ml-processed',
    filename='processed/full.csv',  # ← Dados TRANSFORMADOS
)
df = pd.read_csv(df_path)
# fixed acidity = 2.128232 (transformado!)
```

```python
# Notebook 04 - Baseline Models
df = load_wine_dataframe(repo_id=HF_REPO, filename=FILENAME)
pre = DataPreprocessor(feature_columns=FEATURES, target_column="quality")
df_p = pre.fit_transform(df)  # ← Aplica OUTRO preprocessamento (StandardScaler)
```

#### 💥 Impacto

1. **Incomparabilidade:** Análise estatística (NB02) vs Modelos (NB04-07) usam dados diferentes
2. **Reprodutibilidade quebrada:** Impossível replicar resultados
3. **Feature importance:** Não é comparável entre notebooks
4. **Insights perdidos:** Análise de correlação no NB02 não se aplica aos modelos

#### 📊 Exemplo do Problema

```
Notebook 02 (df_capped):
- fixed acidity: mean=2.174, std=0.123, range=[2.08, 2.50]
  → Transformado com log1p + capping

Notebook 04 (df_raw + StandardScaler):
- fixed acidity: mean=0.0, std=1.0, range=[-2.1, 2.3]
  → Padronizado com StandardScaler

❌ Completamente diferentes! Análise estatística não se aplica.
```

#### ✅ Recomendação

**UNIFICAR O FLUXO DE DADOS:**

**Opção A - Usar Dados Brutos em Todos os Notebooks (Recomendada):**

```python
# 1. TODOS os notebooks carregam dados brutos
df_raw = load_wine_dataframe(repo_id='henriquebap/wine-ml-dataset', filename='WineQT.csv')
df = df_raw.drop_duplicates().reset_index(drop=True)

# 2. Aplicar mesmas transformações em TODOS os notebooks
# Definir uma função centralizada:

def preprocess_features(df, features, fit=True, preprocessor=None):
    """
    Aplica transformações consistentes:
    - Remove duplicatas
    - Clip outliers (apenas se fit=True)
    - StandardScaler
    """
    if preprocessor is None:
        preprocessor = DataPreprocessor(
            feature_columns=features,
            outlier_clip_quantiles=(0.01, 0.99),
            scale_features=True
        )
    
    if fit:
        preprocessor.fit(df)
    
    return preprocessor.transform(df), preprocessor

# 3. Documentar no README qual versão usar
```

**Opção B - Usar Dados Transformados Consistentemente:**

Se a transformação log1p + capping é realmente benéfica:

1. Aplicar em TODOS os notebooks
2. Salvar `preprocessor.pkl` com o transformador fitted
3. Usar o mesmo em app.py
4. Documentar claramente as transformações

**CRIAR UM MÓDULO CENTRALIZADO:**

```python
# src/preprocessing.py
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

class WinePreprocessor:
    """Preprocessamento padronizado para todos os notebooks e app"""
    
    def __init__(self, transform_type='standard'):
        """
        transform_type: 'standard', 'log', 'log_capped'
        """
        self.transform_type = transform_type
        self.scaler = None
        self.clip_bounds = {}
    
    def fit(self, X_train: pd.DataFrame) -> 'WinePreprocessor':
        # Implementar fit consistente
        pass
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Implementar transform consistente
        pass
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

# Uso em TODOS os notebooks:
from src.preprocessing import WinePreprocessor

preprocessor = WinePreprocessor(transform_type='standard')
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
```

---

### 4. 📏 MÉTRICAS INADEQUADAS PARA O PROBLEMA (ALTO)

#### 🔍 Descrição do Problema

As métricas usadas nos notebooks de modelagem não são apropriadas considerando:

1. **Natureza do problema:** Classificação vs Regressão (indefinido)
2. **Desbalanceamento:** 82.7% Média, 13.5% Alta, 3.8% Baixa
3. **Objetivo de negócio:** Não está claro

**Métricas Atuais (Notebooks 04-07):**

```python
# Apenas para regressão
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
```

**Problemas:**

1. **RMSE:** Penaliza igualmente erros em todas as classes
   - Erro de 5→4 tem mesmo peso que 7→8
   - Não considera importância relativa (ex: rejeitar vinho bom é pior que aceitar vinho médio)

2. **MAPE:** 
   - Sensível a valores pequenos (se quality fosse 0-1, seria problemático)
   - Não faz sentido para classificação

3. **Falta baseline:** Não compara com modelo dummy (sempre prediz a classe majoritária)

4. **Falta análise por classe:** Não mostra onde o modelo erra mais

#### 💥 Impacto

1. **Não detecta viés:** Modelo pode estar sempre predizendo "Média" e ter RMSE razoável
2. **Não reflete negócio:** Todos os erros têm mesmo custo?
3. **Comparação enganosa:** Comparar modelos por RMSE pode escolher modelo pior para negócio

#### ✅ Recomendação

**SE FOR CLASSIFICAÇÃO:**

```python
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Métricas principais
y_pred = model.predict(X_test)

# Balanced Accuracy (compensa desbalanceamento)
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {bal_acc:.3f}")

# F1-Score macro (média das F1 de cada classe)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"F1-Score (macro): {f1_macro:.3f}")

# 2. Relatório completo por classe
print(classification_report(y_test, y_pred, target_names=order))

# 3. Confusion Matrix (essencial!)
cm = confusion_matrix(y_test, y_pred, labels=order)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=order)
disp.plot()

# 4. Baseline (dummy classifier)
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_bal_acc = balanced_accuracy_score(y_test, dummy.predict(X_test))
print(f"Baseline (most frequent): {dummy_bal_acc:.3f}")
print(f"Improvement over baseline: {(bal_acc - dummy_bal_acc) / dummy_bal_acc * 100:.1f}%")
```

**SE FOR REGRESSÃO:**

Adicionar métricas complementares:

```python
from sklearn.metrics import mean_absolute_error, r2_score

# Métricas padrão
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Métricas customizadas para o contexto
def within_one_unit_accuracy(y_true, y_pred):
    """% de predições dentro de ±1 ponto de qualidade"""
    return (np.abs(y_true - y_pred) <= 1).mean()

# Análise por faixa de qualidade
def metrics_by_quality_range(y_true, y_pred):
    low = y_true <= 4
    medium = (y_true > 4) & (y_true <= 6)
    high = y_true > 6
    
    print(f"RMSE (Baixa 3-4): {root_mean_squared_error(y_true[low], y_pred[low]):.3f}")
    print(f"RMSE (Média 5-6): {root_mean_squared_error(y_true[medium], y_pred[medium]):.3f}")
    print(f"RMSE (Alta 7-8): {root_mean_squared_error(y_true[high], y_pred[high]):.3f}")
```

**CRIAR UM MÓDULO DE AVALIAÇÃO:**

```python
# src/evaluation.py

class ModelEvaluator:
    """Avaliação consistente para todos os notebooks"""
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.results = {}
    
    def evaluate(self, y_true, y_pred, model_name='model'):
        if self.problem_type == 'classification':
            return self._evaluate_classification(y_true, y_pred, model_name)
        else:
            return self._evaluate_regression(y_true, y_pred, model_name)
    
    def _evaluate_classification(self, y_true, y_pred, model_name):
        # Implementar métricas de classificação
        pass
    
    def _evaluate_regression(self, y_true, y_pred, model_name):
        # Implementar métricas de regressão
        pass
    
    def compare_models(self):
        # Comparar todos os modelos avaliados
        pass
    
    def plot_results(self):
        # Visualizar comparações
        pass

# Uso nos notebooks:
evaluator = ModelEvaluator(problem_type='classification')
results_rf = evaluator.evaluate(y_test, y_pred_rf, 'RandomForest')
results_xgb = evaluator.evaluate(y_test, y_pred_xgb, 'XGBoost')
evaluator.compare_models()
evaluator.plot_results()
```

---

### 5. 🔄 VALIDAÇÃO CRUZADA INADEQUADA (ALTO)

#### 🔍 Descrição do Problema

O projeto usa validação cruzada, mas com problemas:

**Problema 1: Não usa Estratificação Consistentemente**

```python
# Notebook 04
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # ❌ Não estratificado!
scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
```

Para classificação com desbalanceamento (82.7% / 13.5% / 3.8%), é **essencial** usar `StratifiedKFold`:

```python
# Correto para classificação
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Problema 2: CV é feito APÓS preprocessing de todo dataset**

```python
# Código atual
df_p = pre.fit_transform(df)  # ❌ Fit em TODO o dataset
X = df_p[FEATURES]
y = df_p["quality"]
scores = cross_val_score(model, X, y, cv=cv)  # Data leakage!
```

**Problema 3: Não salva os folds para reprodutibilidade**

O notebook 01 gera `stratified_folds.json`, mas os notebooks de modelagem não usam!

```python
# notebook 01 gera
folds = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr, te) in enumerate(skf.split(df_capped[num_cols_no_target], df_capped[target_col])):
    folds.append({'fold': fold, 'train_idx': tr.tolist(), 'test_idx': te.tolist()})

# notebooks 04-07 não usam! ❌
```

#### 💥 Impacto

1. **Folds inconsistentes:** Cada notebook usa folds diferentes
2. **Não reproduzível:** Impossível comparar modelos de notebooks diferentes
3. **Desbalanceamento nos folds:** Classes raras podem não aparecer em alguns folds
4. **Métricas não comparáveis:** Cada experimento usa splits diferentes

#### ✅ Recomendação

**SOLUÇÃO COMPLETA:**

```python
# 1. Criar módulo para gerenciar CV
# src/cross_validation.py

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple

class CVManager:
    """Gerencia folds de validação cruzada de forma consistente"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.folds = None
    
    def create_folds(self, X, y, save_path=None):
        """Cria folds estratificados e opcionalmente salva"""
        skf = StratifiedKFold(
            n_splits=self.n_splits, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        self.folds = []
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            self.folds.append({
                'fold': fold_id,
                'train_idx': train_idx.tolist(),
                'test_idx': test_idx.tolist()
            })
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self.folds, f, indent=2)
        
        return self.folds
    
    def load_folds(self, load_path):
        """Carrega folds salvos"""
        with open(load_path, 'r') as f:
            self.folds = json.load(f)
        return self.folds
    
    def get_fold(self, fold_id) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna índices de treino e teste para um fold"""
        fold = self.folds[fold_id]
        return np.array(fold['train_idx']), np.array(fold['test_idx'])
    
    def cross_validate(self, X, y, model, preprocessor=None):
        """
        Validação cruzada SEM DATA LEAKAGE
        Preprocessor é fitted em cada fold de treino separadamente
        """
        from sklearn.base import clone
        
        scores = []
        for fold_id in range(self.n_splits):
            train_idx, test_idx = self.get_fold(fold_id)
            
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            # Preprocessar (fit apenas em treino!)
            if preprocessor is not None:
                prep_fold = clone(preprocessor)
                prep_fold.fit(X_train_fold)
                X_train_fold = prep_fold.transform(X_train_fold)
                X_test_fold = prep_fold.transform(X_test_fold)
            
            # Treinar modelo
            model_fold = clone(model)
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Avaliar
            y_pred_fold = model_fold.predict(X_test_fold)
            # Calcular métricas desejadas
            # ...
            
        return scores

# 2. Usar em TODOS os notebooks

# Notebook 01 (criar folds)
cv_manager = CVManager(n_splits=5, random_state=42)
cv_manager.create_folds(
    X=df[num_cols_no_target],
    y=df[target_col],
    save_path='data/processed/cv_folds.json'
)

# Notebooks 04-07 (usar folds salvos)
cv_manager = CVManager()
cv_manager.load_folds('data/processed/cv_folds.json')

# Validação cruzada SEM leakage
scores = cv_manager.cross_validate(
    X=X_raw,  # Dados brutos!
    y=y,
    model=RandomForestRegressor(random_state=42),
    preprocessor=DataPreprocessor(...)  # Será fitted em cada fold
)
```

**BENEFÍCIOS:**

1. ✅ Folds idênticos em todos os notebooks
2. ✅ Estratificação automática
3. ✅ Sem data leakage
4. ✅ Reproduzível
5. ✅ Comparabilidade entre modelos

---

### 6. 🔧 PREPROCESSAMENTO INCONSISTENTE (ALTO)

#### 🔍 Descrição do Problema

Diferentes estratégias de preprocessamento são aplicadas sem documentação clara:

**Abordagem 1: Notebook 01 (EDA)**
```python
# Transformações:
# 1. Drop duplicates (1143 → 1018 linhas)
# 2. Log1p em features com alto skew
# 3. Winsorização (capping IQR) POR CLASSE
df_log = df.copy()
for col in high_skew:
    df_log[col] = np.log1p(df_log[col].clip(lower=0))

df_capped = df_log.copy()
for col in cap_cols:
    for cls in order:
        mask = df_capped[target_col] == cls
        Q1, Q3 = df_capped.loc[mask, col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        df_capped.loc[mask, col] = df_capped.loc[mask, col].clip(lower=lo, upper=hi)
```

**Abordagem 2: Notebooks 04-07 (Modelos)**
```python
# Transformações via DataPreprocessor:
# 1. Drop duplicates
# 2. Clip outliers (quantis 0.01, 0.99) - GLOBAL, não por classe
# 3. StandardScaler

pre = DataPreprocessor(
    feature_columns=FEATURES,
    outlier_clip_quantiles=(0.01, 0.99),  # Diferente da winsorização!
    scale_features=True  # StandardScaler
)
df_p = pre.fit_transform(df)
```

**Diferenças Críticas:**

| Aspecto | Notebook 01 (df_capped) | Notebooks 04-07 (DataPreprocessor) |
|---------|-------------------------|-------------------------------------|
| Transformação não-linear | ✅ log1p | ❌ Nenhuma |
| Outlier handling | Winsorização (IQR, por classe) | Clipping (quantis, global) |
| Scaling | ❌ Não aplica | ✅ StandardScaler |
| Features afetadas | Apenas com alto skew | Todas |

#### 💥 Impacto

1. **Feature engineering perdido:** Log1p pode melhorar performance, mas não é usado em modelos
2. **Inconsistência:** EDA sugere transformações que não são aplicadas
3. **Comparabilidade:** Impossível comparar análise estatística (NB02) com modelos (NB04-07)
4. **Reprodutibilidade:** Código do `DataPreprocessor` não reproduz transformações do EDA

#### ✅ Recomendação

**DECISÃO 1: Escolher UMA estratégia**

**Opção A - Usar Transformações do EDA (Mais sofisticada):**

```python
# src/preprocessing.py

class WinePreprocessorAdvanced:
    """
    Preprocessamento baseado no EDA (Notebook 01)
    - Log1p em features com alto skew
    - Winsorização por classe (se classificação)
    - StandardScaler
    """
    
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str = 'quality',
        high_skew_threshold: float = 1.0,
        use_log_transform: bool = True,
        use_winsorization: bool = True,
        winsor_method: str = 'iqr',  # 'iqr' ou 'quantile'
        by_class: bool = False  # Se True, winsoriza por classe (apenas para treino)
    ):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.high_skew_threshold = high_skew_threshold
        self.use_log_transform = use_log_transform
        self.use_winsorization = use_winsorization
        self.winsor_method = winsor_method
        self.by_class = by_class
        
        # Atributos fitted
        self.high_skew_cols = []
        self.clip_bounds = {}  # {col: (lower, upper)}
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'WinePreprocessorAdvanced':
        """
        Fit preprocessor APENAS em dados de treino
        
        Args:
            X: Features (DataFrame)
            y: Target (Series, necessário se by_class=True)
        """
        X = X.copy()
        
        # 1. Identificar features com alto skew
        if self.use_log_transform:
            skews = X[self.feature_columns].skew()
            self.high_skew_cols = skews[skews.abs() > self.high_skew_threshold].index.tolist()
            print(f"Features com alto skew: {self.high_skew_cols}")
        
        # 2. Aplicar log1p temporariamente para calcular bounds
        X_temp = X.copy()
        if self.use_log_transform:
            for col in self.high_skew_cols:
                X_temp[col] = np.log1p(X_temp[col].clip(lower=0))
        
        # 3. Calcular bounds para clipping
        if self.use_winsorization:
            if self.by_class and y is not None:
                # Winsorização por classe
                for col in self.feature_columns:
                    self.clip_bounds[col] = {}
                    for cls in y.unique():
                        mask = y == cls
                        vals = X_temp.loc[mask, col]
                        if self.winsor_method == 'iqr':
                            Q1, Q3 = vals.quantile([0.25, 0.75])
                            IQR = Q3 - Q1
                            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                        else:  # quantile
                            lo, hi = vals.quantile([0.01, 0.99])
                        self.clip_bounds[col][cls] = (lo, hi)
            else:
                # Winsorização global
                for col in self.feature_columns:
                    vals = X_temp[col]
                    if self.winsor_method == 'iqr':
                        Q1, Q3 = vals.quantile([0.25, 0.75])
                        IQR = Q3 - Q1
                        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    else:  # quantile
                        lo, hi = vals.quantile([0.01, 0.99])
                    self.clip_bounds[col] = (lo, hi)
        
        # 4. Transformar para fit do scaler
        X_transformed = self._transform_features(X, y=y)
        self.scaler.fit(X_transformed[self.feature_columns])
        
        return self
    
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transform features
        
        Args:
            X: Features (DataFrame)
            y: Target (Series, usado apenas se by_class=True)
        """
        X_transformed = self._transform_features(X, y=y)
        X_transformed[self.feature_columns] = self.scaler.transform(X_transformed[self.feature_columns])
        return X_transformed
    
    def _transform_features(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        X = X.copy()
        
        # 1. Log1p
        if self.use_log_transform:
            for col in self.high_skew_cols:
                X[col] = np.log1p(X[col].clip(lower=0))
        
        # 2. Clipping
        if self.use_winsorization:
            if self.by_class and y is not None:
                for col in self.feature_columns:
                    for cls in y.unique():
                        mask = y == cls
                        if cls in self.clip_bounds[col]:
                            lo, hi = self.clip_bounds[col][cls]
                            X.loc[mask, col] = X.loc[mask, col].clip(lower=lo, upper=hi)
            else:
                for col in self.feature_columns:
                    lo, hi = self.clip_bounds[col]
                    X[col] = X[col].clip(lower=lo, upper=hi)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)
```

**Opção B - Simplificar para StandardScaler básico:**

Se as transformações complexas não melhoram performance significativamente:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Usar apenas StandardScaler
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# Mais simples, mais robusto, mais fácil de manter
```

**DECISÃO 2: Documentar e Testar**

```python
# notebooks/XX_preprocessing_comparison.ipynb

# Testar diferentes estratégias de preprocessamento
strategies = {
    'none': None,
    'standard_scaler': StandardScaler(),
    'log_standard': WinePreprocessorAdvanced(use_log_transform=True, use_winsorization=False),
    'log_winsor': WinePreprocessorAdvanced(use_log_transform=True, use_winsorization=True),
    'log_winsor_by_class': WinePreprocessorAdvanced(use_log_transform=True, use_winsorization=True, by_class=True),
}

results = []
for name, prep in strategies.items():
    # Avaliar com CV
    score = evaluate_with_cv(X, y, model, preprocessor=prep)
    results.append({'strategy': name, 'score': score})

# Escolher a melhor estratégia
best_strategy = max(results, key=lambda x: x['score'])
print(f"Melhor estratégia: {best_strategy['strategy']}")
```

**RECOMENDAÇÃO FINAL:**

1. **Teste empírico:** Comparar as estratégias com validação cruzada correta
2. **Simplicidade:** Se diferença for < 2%, usar StandardScaler simples
3. **Documentar:** Explicar no README qual estratégia foi escolhida e por quê
4. **Consistência:** Usar MESMA estratégia em EDA, Statistical Analysis, Modelos e App

---

### 7. 🔌 DESCONEXÃO ENTRE NOTEBOOKS E APLICAÇÃO (MÉDIO)

#### 🔍 Descrição do Problema

O `app.py` (aplicação Gradio) está completamente desconectado dos notebooks de modelagem:

**Código atual do app.py:**

```python
def train():
    global model
    df = load_data()  # Carrega dados brutos
    X = df[feature_cols]
    y = df["quality"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)  # ❌ Hiperparâmetros não otimizados!
    model.fit(X, y)  # ❌ Sem preprocessamento!
    return f"Modelo treinado com {len(df)} linhas."
```

**Problemas:**

1. **Hiperparâmetros:** Usa `n_estimators=200`, mas Notebook 05 encontrou melhores parâmetros:
   ```python
   # Notebook 05 - XGBoost
   best_params = {
       'colsample_bytree': 1, 
       'learning_rate': 0.01, 
       'max_depth': 3, 
       'n_estimators': 400, 
       'subsample': 0.8
   }
   # RMSE: 0.7105
   
   # Notebook 04 - RandomForest
   best_params = {
       'bootstrap': True, 
       'max_depth': 25, 
       'max_features': None, 
       'min_samples_leaf': 8, 
       'min_samples_split': 20, 
       'n_estimators': 200
   }
   # RMSE: 0.7025
   ```

2. **Sem preprocessamento:** Treina direto nos dados brutos, mas notebooks usam `DataPreprocessor`

3. **Sem salvamento do modelo:** Treina a cada vez que app reinicia

4. **Sem versionamento:** Não há tracking de qual versão do modelo está em produção

#### 💥 Impacto

1. **Performance subótima:** App usa modelo pior que o encontrado nos notebooks
2. **Lentidão:** Treina a cada inicialização (desnecessário)
3. **Inconsistência:** Predições do app não refletem os experimentos
4. **Não reproduzível:** Não é possível rastrear versão do modelo

#### ✅ Recomendação

**CRIAR UM PIPELINE DE MODELO COMPLETO:**

```python
# src/model.py

import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class WineQualityModel:
    """
    Wrapper para modelo de qualidade de vinho
    Encapsula preprocessamento + modelo + metadata
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        preprocessor=None,
        model_params: Dict[str, Any] = None
    ):
        self.model_type = model_type
        self.preprocessor = preprocessor
        self.model_params = model_params or self._get_default_params()
        self.model = self._create_model()
        
        # Metadata
        self.metadata = {
            'model_type': model_type,
            'model_params': self.model_params,
            'trained_at': None,
            'training_samples': None,
            'features': None,
            'metrics': {}
        }
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Retorna os melhores hiperparâmetros encontrados nos notebooks"""
        if self.model_type == 'xgboost':
            return {
                'colsample_bytree': 1,
                'learning_rate': 0.01,
                'max_depth': 3,
                'n_estimators': 400,
                'subsample': 0.8,
                'random_state': 42
            }
        elif self.model_type == 'random_forest':
            return {
                'bootstrap': True,
                'max_depth': 25,
                'max_features': None,
                'min_samples_leaf': 8,
                'min_samples_split': 20,
                'n_estimators': 200,
                'random_state': 42
            }
        else:
            raise ValueError(f"model_type '{self.model_type}' não suportado")
    
    def _create_model(self):
        """Cria instância do modelo"""
        if self.model_type == 'xgboost':
            return XGBRegressor(**self.model_params)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(**self.model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Treina o modelo completo (preprocessor + model)
        
        Args:
            X: Features (dados brutos)
            y: Target
        """
        # 1. Preprocessar
        if self.preprocessor is not None:
            self.preprocessor.fit(X, y)
            X_processed = self.preprocessor.transform(X, y)
        else:
            X_processed = X
        
        # 2. Treinar modelo
        self.model.fit(X_processed, y)
        
        # 3. Atualizar metadata
        self.metadata['trained_at'] = datetime.now().isoformat()
        self.metadata['training_samples'] = len(X)
        self.metadata['features'] = X.columns.tolist()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predição
        
        Args:
            X: Features (dados brutos)
        
        Returns:
            Predições
        """
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        return self.model.predict(X_processed)
    
    def save(self, path: str):
        """
        Salva modelo completo (preprocessor + model + metadata)
        
        Args:
            path: Caminho para salvar (ex: 'models/wine_model_v1.pkl')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'metadata': self.metadata
        }
        joblib.dump(model_data, path)
        
        # Salvar metadata como JSON (para fácil inspeção)
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Modelo salvo em: {path}")
        print(f"✅ Metadata salvo em: {metadata_path}")
    
    @classmethod
    def load(cls, path: str) -> 'WineQualityModel':
        """
        Carrega modelo completo
        
        Args:
            path: Caminho do modelo salvo
        
        Returns:
            WineQualityModel
        """
        path = Path(path)
        model_data = joblib.load(path)
        
        # Reconstruir objeto
        instance = cls(
            model_type=model_data['model_type'],
            preprocessor=model_data['preprocessor'],
            model_params=model_data['model_params']
        )
        instance.model = model_data['model']
        instance.metadata = model_data['metadata']
        
        print(f"✅ Modelo carregado de: {path}")
        print(f"   Treinado em: {instance.metadata['trained_at']}")
        print(f"   Samples: {instance.metadata['training_samples']}")
        
        return instance
```

**ATUALIZAR APP.PY:**

```python
# app.py (versão atualizada)

import pandas as pd
import gradio as gr
from pathlib import Path
from src.model import WineQualityModel
from src.data_ingestion import load_wine_dataframe

# Configurações
HF_DATASET_REPO = "henriquebap/wine-ml-dataset"
CSV_FILENAME = "WineQT.csv"
MODEL_PATH = "data/models/wine_model_best.pkl"

feature_cols = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
]

# Carregar modelo pré-treinado (se existir)
model = None
if Path(MODEL_PATH).exists():
    model = WineQualityModel.load(MODEL_PATH)
    print("✅ Modelo pré-treinado carregado com sucesso!")
else:
    print("⚠️ Nenhum modelo pré-treinado encontrado. Clique em 'Treinar' para criar um.")

def train_model():
    """Treina (ou retreina) o modelo com os melhores hiperparâmetros"""
    global model
    
    # 1. Carregar dados
    df = load_wine_dataframe(repo_id=HF_DATASET_REPO, filename=CSV_FILENAME)
    df = df.drop_duplicates().reset_index(drop=True)
    
    X = df[feature_cols]
    y = df["quality"]
    
    # 2. Criar e treinar modelo com preprocessamento
    from src.preprocessing import WinePreprocessorAdvanced
    
    preprocessor = WinePreprocessorAdvanced(
        feature_columns=feature_cols,
        use_log_transform=True,
        use_winsorization=True
    )
    
    model = WineQualityModel(
        model_type='xgboost',  # Melhor modelo dos notebooks
        preprocessor=preprocessor
    )
    
    model.fit(X, y)
    
    # 3. Salvar modelo
    model.save(MODEL_PATH)
    
    return f"✅ Modelo treinado e salvo! ({len(df)} amostras)\nTipo: {model.model_type}\nData: {model.metadata['trained_at']}"

def predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    """Faz predição usando o modelo treinado"""
    if model is None:
        return "❌ Erro: Modelo não treinado. Clique em 'Treinar' primeiro."
    
    # Criar DataFrame com entrada
    x = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]], columns=feature_cols)
    
    # Predizer
    pred = float(model.predict(x)[0])
    pred_rounded = int(round(pred))
    
    # Classificar em categoria
    if pred_rounded <= 4:
        category = "Baixa qualidade (3-4)"
    elif pred_rounded <= 6:
        category = "Média qualidade (5-6)"
    else:
        category = "Alta qualidade (7-8)"
    
    return f"🍷 **Qualidade Prevista:** {pred:.2f} (arredondado: {pred_rounded})\n📊 **Categoria:** {category}"

# Interface Gradio (mantém o mesmo layout)
# ...
```

**CRIAR NOTEBOOK PARA TREINAR MODELO FINAL:**

```python
# notebooks/09_train_final_model.ipynb

# Este notebook treina o modelo FINAL que será usado em produção

# 1. Carregar dados completos (sem split - vamos usar tudo para treino)
df = load_wine_dataframe(...)
df = df.drop_duplicates()

X = df[FEATURES]
y = df['quality']

# 2. Criar modelo com os MELHORES hiperparâmetros encontrados nos experimentos
from src.model import WineQualityModel
from src.preprocessing import WinePreprocessorAdvanced

preprocessor = WinePreprocessorAdvanced(
    feature_columns=FEATURES,
    use_log_transform=True,
    use_winsorization=True
)

model = WineQualityModel(
    model_type='xgboost',  # Melhor modelo (ver notebook 05)
    preprocessor=preprocessor,
    model_params={
        'colsample_bytree': 1,
        'learning_rate': 0.01,
        'max_depth': 3,
        'n_estimators': 400,
        'subsample': 0.8,
        'random_state': 42
    }
)

# 3. Treinar
model.fit(X, y)

# 4. Avaliar com validação cruzada (para ter estimativa de performance)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(
    estimator=model,
    X=X,
    y=y,
    cv=5,
    scoring='neg_root_mean_squared_error'
)
print(f"RMSE médio (CV): {-scores.mean():.3f} ± {scores.std():.3f}")

# 5. Salvar modelo final
model.save('data/models/wine_model_best.pkl')
print("✅ Modelo final salvo e pronto para produção!")

# 6. Testar predição
sample = X.iloc[0:1]
prediction = model.predict(sample)
print(f"Teste de predição: {prediction[0]:.2f}")
```

**BENEFÍCIOS:**

1. ✅ App usa os melhores hiperparâmetros dos notebooks
2. ✅ Preprocessamento consistente entre notebooks e app
3. ✅ Modelo é salvo e versionado
4. ✅ Não precisa retreinar a cada inicialização
5. ✅ Fácil atualizar modelo (apenas treinar e salvar novamente)
6. ✅ Rastreabilidade (metadata salva com modelo)

---

### 8. 📝 FALTA DE RASTREABILIDADE (MÉDIO)

#### 🔍 Descrição do Problema

O projeto não tem sistema de rastreamento de experimentos:

**Problemas:**

1. **Notebooks sem versionamento de resultados:**
   - Cada execução sobrescreve resultados anteriores
   - Não há histórico de experimentos
   - Difícil comparar versões diferentes

2. **Falta de documentação dos experimentos:**
   - Não está claro qual experimento gerou qual resultado
   - Gráficos e tabelas não têm contexto (quando foram gerados, com quais dados)

3. **Sem tracking de modelos:**
   - Qual modelo foi o melhor?
   - Quais hiperparâmetros foram testados?
   - Qual versão está em produção?

4. **Dados processados sem metadata:**
   - `df_capped.csv` não tem informação de quando foi gerado, com quais parâmetros
   - Difícil reproduzir

#### ✅ Recomendação

**SOLUÇÃO 1: Adicionar MLflow (Recomendado)**

MLflow é um framework open-source para tracking de experimentos:

```python
# Instalar
pip install mlflow

# Usar em notebooks
import mlflow
import mlflow.sklearn

# Configurar experimento
mlflow.set_experiment("wine-quality-regression")

# Em cada notebook de modelagem (04-07)
with mlflow.start_run(run_name="RandomForest_baseline"):
    # Log parâmetros
    mlflow.log_params({
        'n_estimators': 200,
        'max_depth': 25,
        'model_type': 'RandomForest'
    })
    
    # Treinar modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Log métricas
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metrics({
        'rmse': rmse,
        'mae': mae,
        'r2': r2_score(y_test, y_pred)
    })
    
    # Log modelo
    mlflow.sklearn.log_model(model, "model")
    
    # Log artefatos (gráficos, etc)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    plt.savefig('predictions.png')
    mlflow.log_artifact('predictions.png')

# Visualizar experimentos
# mlflow ui
# Abrir http://localhost:5000
```

**SOLUÇÃO 2: Sistema simples de logging (Alternativa leve)**

Se MLflow for muito complexo:

```python
# src/experiment_tracker.py

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class ExperimentTracker:
    """Sistema simples para rastrear experimentos"""
    
    def __init__(self, log_dir: str = 'experiments'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'experiments.jsonl'
    
    def log_experiment(
        self,
        experiment_name: str,
        model_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        notes: str = ""
    ) -> str:
        """
        Log um experimento
        
        Returns:
            experiment_id (str)
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        log_entry = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'params': params,
            'metrics': metrics,
            'notes': notes
        }
        
        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"✅ Experimento logado: {experiment_id}")
        return experiment_id
    
    def get_all_experiments(self) -> pd.DataFrame:
        """Retorna todos os experimentos como DataFrame"""
        if not self.log_file.exists():
            return pd.DataFrame()
        
        experiments = []
        with open(self.log_file, 'r') as f:
            for line in f:
                experiments.append(json.loads(line))
        
        df = pd.DataFrame(experiments)
        return df
    
    def get_best_experiment(self, metric: str = 'rmse', ascending: bool = True) -> Dict[str, Any]:
        """Retorna o melhor experimento baseado em uma métrica"""
        df = self.get_all_experiments()
        if df.empty:
            return {}
        
        # Extrair métrica (está dentro de um dict)
        df['metric_value'] = df['metrics'].apply(lambda x: x.get(metric))
        df_sorted = df.sort_values('metric_value', ascending=ascending)
        
        best = df_sorted.iloc[0].to_dict()
        return best
    
    def compare_experiments(
        self,
        experiment_names: List[str] = None,
        metrics: List[str] = ['rmse', 'mae', 'r2']
    ) -> pd.DataFrame:
        """Compara experimentos"""
        df = self.get_all_experiments()
        
        if experiment_names:
            df = df[df['experiment_name'].isin(experiment_names)]
        
        # Expandir metrics
        for metric in metrics:
            df[metric] = df['metrics'].apply(lambda x: x.get(metric))
        
        comparison = df[['experiment_id', 'experiment_name', 'model_type', 'timestamp'] + metrics]
        return comparison.sort_values(metrics[0])

# Uso nos notebooks:

tracker = ExperimentTracker(log_dir='experiments')

# Experimento 1
tracker.log_experiment(
    experiment_name='baseline_randomforest',
    model_type='RandomForestRegressor',
    params={'n_estimators': 200, 'max_depth': 25},
    metrics={'rmse': 0.7025, 'mae': 0.53, 'r2': 0.32},
    notes='Modelo baseline com hiperparâmetros padrão'
)

# Experimento 2
tracker.log_experiment(
    experiment_name='tuned_xgboost',
    model_type='XGBRegressor',
    params={'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.01},
    metrics={'rmse': 0.7105, 'mae': 0.54, 'r2': 0.30},
    notes='XGBoost com hiperparâmetros otimizados via GridSearchCV'
)

# Comparar todos os experimentos
comparison = tracker.compare_experiments()
print(comparison)

# Encontrar melhor modelo
best = tracker.get_best_experiment(metric='rmse', ascending=True)
print(f"Melhor modelo: {best['experiment_name']} com RMSE={best['metrics']['rmse']}")
```

**BENEFÍCIOS:**

1. ✅ Histórico completo de experimentos
2. ✅ Fácil comparação de modelos
3. ✅ Reprodutibilidade (salva todos os parâmetros)
4. ✅ Rastreabilidade (sabe qual modelo está em produção)
5. ✅ Facilita relatório final (pode gerar tabelas automaticamente)

---

## 🎯 PLANO DE AÇÃO RECOMENDADO

### PRIORIDADE 1 - CRÍTICO (Fazer Imediatamente)

#### 1. Decidir: Classificação ou Regressão?

**Ação:**
1. Revisar o problema de negócio com stakeholders (ou usar bom senso)
2. **RECOMENDAÇÃO:** Escolher **CLASSIFICAÇÃO** (mais interpretável, alinha com EDA)
3. Documentar decisão no `README.md`
4. Atualizar todos os notebooks (04-07) para classificação

**Código:**
```python
# Adicionar no README.md
## 🎯 Problema de Machine Learning

**Tipo:** Classificação Multi-classe

**Objetivo:** Classificar vinhos em 3 categorias de qualidade baseado em características físico-químicas.

**Classes:**
- Baixa qualidade (3-4): Vinhos de qualidade inferior
- Média qualidade (5-6): Vinhos de qualidade padrão
- Alta qualidade (7-8): Vinhos de qualidade superior

**Justificativa:** 
- Mais interpretável para tomada de decisão no processo produtivo
- Alinha com necessidades do negócio (aceitar/rejeitar lotes)
- Facilita estratégias diferenciadas por categoria
```

#### 2. Corrigir Data Leakage

**Ação:**
1. Reescrever notebooks 04-07 para fazer split ANTES de preprocessing
2. Criar `src/preprocessing.py` com classe que funciona com sklearn Pipeline
3. Usar Pipeline em validação cruzada

**Código de exemplo no próximo item...**

#### 3. Unificar Fluxo de Dados

**Ação:**
1. Decidir qual estratégia de preprocessing usar (testar com CV)
2. Atualizar TODOS os notebooks para usar mesma estratégia
3. Criar módulo `src/preprocessing.py` centralizado
4. Atualizar notebooks 02-03 para usar mesmos dados que 04-07

**Template para atualizar notebooks:**
```python
# Template para TODOS os notebooks (01-07)

# === CONFIGURAÇÃO PADRÃO ===
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from src.data_ingestion import load_wine_dataframe
from src.preprocessing import WinePreprocessor
from src.evaluation import ModelEvaluator

# Configurações
HF_REPO = "henriquebap/wine-ml-dataset"
FILENAME = "WineQT.csv"
FEATURES = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
]
TARGET = "quality_class"  # ou "quality" se for regressão
RANDOM_STATE = 42

# === CARREGAMENTO DE DADOS ===
# 1. Carregar dados brutos
df_raw = load_wine_dataframe(repo_id=HF_REPO, filename=FILENAME)
df = df_raw.drop_duplicates().reset_index(drop=True)

# 2. Criar target (se for classificação)
if TARGET == "quality_class":
    df[TARGET] = df['quality'].apply(
        lambda x: 'Baixa (3-4)' if x <= 4 else 'Média (5-6)' if x <= 6 else 'Alta (7-8)'
    )

# 3. Separar X e y
X = df[FEATURES]
y = df[TARGET]

# === SPLIT (ANTES de qualquer preprocessing!) ===
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y if TARGET == "quality_class" else None,
    random_state=RANDOM_STATE
)

# === PREPROCESSING (FIT apenas em treino) ===
preprocessor = WinePreprocessor(
    feature_columns=FEATURES,
    transform_type='standard'  # ou 'log', 'log_capped'
)
preprocessor.fit(X_train, y_train)

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# === MODELAGEM ===
# Agora pode treinar modelos sem data leakage

# Para CV, usar Pipeline:
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=RANDOM_STATE))
])

# CV sem leakage
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring='balanced_accuracy'
)
print(f"Balanced Accuracy (CV): {scores.mean():.3f} ± {scores.std():.3f}")
```

---

### PRIORIDADE 2 - ALTO (Fazer logo após P1)

#### 4. Implementar Métricas Adequadas

**Ação:**
1. Criar `src/evaluation.py` com métricas apropriadas
2. Atualizar notebooks 04-07 para usar métricas corretas
3. Adicionar baseline (DummyClassifier) em todos os notebooks
4. Adicionar confusion matrix em todos os notebooks

**Código:**
```python
# src/evaluation.py

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier

class ModelEvaluator:
    """Avaliação consistente para problemas de classificação"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['Baixa (3-4)', 'Média (5-6)', 'Alta (7-8)']
        self.results = []
    
    def evaluate_classification(
        self,
        y_true,
        y_pred,
        model_name: str = 'model',
        X_train=None,
        y_train=None
    ) -> Dict[str, Any]:
        """
        Avalia modelo de classificação
        
        Args:
            y_true: True labels (test set)
            y_pred: Predicted labels (test set)
            model_name: Nome do modelo (para tracking)
            X_train: Features de treino (para calcular baseline)
            y_train: Labels de treino (para calcular baseline)
        
        Returns:
            Dict com métricas
        """
        # Métricas principais
        metrics = {
            'model_name': model_name,
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Comparar com baseline
        if X_train is not None and y_train is not None:
            dummy = DummyClassifier(strategy='most_frequent', random_state=42)
            dummy.fit(X_train, y_train)
            y_pred_dummy = dummy.predict(X_train.iloc[:len(y_true)])  # Simula teste
            
            baseline_bal_acc = balanced_accuracy_score(y_true, y_pred_dummy)
            metrics['baseline_balanced_accuracy'] = baseline_bal_acc
            metrics['improvement_over_baseline'] = (
                (metrics['balanced_accuracy'] - baseline_bal_acc) / baseline_bal_acc * 100
            )
        
        # Armazenar resultado
        self.results.append(metrics)
        
        # Print resumo
        print(f"\n{'='*60}")
        print(f"📊 AVALIAÇÃO: {model_name}")
        print(f"{'='*60}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"F1-Score (macro):  {metrics['f1_macro']:.3f}")
        print(f"F1-Score (weighted): {metrics['f1_weighted']:.3f}")
        
        if 'baseline_balanced_accuracy' in metrics:
            print(f"\nBaseline (most frequent): {metrics['baseline_balanced_accuracy']:.3f}")
            print(f"Improvement: {metrics['improvement_over_baseline']:.1f}%")
        
        print(f"\n{'='*60}")
        print("📋 CLASSIFICATION REPORT:")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_true, y_pred, model_name)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str = 'model'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.show()
        
        # Calcular % de acerto por classe
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"\n{'='*60}")
        print("📊 ACURÁCIA POR CLASSE:")
        print(f"{'='*60}")
        for i, class_name in enumerate(self.class_names):
            acc = cm_normalized[i, i]
            print(f"{class_name}: {acc:.1%}")
    
    def compare_models(self) -> pd.DataFrame:
        """Compara todos os modelos avaliados"""
        if not self.results:
            print("⚠️ Nenhum modelo avaliado ainda.")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        df_sorted = df.sort_values('balanced_accuracy', ascending=False)
        
        print(f"\n{'='*60}")
        print("🏆 COMPARAÇÃO DE MODELOS:")
        print(f"{'='*60}")
        print(df_sorted.to_string(index=False))
        
        # Plot comparação
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_sorted))
        width = 0.25
        
        ax.bar(x - width, df_sorted['balanced_accuracy'], width, label='Balanced Accuracy', color='skyblue')
        ax.bar(x, df_sorted['f1_macro'], width, label='F1-Score (macro)', color='lightcoral')
        ax.bar(x + width, df_sorted['f1_weighted'], width, label='F1-Score (weighted)', color='lightgreen')
        
        ax.set_ylabel('Score')
        ax.set_title('Comparação de Métricas por Modelo')
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['model_name'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return df_sorted

# Uso nos notebooks:

evaluator = ModelEvaluator(class_names=['Baixa (3-4)', 'Média (5-6)', 'Alta (7-8)'])

# Avaliar modelo 1
model_rf.fit(X_train_proc, y_train)
y_pred_rf = model_rf.predict(X_test_proc)
evaluator.evaluate_classification(
    y_test, y_pred_rf,
    model_name='RandomForest',
    X_train=X_train, y_train=y_train
)

# Avaliar modelo 2
model_xgb.fit(X_train_proc, y_train)
y_pred_xgb = model_xgb.predict(X_test_proc)
evaluator.evaluate_classification(
    y_test, y_pred_xgb,
    model_name='XGBoost',
    X_train=X_train, y_train=y_train
)

# Comparar todos
evaluator.compare_models()
```

#### 5. Implementar Validação Cruzada Correta

**Ação:**
1. Criar `src/cross_validation.py` (código já fornecido na seção 5)
2. Usar em TODOS os notebooks de modelagem
3. Salvar folds no notebook 01
4. Carregar mesmos folds em notebooks 04-07

#### 6. Definir Estratégia de Preprocessamento

**Ação:**
1. Criar notebook `10_preprocessing_comparison.ipynb`
2. Testar diferentes estratégias:
   - Nenhuma transformação
   - StandardScaler apenas
   - Log1p + StandardScaler
   - Log1p + Winsorização + StandardScaler
3. Usar CV correto para avaliar cada estratégia
4. Escolher a melhor (ou simplificar se diferença < 2%)
5. Documentar decisão

---

### PRIORIDADE 3 - MÉDIO (Fazer depois de P1 e P2)

#### 7. Conectar App aos Notebooks

**Ação:**
1. Criar `src/model.py` (código já fornecido na seção 7)
2. Criar notebook `09_train_final_model.ipynb` para treinar modelo final
3. Atualizar `app.py` para usar modelo salvo
4. Testar app localmente
5. Documentar no README como atualizar o modelo

#### 8. Implementar Tracking de Experimentos

**Ação:**
1. Escolher: MLflow (mais robusto) ou ExperimentTracker simples (código fornecido)
2. Adicionar tracking em notebooks 04-07
3. Comparar todos os experimentos
4. Documentar melhor modelo no README

---

## 📊 CHECKLIST FINAL PARA AVALIAÇÃO

Antes de entregar o projeto, verificar:

### ✅ Definição do Problema
- [ ] Objetivo está claramente definido (classificação ou regressão?)
- [ ] Classes/Target está bem documentado no README
- [ ] Justificativa da escolha está explicada

### ✅ Qualidade dos Dados
- [ ] Fluxo de dados está unificado (mesma fonte em todos notebooks)
- [ ] Transformações estão documentadas e consistentes
- [ ] Duplicatas foram tratadas adequadamente
- [ ] Outliers foram tratados (e método está documentado)

### ✅ Análise Exploratória
- [ ] EDA alinha com tipo de problema (classificação vs regressão)
- [ ] Visualizações são informativas e bem legendadas
- [ ] Análise estatística faz sentido para o problema
- [ ] Insights foram documentados

### ✅ Preprocessamento
- [ ] Mesma estratégia em todos os notebooks
- [ ] Sem data leakage (split antes de preprocessing)
- [ ] Código é modular e reutilizável (módulo src/preprocessing.py)
- [ ] Decisões de preprocessamento estão justificadas

### ✅ Modelagem
- [ ] Baseline foi calculado e reportado
- [ ] Múltiplos modelos foram comparados
- [ ] Validação cruzada foi feita corretamente (sem leakage, estratificada)
- [ ] Hiperparâmetros foram otimizados

### ✅ Avaliação
- [ ] Métricas são apropriadas para o problema
- [ ] Confusion matrix foi analisada (para classificação)
- [ ] Performance por classe foi reportada
- [ ] Comparação com baseline está clara

### ✅ Reprodutibilidade
- [ ] Random state fixado em todos os lugares (42)
- [ ] Requirements.txt atualizado
- [ ] Código é executável do início ao fim
- [ ] Folds de CV são salvos e reutilizados

### ✅ Aplicação
- [ ] App usa modelo treinado nos notebooks
- [ ] Preprocessamento é consistente com treino
- [ ] App funciona corretamente
- [ ] Interface é clara e intuitiva

### ✅ Documentação
- [ ] README é claro e completo
- [ ] Instruções de setup funcionam
- [ ] Estrutura do projeto está explicada
- [ ] Decisões importantes estão documentadas

### ✅ Organização
- [ ] Código está modular (src/ com módulos reutilizáveis)
- [ ] Notebooks têm narrativa clara
- [ ] Nomes de arquivos/variáveis são descritivos
- [ ] Git history é limpo (commits com mensagens claras)

---

## 🎓 CONCLUSÃO

Este projeto tem uma **base sólida**, mas apresenta **inconsistências críticas** que comprometem a validade científica e a avaliação. Os principais problemas são:

1. **Definição ambígua do problema** (classificação vs regressão)
2. **Data leakage severo** (métricas infladas artificialmente)
3. **Desconexão no fluxo de dados** (notebooks usam dados diferentes)

### Recomendação Final

**INVESTIR 2-3 DIAS** para corrigir os problemas críticos (Prioridade 1 e 2):

1. **Dia 1:** Definir problema, corrigir data leakage, unificar fluxo de dados
2. **Dia 2:** Implementar métricas corretas, CV adequado, testar preprocessing
3. **Dia 3:** Conectar app, documentar, fazer checklist final

**Com essas correções, o projeto terá:**
- ✅ Rigor científico
- ✅ Reprodutibilidade
- ✅ Rastreabilidade
- ✅ Clareza para avaliadores
- ✅ Código profissional e reutilizável

**IMPORTANTE:** Para um case avaliativo, a **qualidade metodológica** é tão importante quanto os resultados. É melhor ter um RMSE de 0.75 com metodologia correta do que 0.70 com data leakage.

Boa sorte! 🚀

---

**Documento gerado em:** 06/10/2025  
**Por:** Análise Completa do Projeto Wine ML  
**Versão:** 1.0
