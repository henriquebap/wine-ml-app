# 🚀 PLANO DE AÇÃO IMEDIATO - Wine ML Project

**Objetivo:** Corrigir problemas críticos em 2-3 dias

---

## 📋 RESUMO DOS PROBLEMAS CRÍTICOS

| # | Problema | Gravidade | Impacto | Tempo |
|---|----------|-----------|---------|-------|
| 1 | Inconsistência Classificação vs Regressão | 🔴 CRÍTICO | Invalida análise e modelagem | 4h |
| 2 | Data Leakage Severo | 🔴 CRÍTICO | Métricas infladas ~10-15% | 6h |
| 3 | Fluxo de Dados Desconectado | 🔴 CRÍTICO | Reprodutibilidade comprometida | 4h |
| 4 | Métricas Inadequadas | 🟠 ALTO | Avaliação enganosa | 3h |
| 5 | Validação Cruzada Inadequada | 🟠 ALTO | Comparações inválidas | 2h |
| 6 | Preprocessamento Inconsistente | 🟠 ALTO | Resultados não reproduzíveis | 3h |

**TOTAL ESTIMADO:** 22 horas (2.5 dias úteis)

---

## 📅 DIA 1: CORRIGIR PROBLEMAS CRÍTICOS (8h)

### Manhã (4h)

#### ✅ Tarefa 1.1: Definir Problema (1h)

**Decisão:** Usar **CLASSIFICAÇÃO** (recomendado)

**Ação:**
```bash
# 1. Atualizar README.md
```

Adicionar no topo do README:

```markdown
## 🎯 Definição do Problema

**Tipo:** Classificação Multi-classe  
**Objetivo:** Classificar vinhos em categorias de qualidade baseado em características físico-químicas

**Classes:**
1. **Baixa qualidade (3-4):** Vinhos de qualidade inferior (~4% do dataset)
2. **Média qualidade (5-6):** Vinhos de qualidade padrão (~83% do dataset)
3. **Alta qualidade (7-8):** Vinhos de qualidade superior (~13% do dataset)

**Justificativa:**
- ✅ Mais interpretável para decisões de negócio
- ✅ Alinha com necessidade de categorização (aceitar/rejeitar lotes)
- ✅ Facilita estratégias diferenciadas por categoria
- ✅ Análise EDA já foi feita para classificação

**Desafio:** Dataset altamente desbalanceado (83% classe majoritária)
```

**Validação:** ✓ README atualizado e commitado

---

#### ✅ Tarefa 1.2: Criar Módulo de Preprocessamento (2h)

**Ação:** Criar arquivo `src/preprocessing.py`

Copiar código da seção 6 da ANALISE_COMPLETA_DO_PROJETO.md (classe `WinePreprocessorAdvanced`).

**Versão simplificada para começar:**

```python
# src/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

class WinePreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessamento padronizado para Wine Quality
    Compatível com sklearn Pipeline
    """
    
    def __init__(
        self,
        feature_columns: List[str],
        scale: bool = True,
        clip_outliers: bool = True,
        clip_quantiles: tuple = (0.01, 0.99)
    ):
        self.feature_columns = feature_columns
        self.scale = scale
        self.clip_outliers = clip_outliers
        self.clip_quantiles = clip_quantiles
        
        # Atributos fitted
        self.scaler_ = StandardScaler() if scale else None
        self.clip_bounds_ = {}
    
    def fit(self, X, y=None):
        """Fit preprocessor (apenas em dados de treino!)"""
        X = pd.DataFrame(X, columns=self.feature_columns)
        
        # 1. Calcular bounds para clipping (se ativado)
        if self.clip_outliers:
            for col in self.feature_columns:
                q_low, q_high = self.clip_quantiles
                low = X[col].quantile(q_low)
                high = X[col].quantile(q_high)
                self.clip_bounds_[col] = (low, high)
        
        # 2. Fit scaler (após clipping)
        X_clipped = self._clip(X)
        if self.scaler_:
            self.scaler_.fit(X_clipped)
        
        return self
    
    def transform(self, X):
        """Transform features"""
        X = pd.DataFrame(X, columns=self.feature_columns)
        
        # 1. Clip outliers
        X_clipped = self._clip(X)
        
        # 2. Scale
        if self.scaler_:
            X_scaled = pd.DataFrame(
                self.scaler_.transform(X_clipped),
                columns=self.feature_columns,
                index=X.index
            )
            return X_scaled
        
        return X_clipped
    
    def _clip(self, X):
        """Aplica clipping de outliers"""
        if not self.clip_outliers:
            return X
        
        X_clipped = X.copy()
        for col in self.feature_columns:
            if col in self.clip_bounds_:
                low, high = self.clip_bounds_[col]
                X_clipped[col] = X_clipped[col].clip(lower=low, upper=high)
        
        return X_clipped

# Teste rápido
if __name__ == "__main__":
    from data_ingestion import load_wine_dataframe
    
    df = load_wine_dataframe()
    df = df.drop_duplicates()
    
    features = [c for c in df.columns if c != 'quality']
    
    X = df[features]
    y = df['quality']
    
    preprocessor = WinePreprocessor(feature_columns=features)
    preprocessor.fit(X[:800])
    X_transformed = preprocessor.transform(X[800:])
    
    print("✅ Preprocessor funcionando!")
    print(f"Shape original: {X.shape}")
    print(f"Shape transformado: {X_transformed.shape}")
```

**Testar:**
```bash
cd /Users/henriquebap/Pessoal/Personal\ -\ Projects/Wine_MLProject/wine-ml-app
source .venv311/bin/activate
python src/preprocessing.py
```

**Validação:** ✓ Script executa sem erros

---

#### ✅ Tarefa 1.3: Criar Módulo de Avaliação (1h)

**Ação:** Criar arquivo `src/evaluation.py`

Copiar código da seção 4 da ANALISE_COMPLETA_DO_PROJETO.md (classe `ModelEvaluator`).

**Testar:**
```bash
python -c "from src.evaluation import ModelEvaluator; print('✅ OK')"
```

---

### Tarde (4h)

#### ✅ Tarefa 1.4: Corrigir Data Leakage nos Notebooks (4h)

**Ação:** Reescrever notebooks 04, 05, 06, 07

**Template padrão (copiar para cada notebook):**

```python
# === CONFIGURAÇÃO ===
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data_ingestion import load_wine_dataframe
from src.preprocessing import WinePreprocessor
from src.evaluation import ModelEvaluator

# Config
FEATURES = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
]
RANDOM_STATE = 42

# === 1. CARREGAR DADOS BRUTOS ===
df = load_wine_dataframe(repo_id="henriquebap/wine-ml-dataset", filename="WineQT.csv")
df = df.drop_duplicates().reset_index(drop=True)

# Criar target para CLASSIFICAÇÃO
df['quality_class'] = df['quality'].apply(
    lambda x: 'Baixa (3-4)' if x <= 4 else 'Média (5-6)' if x <= 6 else 'Alta (7-8)'
)

X = df[FEATURES]
y = df['quality_class']

print(f"Dataset: {X.shape}")
print(f"Distribuição classes:\n{y.value_counts(normalize=True)}")

# === 2. SPLIT PRIMEIRO (ANTES de preprocessing!) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# === 3. CRIAR PIPELINE (SEM DATA LEAKAGE) ===
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('preprocessor', WinePreprocessor(feature_columns=FEATURES)),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
])

# === 4. TREINAR ===
pipeline.fit(X_train, y_train)

# === 5. AVALIAR ===
evaluator = ModelEvaluator(class_names=['Baixa (3-4)', 'Média (5-6)', 'Alta (7-8)'])

y_pred = pipeline.predict(X_test)
evaluator.evaluate_classification(
    y_test, y_pred,
    model_name='RandomForest_baseline',
    X_train=X_train, y_train=y_train
)

# === 6. VALIDAÇÃO CRUZADA (SEM LEAKAGE) ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1
)
print(f"\nValidação Cruzada (Balanced Accuracy): {scores.mean():.3f} ± {scores.std():.3f}")
```

**Aplicar em:**
- [ ] `notebooks/04_baseline_models.ipynb`
- [ ] `notebooks/05_advanced_models.ipynb`
- [ ] `notebooks/06_hyperparameter_tuning.ipynb`
- [ ] `notebooks/07_model_evaluation.ipynb`

**Para cada notebook:**
1. Fazer backup (copiar para `04_baseline_models_OLD.ipynb`)
2. Substituir o código de carregamento/preprocessamento pelo template
3. Ajustar modelos específicos de cada notebook
4. Executar célula por célula
5. Verificar que não há erros

**Validação:** ✓ Todos notebooks executam sem erros

---

## 📅 DIA 2: UNIFICAR FLUXO E MÉTRICAS (8h)

### Manhã (4h)

#### ✅ Tarefa 2.1: Atualizar Notebooks 02 e 03 (2h)

**Problema:** Notebooks 02 e 03 carregam dados processados (`df_capped.csv`), enquanto 04-07 carregam brutos.

**Ação:**

1. Atualizar Notebook 02 (Statistical Analysis):

```python
# REMOVER:
# df_path = hf_hub_download(repo_id='henriquebap/wine-ml-processed', filename='processed/full.csv')

# ADICIONAR:
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from src.data_ingestion import load_wine_dataframe
from src.preprocessing import WinePreprocessor

# Carregar dados brutos (mesmo que notebooks 04-07)
df = load_wine_dataframe(repo_id="henriquebap/wine-ml-dataset", filename="WineQT.csv")
df = df.drop_duplicates().reset_index(drop=True)

# Criar target
df['quality_class'] = df['quality'].apply(
    lambda x: 'Baixa (3-4)' if x <= 4 else 'Média (5-6)' if x <= 6 else 'Alta (7-8)'
)

# NÃO aplicar preprocessamento aqui (análise estatística deve ser em dados originais)
# Apenas salvar selected_features
selected_features = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'quality']

print('Usando', len(selected_features), 'features.')
df[selected_features + ['quality_class']].head()
```

2. Atualizar Notebook 03 (Visualization):
   - Mesma mudança que Notebook 02
   - Visualizações devem ser em dados originais (mais interpretável)

**Validação:** ✓ Notebooks 02 e 03 executam com dados brutos

---

#### ✅ Tarefa 2.2: Adicionar Comparação de Modelos (2h)

**Ação:** Criar nova seção no final de cada notebook de modelagem (04-07)

```python
# === COMPARAÇÃO DE TODOS OS MODELOS ===

# (Após treinar todos os modelos do notebook)

# Avaliar cada modelo
models_to_compare = [
    ('LinearRegression', pipeline_lr, y_pred_lr),
    ('Ridge', pipeline_ridge, y_pred_ridge),
    ('RandomForest', pipeline_rf, y_pred_rf),
]

evaluator = ModelEvaluator(class_names=['Baixa (3-4)', 'Média (5-6)', 'Alta (7-8)'])

for name, pipeline, y_pred in models_to_compare:
    evaluator.evaluate_classification(
        y_test, y_pred,
        model_name=name,
        X_train=X_train, y_train=y_train
    )

# Comparar todos
comparison = evaluator.compare_models()
display(comparison)
```

**Validação:** ✓ Comparação funciona e mostra tabela de resultados

---

### Tarde (4h)

#### ✅ Tarefa 2.3: Implementar Cross-Validation Correto (2h)

**Ação:** Criar `src/cross_validation.py`

Copiar código da seção 5 da ANALISE_COMPLETA_DO_PROJETO.md (classe `CVManager`).

**Simplificar para MVP:**

```python
# src/cross_validation.py

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple

class CVManager:
    """Gerencia Cross-Validation de forma consistente"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.folds = None
    
    def create_folds(self, X, y, save_path=None):
        """Cria folds estratificados"""
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
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.folds, f, indent=2)
            print(f"✅ Folds salvos em: {save_path}")
        
        return self.folds
    
    def load_folds(self, load_path):
        """Carrega folds salvos"""
        with open(load_path, 'r') as f:
            self.folds = json.load(f)
        print(f"✅ Folds carregados de: {load_path}")
        return self.folds
    
    def get_fold(self, fold_id) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna índices de um fold"""
        fold = self.folds[fold_id]
        return np.array(fold['train_idx']), np.array(fold['test_idx'])
```

**Usar no Notebook 01 (final):**

```python
# notebooks/01_exploratory_data_analysis.ipynb
# Adicionar no final

from src.cross_validation import CVManager

# Criar folds consistentes para TODOS os notebooks
cv_manager = CVManager(n_splits=5, random_state=42)
cv_manager.create_folds(
    X=df[num_cols_no_target],
    y=df['quality_class'],
    save_path='data/processed/cv_folds.json'
)
```

**Usar nos Notebooks 04-07:**

```python
# Carregar folds salvos
from src.cross_validation import CVManager

cv_manager = CVManager()
cv_manager.load_folds('../data/processed/cv_folds.json')

# Usar em CV manual (se necessário)
# for fold_id in range(5):
#     train_idx, test_idx = cv_manager.get_fold(fold_id)
#     ...
```

**Validação:** ✓ Folds são salvos e reutilizados

---

#### ✅ Tarefa 2.4: Atualizar Notebook 01 (EDA) (2h)

**Ação:** Revisar Notebook 01 para alinhamento completo

1. **Remover criação de df_log e df_capped:**
   - Comentar ou remover seções que criam essas versões
   - Manter apenas análise em dados originais

2. **Adicionar seção de preprocessamento comparativo (opcional):**
   ```python
   # === TESTE DE ESTRATÉGIAS DE PREPROCESSAMENTO ===
   
   from src.preprocessing import WinePreprocessor
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_score
   from sklearn.pipeline import Pipeline
   
   strategies = {
       'sem_preprocessing': None,
       'standardscaler': WinePreprocessor(feature_columns=num_cols_no_target, clip_outliers=False),
       'com_clip': WinePreprocessor(feature_columns=num_cols_no_target, clip_outliers=True),
   }
   
   results = []
   for name, prep in strategies.items():
       if prep is None:
           pipeline = RandomForestClassifier(random_state=42, class_weight='balanced')
       else:
           pipeline = Pipeline([
               ('preprocessor', prep),
               ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
           ])
       
       scores = cross_val_score(pipeline, X, y, cv=5, scoring='balanced_accuracy')
       results.append({
           'strategy': name,
           'mean_score': scores.mean(),
           'std_score': scores.std()
       })
       print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
   
   # Escolher a melhor
   best = max(results, key=lambda x: x['mean_score'])
   print(f"\n✅ Melhor estratégia: {best['strategy']} ({best['mean_score']:.3f})")
   ```

3. **Gerar CV folds no final (conforme Tarefa 2.3)**

**Validação:** ✓ Notebook 01 executa e gera folds

---

## 📅 DIA 3: CONECTAR APP E FINALIZAR (6h)

### Manhã (3h)

#### ✅ Tarefa 3.1: Criar Módulo de Modelo (1h)

**Ação:** Criar `src/model.py`

Copiar código da seção 7 da ANALISE_COMPLETA_DO_PROJETO.md (classe `WineQualityModel`).

**Validar:**
```bash
python -c "from src.model import WineQualityModel; print('✅ OK')"
```

---

#### ✅ Tarefa 3.2: Criar Notebook de Treinamento Final (1h)

**Ação:** Criar `notebooks/09_train_final_model.ipynb`

```python
# 09 - Train Final Model for Production

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
from src.data_ingestion import load_wine_dataframe
from src.preprocessing import WinePreprocessor
from src.model import WineQualityModel

# Config
FEATURES = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
]

# 1. Carregar TODOS os dados (sem split - modelo final)
df = load_wine_dataframe(repo_id="henriquebap/wine-ml-dataset", filename="WineQT.csv")
df = df.drop_duplicates().reset_index(drop=True)

df['quality_class'] = df['quality'].apply(
    lambda x: 'Baixa (3-4)' if x <= 4 else 'Média (5-6)' if x <= 6 else 'Alta (7-8)'
)

X = df[FEATURES]
y = df['quality_class']

print(f"Dataset completo: {X.shape}")
print(f"Distribuição:\n{y.value_counts(normalize=True)}")

# 2. Criar modelo com MELHORES hiperparâmetros (do notebook 05)
# AJUSTAR ESTES VALORES APÓS FINALIZAR NOTEBOOKS 04-07!
from sklearn.ensemble import RandomForestClassifier

preprocessor = WinePreprocessor(feature_columns=FEATURES)

model = WineQualityModel(
    model_type='random_forest',  # ou 'xgboost' se for melhor
    preprocessor=preprocessor,
    model_params={
        'n_estimators': 200,
        'max_depth': 25,
        'min_samples_split': 20,
        'min_samples_leaf': 8,
        'bootstrap': True,
        'max_features': None,
        'random_state': 42,
        'class_weight': 'balanced'
    }
)

# 3. Treinar
print("\n🔄 Treinando modelo final...")
model.fit(X, y)

# 4. Estimar performance com CV (apenas para referência)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

pipeline_for_cv = Pipeline([
    ('preprocessor', WinePreprocessor(feature_columns=FEATURES)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=20,
        min_samples_leaf=8,
        random_state=42,
        class_weight='balanced'
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline_for_cv, X, y, cv=cv, scoring='balanced_accuracy')
print(f"\n📊 Estimativa de Performance (CV):")
print(f"   Balanced Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# 5. Salvar modelo final
model_path = '../data/models/wine_model_best.pkl'
model.save(model_path)

print(f"\n✅ Modelo final salvo em: {model_path}")
print("   Pronto para uso em produção!")

# 6. Teste rápido
sample = X.iloc[0:1]
prediction = model.predict(sample)
print(f"\n🧪 Teste de predição:")
print(f"   Input: {sample.values[0][:3]}...")
print(f"   Output: {prediction[0]}")
```

**Executar:**
```bash
jupyter nbconvert --to notebook --execute notebooks/09_train_final_model.ipynb
```

**Validação:** ✓ Modelo salvo em `data/models/wine_model_best.pkl`

---

#### ✅ Tarefa 3.3: Atualizar App.py (1h)

**Ação:** Reescrever `app.py`

```python
# app.py - Versão Atualizada

import pandas as pd
import gradio as gr
from pathlib import Path

# Importar módulos do projeto
import sys
sys.path.append(str(Path(__file__).parent))

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

# Carregar modelo pré-treinado
model = None
model_info = ""

if Path(MODEL_PATH).exists():
    try:
        model = WineQualityModel.load(MODEL_PATH)
        model_info = f"""
✅ **Modelo carregado com sucesso!**
- Tipo: {model.model_type}
- Treinado em: {model.metadata.get('trained_at', 'N/A')}
- Amostras de treino: {model.metadata.get('training_samples', 'N/A')}
"""
        print(model_info)
    except Exception as e:
        print(f"⚠️ Erro ao carregar modelo: {e}")
        model = None
        model_info = "⚠️ Erro ao carregar modelo pré-treinado. Clique em 'Treinar' para criar um novo."
else:
    model_info = "⚠️ Nenhum modelo encontrado. Clique em 'Treinar' para criar um."
    print(model_info)

def train_model():
    """Treina (ou retreina) o modelo"""
    global model, model_info
    
    try:
        # 1. Carregar dados
        df = load_wine_dataframe(repo_id=HF_DATASET_REPO, filename=CSV_FILENAME)
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Criar target
        df['quality_class'] = df['quality'].apply(
            lambda x: 'Baixa (3-4)' if x <= 4 else 'Média (5-6)' if x <= 6 else 'Alta (7-8)'
        )
        
        X = df[feature_cols]
        y = df['quality_class']
        
        # 2. Criar e treinar modelo
        from src.preprocessing import WinePreprocessor
        from sklearn.ensemble import RandomForestClassifier
        
        preprocessor = WinePreprocessor(feature_columns=feature_cols)
        
        model = WineQualityModel(
            model_type='random_forest',
            preprocessor=preprocessor,
            model_params={
                'n_estimators': 200,
                'max_depth': 25,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        )
        
        model.fit(X, y)
        
        # 3. Salvar
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)
        
        model_info = f"""
✅ **Modelo treinado e salvo com sucesso!**
- Amostras: {len(df)}
- Classes: {y.nunique()}
- Tipo: {model.model_type}
- Data: {model.metadata['trained_at']}
"""
        
        return model_info
    
    except Exception as e:
        error_msg = f"❌ Erro ao treinar modelo: {str(e)}"
        print(error_msg)
        return error_msg

def predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    """Faz predição"""
    if model is None:
        return "❌ Erro: Modelo não carregado. Clique em 'Treinar Modelo' primeiro."
    
    try:
        # Criar DataFrame com input
        x = pd.DataFrame([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
        ]], columns=feature_cols)
        
        # Predizer
        pred_class = model.predict(x)[0]
        
        # Emoji por classe
        emoji_map = {
            'Baixa (3-4)': '🔴',
            'Média (5-6)': '🟡',
            'Alta (7-8)': '🟢'
        }
        
        emoji = emoji_map.get(pred_class, '⚪')
        
        result = f"""
{emoji} **Qualidade Prevista:** {pred_class}

**Interpretação:**
"""
        if pred_class == 'Baixa (3-4)':
            result += "Vinho de qualidade inferior. Considerar reprocessamento ou uso em produtos secundários."
        elif pred_class == 'Média (5-6)':
            result += "Vinho de qualidade padrão. Adequado para comercialização regular."
        else:
            result += "Vinho de qualidade superior. Candidato a produto premium ou edição especial."
        
        return result
    
    except Exception as e:
        return f"❌ Erro na predição: {str(e)}"

# Interface Gradio
with gr.Blocks(title="Wine Quality Classifier") as demo:
    gr.Markdown("# 🍷 Wine Quality Classifier")
    gr.Markdown("Classificação de vinhos em categorias de qualidade baseado em características físico-químicas")
    
    # Status do modelo
    with gr.Row():
        status_box = gr.Textbox(
            label="Status do Modelo",
            value=model_info,
            interactive=False,
            lines=5
        )
    
    # Botão treinar
    with gr.Row():
        btn_train = gr.Button("🔄 Treinar/Re-treinar Modelo", variant="secondary")
    
    btn_train.click(fn=train_model, outputs=status_box)
    
    gr.Markdown("---")
    gr.Markdown("## 📊 Fazer Predição")
    gr.Markdown("Insira as características do vinho:")
    
    # Inputs (organizados em 3 colunas)
    with gr.Row():
        with gr.Column():
            fixed_acidity = gr.Number(value=7.5, label="Fixed Acidity")
            volatile_acidity = gr.Number(value=0.5, label="Volatile Acidity")
            citric_acid = gr.Number(value=0.25, label="Citric Acid")
            residual_sugar = gr.Number(value=2.0, label="Residual Sugar")
        
        with gr.Column():
            chlorides = gr.Number(value=0.08, label="Chlorides")
            free_sd = gr.Number(value=15.0, label="Free Sulfur Dioxide")
            total_sd = gr.Number(value=50.0, label="Total Sulfur Dioxide")
            density = gr.Number(value=0.996, label="Density")
        
        with gr.Column():
            pH = gr.Number(value=3.3, label="pH")
            sulphates = gr.Number(value=0.6, label="Sulphates")
            alcohol = gr.Number(value=10.0, label="Alcohol")
    
    # Output
    output = gr.Textbox(label="Resultado da Predição", lines=5)
    
    # Botão predizer
    btn_pred = gr.Button("🔮 Predizer Qualidade", variant="primary")
    btn_pred.click(
        predict,
        inputs=[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                free_sd, total_sd, density, pH, sulphates, alcohol],
        outputs=output
    )
    
    gr.Markdown("---")
    gr.Markdown("""
### ℹ️ Sobre o Modelo

**Classes de Qualidade:**
- 🔴 **Baixa (3-4):** Vinhos de qualidade inferior
- 🟡 **Média (5-6):** Vinhos de qualidade padrão
- 🟢 **Alta (7-8):** Vinhos de qualidade superior

**Modelo:** RandomForest Classifier com class_weight='balanced'  
**Dataset:** WineQT (UCI Machine Learning Repository)
""")

# Iniciar
if __name__ == "__main__":
    demo.launch()
```

**Testar:**
```bash
python app.py
```

**Validação:** ✓ App abre, carrega modelo, faz predições

---

### Tarde (3h)

#### ✅ Tarefa 3.4: Atualizar README.md (1.5h)

**Ação:** Reescrever README com instruções completas

Estrutura sugerida:
```markdown
# 🍷 Wine Quality ML Project

## 🎯 Objetivo
Classificar vinhos em categorias de qualidade...

## 📊 Problema de Machine Learning
[Copiar da tarefa 1.1]

## 🏗️ Arquitetura do Projeto
```
wine-ml-app/
├── src/                          # Módulos reutilizáveis
│   ├── data_ingestion.py        # Carregamento de dados
│   ├── preprocessing.py          # Preprocessamento
│   ├── evaluation.py             # Avaliação de modelos
│   ├── cross_validation.py       # Gerenciamento de CV
│   └── model.py                  # Modelo final para produção
├── notebooks/                    # Análise e experimentação
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_data_visualization.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 09_train_final_model.ipynb
├── data/
│   ├── processed/                # Dados processados e folds
│   └── models/                   # Modelos salvos
├── app.py                        # Aplicação Gradio
└── requirements.txt
```

## 🚀 Setup

### Pré-requisitos
- Python 3.11+
- pip

### Instalação
```bash
# 1. Clonar repositório
git clone <repo>
cd wine-ml-app

# 2. Criar ambiente virtual
python -m venv .venv311
source .venv311/bin/activate  # macOS/Linux
# .venv311\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente (opcional)
cp .env.example .env
# Editar .env com seu HF_TOKEN (se necessário)
```

## 📓 Executar Notebooks

```bash
jupyter lab
# Abrir e executar notebooks na ordem (01 → 09)
```

**Ordem de execução:**
1. `01_exploratory_data_analysis.ipynb` - EDA e criação de CV folds
2. `02_statistical_analysis.ipynb` - Testes estatísticos
3. `03_data_visualization.ipynb` - Visualizações
4. `04_baseline_models.ipynb` - Modelos baseline
5. `05_advanced_models.ipynb` - Modelos avançados (XGBoost, GBR)
6. `06_hyperparameter_tuning.ipynb` - Otimização de hiperparâmetros
7. `07_model_evaluation.ipynb` - Avaliação final
8. `09_train_final_model.ipynb` - Treinar modelo para produção

## 🌐 Executar Aplicação

```bash
python app.py
# Abrir http://127.0.0.1:7860
```

## 📈 Resultados

[ATUALIZAR APÓS FINALIZAR NOTEBOOKS]

**Melhor Modelo:** RandomForest / XGBoost  
**Balanced Accuracy:** X.XXX ± X.XXX (CV 5-fold)  
**F1-Score (macro):** X.XXX

**Performance por Classe:**
- Baixa (3-4): XX%
- Média (5-6): XX%
- Alta (7-8): XX%

## 🔧 Decisões Técnicas

### Problema: Classificação vs Regressão
✅ **Escolhido:** Classificação Multi-classe  
**Motivo:** Mais interpretável para negócio e tomada de decisão

### Preprocessamento
✅ **Escolhido:** StandardScaler com clipping de outliers (quantis 0.01-0.99)  
**Motivo:** [Adicionar após testar no notebook 01]

### Tratamento de Desbalanceamento
✅ **Estratégia:** `class_weight='balanced'` nos classificadores  
**Alternativas testadas:** SMOTE, undersampling (se aplicável)

### Validação
✅ **Estratégia:** 5-fold Stratified Cross-Validation  
**Motivo:** Garante distribuição balanceada em cada fold

## 📚 Referências

- Dataset: [UCI Wine Quality](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- Hugging Face Dataset: `henriquebap/wine-ml-dataset`

## 👤 Autor

[Seu Nome]  
Tech Challenge - Fase 3  
Pós Tech - FIAP
```

**Validação:** ✓ README completo e atualizado

---

#### ✅ Tarefa 3.5: Criar Checklist de Entrega (0.5h)

**Ação:** Criar `CHECKLIST_ENTREGA.md`

```markdown
# ✅ CHECKLIST DE ENTREGA

## 🎯 Definição do Problema
- [ ] Objetivo está claro no README (classificação de 3 classes)
- [ ] Classes estão bem definidas e justificadas
- [ ] Desbalanceamento foi identificado e documentado

## 📊 Dados
- [ ] Dataset está acessível via Hugging Face
- [ ] Duplicatas foram removidas (1143 → 1018 linhas)
- [ ] Fluxo de dados está unificado (todos notebooks usam mesma fonte)

## 🔬 Análise Exploratória
- [ ] EDA está completo (notebook 01)
- [ ] Estatísticas descritivas estão presentes
- [ ] Visualizações são informativas
- [ ] Análise de correlação foi feita
- [ ] Outliers foram identificados e tratados

## 🧪 Análise Estatística
- [ ] Testes estatísticos apropriados (Kruskal-Wallis, Spearman)
- [ ] Análise por classe foi realizada
- [ ] Resultados foram salvos (CSV)

## 🎨 Visualizações
- [ ] Histogramas por classe
- [ ] Boxplots por classe
- [ ] Correlation heatmap
- [ ] PCA 2D
- [ ] Visualizações são legíveis e têm títulos

## ⚙️ Preprocessamento
- [ ] Módulo `src/preprocessing.py` criado
- [ ] Sem data leakage (split antes de preprocessing)
- [ ] Mesma estratégia em todos os notebooks
- [ ] Pipeline com sklearn implementado

## 🤖 Modelagem
- [ ] Baseline (DummyClassifier) foi calculado
- [ ] Múltiplos modelos foram testados (mínimo 3)
- [ ] Hiperparâmetros foram otimizados
- [ ] GridSearchCV ou RandomizedSearchCV foi usado
- [ ] Validação cruzada foi feita corretamente (estratificada, sem leakage)

## 📏 Avaliação
- [ ] Métricas apropriadas: Balanced Accuracy, F1-Score
- [ ] Confusion matrix foi analisada
- [ ] Performance por classe foi reportada
- [ ] Comparação com baseline está clara
- [ ] Tabela comparativa de modelos foi criada

## 🔄 Validação Cruzada
- [ ] Folds foram salvos (`cv_folds.json`)
- [ ] Mesmos folds usados em todos os notebooks
- [ ] Estratificação foi aplicada
- [ ] Sem data leakage

## 🌐 Aplicação
- [ ] App.py funciona
- [ ] App usa modelo final treinado (`.pkl`)
- [ ] Preprocessamento é consistente com treino
- [ ] Interface é clara e intuitiva
- [ ] Predições estão corretas

## 📝 Código
- [ ] Módulos `src/` estão completos e funcionais
- [ ] Código é modular e reutilizável
- [ ] Variáveis têm nomes descritivos
- [ ] Comentários estão presentes onde necessário
- [ ] Sem código duplicado

## 📚 Documentação
- [ ] README está completo
- [ ] Instruções de setup funcionam
- [ ] Estrutura do projeto está explicada
- [ ] Decisões técnicas estão documentadas
- [ ] Resultados estão reportados

## 🔁 Reprodutibilidade
- [ ] Random state fixado (42) em todos os lugares
- [ ] requirements.txt está atualizado
- [ ] Todos os notebooks executam do início ao fim sem erros
- [ ] Folds de CV são consistentes

## 📦 Entrega
- [ ] Repositório Git está limpo
- [ ] Commits têm mensagens descritivas
- [ ] .gitignore está configurado
- [ ] Não há arquivos sensíveis no repositório (tokens, senhas)
- [ ] README tem link para Hugging Face Spaces (se deployed)

## 🎓 Extras (Opcional)
- [ ] MLflow ou tracking de experimentos implementado
- [ ] Testes unitários criados
- [ ] CI/CD configurado
- [ ] Deploy no Hugging Face Spaces funciona
- [ ] Documentação API (se houver FastAPI)

---

**Data de Verificação:** ___/___/2025  
**Verificado por:** _______________
```

**Validação:** ✓ Checklist criado

---

#### ✅ Tarefa 3.6: Executar Todos os Notebooks e Verificar (1h)

**Ação:** Executar notebooks em ordem e verificar erros

```bash
cd notebooks

# Executar em ordem
jupyter nbconvert --to notebook --execute 01_exploratory_data_analysis.ipynb
jupyter nbconvert --to notebook --execute 02_statistical_analysis.ipynb
jupyter nbconvert --to notebook --execute 03_data_visualization.ipynb
jupyter nbconvert --to notebook --execute 04_baseline_models.ipynb
jupyter nbconvert --to notebook --execute 05_advanced_models.ipynb
jupyter nbconvert --to notebook --execute 06_hyperparameter_tuning.ipynb
jupyter nbconvert --to notebook --execute 07_model_evaluation.ipynb
jupyter nbconvert --to notebook --execute 09_train_final_model.ipynb

# Se algum falhar, corrigir e re-executar
```

**Validação:**
- [ ] Todos notebooks executam sem erros
- [ ] Modelo final foi salvo em `data/models/wine_model_best.pkl`
- [ ] Folds foram salvos em `data/processed/cv_folds.json`

---

## 📦 FINALIZAÇÃO

### ✅ Commit Final

```bash
# Adicionar todos os arquivos
git add .

# Commit
git commit -m "refactor: corrigir problemas críticos

- Definir problema como classificação multi-classe
- Corrigir data leakage (split antes de preprocessing)
- Unificar fluxo de dados (todos notebooks usam mesmos dados brutos)
- Implementar métricas adequadas (Balanced Accuracy, F1-Score)
- Criar módulos reutilizáveis (src/preprocessing.py, src/evaluation.py, src/model.py)
- Conectar app.py com modelo treinado nos notebooks
- Atualizar documentação (README, CHECKLIST)"

# Push
git push origin main
```

---

### ✅ Preencher CHECKLIST_ENTREGA.md

Ir item por item e marcar `[x]` o que foi completado.

---

### ✅ Atualizar README com Resultados Finais

Após executar todos os notebooks, atualizar seção "📈 Resultados" do README com:

- Melhor modelo
- Métricas (Balanced Accuracy, F1-Score)
- Performance por classe
- Gráficos (opcional: salvar confusion matrix e adicionar ao README)

---

## 🎉 PRONTO PARA ENTREGA!

Após completar todas as tarefas:

1. ✅ Todos os notebooks executam sem erros
2. ✅ App funciona e faz predições corretas
3. ✅ Documentação está completa
4. ✅ Código está limpo e modular
5. ✅ Checklist está preenchido
6. ✅ Git está atualizado

---

## 🆘 TROUBLESHOOTING

### Problema: Notebooks dão erro de import

**Solução:**
```python
# Adicionar no início de cada notebook
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
```

### Problema: App não carrega modelo

**Solução:**
1. Verificar se `data/models/wine_model_best.pkl` existe
2. Executar `notebooks/09_train_final_model.ipynb`
3. Verificar caminho do modelo em `app.py`

### Problema: CV folds não encontrados

**Solução:**
1. Executar notebook 01 até o final
2. Verificar se `data/processed/cv_folds.json` foi criado
3. Ajustar caminho relativo nos notebooks se necessário

### Problema: Métricas muito baixas

**Possíveis causas:**
1. Preprocessamento inadequado
2. Hiperparâmetros não otimizados
3. Class weight não configurado (desbalanceamento)
4. Features não informativas

**Solução:**
- Testar diferentes estratégias de preprocessamento (notebook 01)
- Aumentar espaço de busca do GridSearchCV
- Usar `class_weight='balanced'`
- Revisar feature engineering

---

## 📞 CONTATO

Se tiver dúvidas, revisar:
1. `ANALISE_COMPLETA_DO_PROJETO.md` (análise detalhada)
2. `PLANO_ACAO_IMEDIATO.md` (este documento)
3. Código de exemplo em cada seção

Boa sorte! 🚀
