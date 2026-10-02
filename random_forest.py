import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification # Apenas para gerar dados de teste

# ==========================================
# 1. CARREGAR A BASE DE DADOS
# ==========================================
# Para testar o código agora, estou gerando um dataset fictício com valores float e 15 atributos.
# No seu trabalho, você vai substituir as duas linhas abaixo pelo seu dataset (ex: pd.read_csv)
X, y = make_classification(n_samples=1000, n_features=15, n_classes=2, random_state=42)

# Exemplo de como você faria com o seu dataset:
# df = pd.read_csv("seu_dataset.csv")
# X = df.drop("coluna_alvo", axis=1)
# y = df["coluna_alvo"]

# ==========================================
# 2. CRIAR O PIPELINE (Normalização + Modelo)
# ==========================================
# A Random Forest, por ser baseada em árvores, não exige normalização. 
# Porém, incluímos o StandardScaler para cumprir a regra do seu trabalho e 
# manter o mesmo padrão que você usará para o KNN e SVM (onde ela é obrigatória).
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('rf', RandomForestClassifier(random_state=42))
])

# ==========================================
# 3. DEFINIR OS PARÂMETROS PARA CALIBRAÇÃO
# ==========================================
# O GridSearchCV vai testar todas as combinações possíveis abaixo
param_grid = {
    'rf__n_estimators': [50, 100, 200],      # Quantidade de árvores
    'rf__max_depth': [None, 10, 20],         # Profundidade máxima das árvores
    'rf__min_samples_split': [2, 5, 10]      # Mínimo de amostras para dividir um nó
}

# ==========================================
# 4. CONFIGURAR K-FOLD E GRID SEARCH
# ==========================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid, 
    cv=kf, 
    scoring='accuracy',
    n_jobs=-1 # Usa todos os núcleos do processador para acelerar
)

# ==========================================
# 5. TREINAR E CALIBRAR
# ==========================================
print("Iniciando a calibração com K-Fold=10... Aguarde.")
grid_search.fit(X, y)

# Pega o melhor modelo encontrado pela calibração
melhor_modelo = grid_search.best_estimator_

# ==========================================
# 6. MÉTRICAS E RESULTADOS
# ==========================================
print("\n--- RESULTADOS DA CALIBRAÇÃO ---")
print(f"Melhores Parâmetros: {grid_search.best_params_}")
print(f"Acurácia Média (10-Fold): {grid_search.best_score_:.4f}")

# Para gerar uma única Matriz de Confusão robusta validada pelo K-Fold, 
# usamos o cross_val_predict. Ele faz as previsões rodando os 10 folds.
y_pred_cv = cross_val_predict(melhor_modelo, X, y, cv=kf)

cm = confusion_matrix(y, y_pred_cv)

print("\n--- MATRIZ DE CONFUSÃO (Texto) ---")
print(cm)

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y, y_pred_cv))

# ==========================================
# 7. PLOTAR A MATRIZ DE CONFUSÃO (Para o Relatório)
# ==========================================
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão - Random Forest (10-Fold CV)')
plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')
plt.tight_layout()
plt.show()