import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification # Apenas para gerar dados de teste

# ==========================================
# 1. CARREGAR A BASE DE DADOS
# ==========================================
# Gerando dados de teste (substitua pelo seu pd.read_csv no trabalho real)
X, y = make_classification(n_samples=1000, n_features=15, n_classes=2, random_state=42)

# ==========================================
# 2. CRIAR O PIPELINE (Normalização + Modelo)
# ==========================================
# ATENÇÃO: Para o KNN, o StandardScaler é essencial! 
# Ele coloca todas as variáveis na mesma escala para o cálculo de distância funcionar.
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('knn', KNeighborsClassifier())
])

# ==========================================
# 3. DEFINIR OS PARÂMETROS PARA CALIBRAÇÃO
# ==========================================
# O GridSearchCV vai testar as seguintes configurações do KNN:
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15],  # Quantidade de vizinhos mais próximos (K)
    'knn__weights': ['uniform', 'distance'],   # Todos têm o mesmo peso vs vizinhos mais próximos pesam mais
    'knn__metric': ['euclidean', 'manhattan']  # Fórmula matemática para medir a distância
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
    n_jobs=-1 # Usa todos os núcleos do processador
)

# ==========================================
# 5. TREINAR E CALIBRAR
# ==========================================
print("Iniciando a calibração do KNN com K-Fold=10... Aguarde.")
grid_search.fit(X, y)

# Pega o melhor modelo encontrado pela calibração
melhor_modelo = grid_search.best_estimator_

# ==========================================
# 6. MÉTRICAS E RESULTADOS
# ==========================================
print("\n--- RESULTADOS DA CALIBRAÇÃO (KNN) ---")
print(f"Melhores Parâmetros: {grid_search.best_params_}")
print(f"Acurácia Média (10-Fold): {grid_search.best_score_:.4f}")

# Previsões validadas pelo K-Fold para gerar a Matriz de Confusão
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False) # Mudei para 'Greens' para diferenciar no relatório
plt.title('Matriz de Confusão - KNN (10-Fold CV)')
plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')
plt.tight_layout()
plt.show()