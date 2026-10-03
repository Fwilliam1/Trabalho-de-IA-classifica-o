import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, cross_val_score, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification # Apenas para gerar dados de teste

# ==========================================
# 1. CARREGAR A BASE DE DADOS
# ==========================================
caminho_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Heart Failure Prediction Dataset", "heart.csv")
if os.path.exists(caminho_csv):
    df = pd.read_csv(caminho_csv)
    df = pd.get_dummies(df, drop_first=True, dtype=float)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
else:
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
# 6. AVALIAÇÃO E EXTRAÇÃO DE MÉTRICAS
# ==========================================
print("\n--- RESULTADOS DA CALIBRAÇÃO (KNN) ---")
print(f"Melhores Parâmetros: {grid_search.best_params_}")

# Desempenho ao longo dos 10 folds (Média e Desvio Padrão)
scores = cross_val_score(melhor_modelo, X, y, cv=kf, scoring='accuracy')
print(f"\nAcurácia por Fold (10 folds): {[round(s, 4) for s in scores]}")
print(f"Acurácia Média (10-Fold CV): {scores.mean():.4f}")
print(f"Desvio Padrão: {scores.std():.4f}")

# Previsões validadas pelo K-Fold para a Matriz de Confusão e Acurácia Global
y_pred_cv = cross_val_predict(melhor_modelo, X, y, cv=kf)
acuracia_global = accuracy_score(y, y_pred_cv)
print(f"Acurácia Global (cross_val_predict): {acuracia_global:.4f}")

cm = confusion_matrix(y, y_pred_cv)
print("\n--- MATRIZ DE CONFUSÃO (Texto) ---")
print(cm)

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y, y_pred_cv))

# ==========================================
# 7. GRÁFICOS: MATRIZ DE CONFUSÃO E CURVA DE APRENDIZAGEM
# ==========================================
# 7.1 Matriz de Confusão
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Matriz de Confusão - KNN (10-Fold CV)')
plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')
plt.tight_layout()

# 7.2 Curva de Aprendizagem (Learning Curve)
train_sizes, train_scores, val_scores = learning_curve(
    estimator=melhor_modelo,
    X=X,
    y=y,
    cv=kf,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(7, 5))
plt.plot(train_sizes, train_mean, 'o-', color='crimson', label='Treino')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='crimson')
plt.plot(train_sizes, val_mean, 'o-', color='forestgreen', label='Validação (10-Fold CV)')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='forestgreen')
plt.title('Curva de Aprendizagem - KNN')
plt.xlabel('Tamanho do Conjunto de Treino')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

plt.show()