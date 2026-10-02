import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
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
# ATENÇÃO: Assim como no KNN, a normalização é OBRIGATÓRIA no SVM!
# O SVM tenta maximizar a margem (distância) entre as classes. Se as escalas 
# numéricas forem diferentes, o cálculo geométrico será distorcido.
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('svm', SVC(random_state=42))
])

# ==========================================
# 3. DEFINIR OS PARÂMETROS PARA CALIBRAÇÃO
# ==========================================
# O GridSearchCV vai testar as seguintes configurações do SVM:
param_grid = {
    'svm__C': [0.1, 1, 10, 100],               # Parâmetro de regularização (Margem suave vs rígida)
    'svm__kernel': ['linear', 'rbf'],          # Tipo de fronteira (linha reta vs fronteira curva/complexa)
    'svm__gamma': ['scale', 'auto']            # Coeficiente do kernel (relevante para o rbf)
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
print("Iniciando a calibração do SVM com K-Fold=10... Aguarde.")
grid_search.fit(X, y)

# Pega o melhor modelo encontrado pela calibração
melhor_modelo = grid_search.best_estimator_

# ==========================================
# 6. MÉTRICAS E RESULTADOS
# ==========================================
print("\n--- RESULTADOS DA CALIBRAÇÃO (SVM) ---")
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False) # Mudei para 'Oranges' para diferenciar
plt.title('Matriz de Confusão - SVM (10-Fold CV)')
plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')
plt.tight_layout()
plt.show()