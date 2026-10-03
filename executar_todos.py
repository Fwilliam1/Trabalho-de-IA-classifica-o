import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def carregar_dados():
    """
    Carrega a base de dados do Heart Failure Prediction Dataset se existir,
    faz o pré-processamento necessário (encoding de variáveis categóricas)
    ou gera uma base sintética como fallback.
    """
    diretorio_base = os.path.dirname(os.path.abspath(__file__))
    caminhos_possiveis = [
        os.path.join(diretorio_base, "Heart Failure Prediction Dataset", "heart.csv"),
        os.path.join(diretorio_base, "heart.csv"),
        os.path.join("Heart Failure Prediction Dataset", "heart.csv"),
        "heart.csv"
    ]
    
    arquivo_encontrado = None
    for p in caminhos_possiveis:
        if os.path.exists(p):
            arquivo_encontrado = p
            break
            
    if arquivo_encontrado:
        print(f"[1/4] Carregando dataset: '{arquivo_encontrado}'...")
        df = pd.read_csv(arquivo_encontrado)
        # One-Hot Encoding para variáveis categóricas (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
        df_encoded = pd.get_dummies(df, drop_first=True, dtype=float)
        X = df_encoded.drop("HeartDisease", axis=1)
        y = df_encoded["HeartDisease"]
        print(f"      Dataset carregado: {X.shape[0]} instâncias e {X.shape[1]} atributos.")
        return X, y
    else:
        print("[1/4] 'heart.csv' não encontrado. Gerando dados de teste com make_classification...")
        X, y = make_classification(n_samples=1000, n_features=15, n_classes=2, random_state=42)
        return X, y


def definir_modelos():
    """
    Define as configurações, pipelines e grids de hiperparâmetros
    para os algoritmos KNN, Random Forest e SVM.
    """
    modelos = {
        'KNN': {
            'nome': 'K-Nearest Neighbors',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())
            ]),
            'param_grid': {
                'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['euclidean', 'manhattan']
            },
            'cmap': 'Greens',
            'color': 'forestgreen'
        },
        'Random Forest': {
            'nome': 'Random Forest',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(random_state=42))
            ]),
            'param_grid': {
                'rf__n_estimators': [50, 100, 200],
                'rf__max_depth': [None, 10, 20],
                'rf__min_samples_split': [2, 5, 10]
            },
            'cmap': 'Blues',
            'color': 'royalblue'
        },
        'SVM': {
            'nome': 'Support Vector Machine',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(random_state=42))
            ]),
            'param_grid': {
                'svm__C': [0.1, 1, 10, 100],
                'svm__kernel': ['linear', 'rbf'],
                'svm__gamma': ['scale', 'auto']
            },
            'cmap': 'Oranges',
            'color': 'darkorange'
        }
    }
    return modelos


def executar_avaliacoes(X, y, kf=None):
    """
    Executa o treinamento, calibração com GridSearchCV (10-Fold CV),
    extração de métricas e curvas de aprendizagem para todos os algoritmos.
    """
    if kf is None:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

    modelos = definir_modelos()
    resultados = {}

    print("\n[2/4] Iniciando treinamento e calibração dos 3 modelos (10-Fold CV)...")
    
    for chave, config in modelos.items():
        print(f"\n" + "=" * 55)
        print(f" Treinando e Calibrando: {config['nome']} ({chave})")
        print("=" * 55)

        # 1. Calibração de Hiperparâmetros via GridSearch
        grid = GridSearchCV(
            estimator=config['pipeline'],
            param_grid=config['param_grid'],
            cv=kf,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X, y)
        melhor_modelo = grid.best_estimator_

        # 2. Desempenho ao longo dos 10 folds (Média e Desvio Padrão)
        scores = cross_val_score(melhor_modelo, X, y, cv=kf, scoring='accuracy')

        # 3. Previsões e Matriz de Confusão
        y_pred = cross_val_predict(melhor_modelo, X, y, cv=kf)
        acc_global = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred)

        # 4. Curva de Aprendizagem (Learning Curve)
        print(f" -> Calculando Curva de Aprendizagem para {chave}...")
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=melhor_modelo,
            X=X,
            y=y,
            cv=kf,
            scoring='accuracy',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )

        resultados[chave] = {
            'nome': config['nome'],
            'melhor_modelo': melhor_modelo,
            'melhores_params': grid.best_params_,
            'scores': scores,
            'media_acuracia': scores.mean(),
            'desvio_padrao': scores.std(),
            'acuracia_global': acc_global,
            'y_pred': y_pred,
            'cm': cm,
            'report': report,
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'cmap': config['cmap'],
            'color': config['color']
        }

        # Exibição no console
        print(f" Melhores Parâmetros: {grid.best_params_}")
        print(f" Acurácia por Fold: {[round(s, 4) for s in scores]}")
        print(f" Acurácia Média: {scores.mean():.4f}")
        print(f" Desvio Padrão: {scores.std():.4f}")
        print(f" Acurácia Global (CV): {acc_global:.4f}")
        print("\n Matriz de Confusão:")
        print(cm)
        print("\n Relatório de Classificação:")
        print(report)

    return resultados


def exibir_tabela_resumo(resultados):
    """
    Exibe uma tabela comparativa com as métricas consolidadas de todos os modelos.
    """
    print("\n" + "=" * 65)
    print("               TABELA COMPARATIVA DE RESULTADOS")
    print("=" * 65)
    
    tabela = []
    for chave, r in resultados.items():
        tabela.append({
            'Algoritmo': r['nome'],
            'Acurácia Média (10-Fold)': f"{r['media_acuracia']:.4f}",
            'Desvio Padrão': f"{r['desvio_padrao']:.4f}",
            'Acurácia Global': f"{r['acuracia_global']:.4f}"
        })
    df_resumo = pd.DataFrame(tabela)
    print(df_resumo.to_string(index=False))
    print("=" * 65)


def gerar_graficos(resultados):
    """
    Gera as visualizações solicitadas:
    1. Comparativo Geral de Acurácia (Média e Desvio Padrão)
    2. Matrizes de Confusão comparativas lado a lado
    3. Curvas de Aprendizagem comparativas lado a lado
    """
    print("\n[3/4] Gerando gráficos comparativos...")
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------
    # Gráfico 1: Comparativo de Acurácia com Barra de Desvio Padrão
    # -------------------------------------------------------------
    modelos_nomes = [r['nome'] for r in resultados.values()]
    medias = [r['media_acuracia'] for r in resultados.values()]
    desvios = [r['desvio_padrao'] for r in resultados.values()]
    cores = [r['color'] for r in resultados.values()]

    plt.figure(figsize=(8, 5))
    barras = plt.bar(modelos_nomes, medias, yerr=desvios, capsize=8, color=cores, alpha=0.85, edgecolor='black')
    plt.title('Comparativo de Desempenho - Acurácia Média ± Desvio Padrão (10-Fold CV)', fontsize=12, fontweight='bold')
    plt.ylabel('Acurácia Média', fontsize=11)
    plt.ylim([max(0.0, min(medias) - 0.15), min(1.05, max(medias) + 0.1)])

    for barra, media, desvio in zip(barras, medias, desvios):
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2.0, yval + desvio + 0.01, f"{media:.4f} (±{desvio:.3f})", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # -------------------------------------------------------------
    # Gráfico 2: Matrizes de Confusão dos 3 Modelos Lado a Lado
    # -------------------------------------------------------------
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (chave, r) in zip(axes_cm, resultados.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap=r['cmap'], cbar=False, ax=ax,
                    annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title(f"Matriz de Confusão - {chave}", fontsize=12, fontweight='bold')
        ax.set_ylabel('Classe Real', fontsize=10)
        ax.set_xlabel('Classe Prevista', fontsize=10)
    fig_cm.suptitle('Matrizes de Confusão (10-Fold Cross-Validation)', fontsize=14, fontweight='bold', y=1.03)
    fig_cm.tight_layout()

    # -------------------------------------------------------------
    # Gráfico 3: Curvas de Aprendizagem dos 3 Modelos Lado a Lado
    # -------------------------------------------------------------
    fig_lc, axes_lc = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (chave, r) in zip(axes_lc, resultados.items()):
        train_mean = np.mean(r['train_scores'], axis=1)
        train_std = np.std(r['train_scores'], axis=1)
        val_mean = np.mean(r['val_scores'], axis=1)
        val_std = np.std(r['val_scores'], axis=1)

        ax.plot(r['train_sizes'], train_mean, 'o-', color='crimson', label='Treino', linewidth=2)
        ax.fill_between(r['train_sizes'], train_mean - train_std, train_mean + train_std, alpha=0.15, color='crimson')

        ax.plot(r['train_sizes'], val_mean, 'o-', color=r['color'], label='Validação (10-Fold)', linewidth=2)
        ax.fill_between(r['train_sizes'], val_mean - val_std, val_mean + val_std, alpha=0.15, color=r['color'])

        ax.set_title(f"Curva de Aprendizagem - {chave}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Tamanho do Conjunto de Treino', fontsize=10)
        ax.set_ylabel('Acurácia', fontsize=10)
        ax.legend(loc='lower right', frameon=True)
        ax.grid(True)

    fig_lc.suptitle('Curvas de Aprendizagem (Análise de Overfitting vs Underfitting)', fontsize=14, fontweight='bold', y=1.03)
    fig_lc.tight_layout()

    print("[4/4] Exibindo todos os gráficos na tela...")
    plt.show()


if __name__ == '__main__':
    # 1. Carrega dados
    X, y = carregar_dados()

    # 2. Executa calibração e avaliações simultaneamente para os 3 modelos
    resultados = executar_avaliacoes(X, y)

    # 3. Exibe tabela resumo no console
    exibir_tabela_resumo(resultados)

    # 4. Gera todos os gráficos
    gerar_graficos(resultados)
