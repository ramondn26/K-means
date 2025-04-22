import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import cm

plt.style.use('seaborn-v0_8')
np.random.seed(42)

# ----------------------------------------------
# 1. Carregamento do Dataset Iris
# ----------------------------------------------

dados_iris = datasets.load_iris()
X = dados_iris.data
y = dados_iris.target
nomes_classes = dados_iris.target_names

print(f"\nDataset Iris carregado com {X.shape[0]} amostras e {X.shape[1]} características")
print("Classes disponíveis:", nomes_classes)

# -------------------------------------------
# 2. Implementação do K-Means
# -------------------------------------------

def inicializar_centroides(X, k):
    """Seleciona k pontos aleatórios como centroides iniciais"""
    indices = np.random.permutation(len(X))[:k]
    return X[indices].copy()

def atribuir_clusters(X, centroides):
    """Atribui cada ponto ao cluster mais próximo"""
    distancias = np.sqrt(((X[:, np.newaxis] - centroides)**2).sum(axis=2))
    return np.argmin(distancias, axis=1)

def atualizar_centroides(X, rotulos, k):
    """Recalcula centroides como médias dos pontos do cluster"""
    return np.array([X[rotulos == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iter=100, tol=1e-4):
    """Implementação do algoritmo K-Means"""
    centroides = inicializar_centroides(X, k)
    rotulos_antigos = np.zeros(len(X))
    
    for iteracao in range(max_iter):
        rotulos = atribuir_clusters(X, centroides)
        novos_centroides = atualizar_centroides(X, rotulos, k)
        
        if np.allclose(centroides, novos_centroides, atol=tol):
            print(f"Convergência atingida na iteração {iteracao}")
            break
            
        centroides = novos_centroides
        
        if np.array_equal(rotulos, rotulos_antigos):
            break
        rotulos_antigos = rotulos.copy()
    
    return rotulos, centroides

valores_k = [3, 5]
resultados_manual = {}

print("\nTestando implementação do K-Means:")
for k in valores_k:
    inicio = perf_counter()
    rotulos, centroides = kmeans(X, k)
    tempo = perf_counter() - inicio
    silhueta = silhouette_score(X, rotulos)
    resultados_manual[k] = {
        'rotulos': rotulos,
        'centroides': centroides,
        'tempo': tempo,
        'silhueta': silhueta
    }
    print(f"k={k}: Score de Silhueta={silhueta:.3f}, Tempo={tempo:.4f}s")

# -------------------------------------------
# 3. Comparação com Scikit-Learn
# -------------------------------------------

resultados_sklearn = {}

print("\nComparando com a implementação do Scikit-Learn:")
for k in valores_k:
    inicio = perf_counter()
    modelo = KMeans(n_clusters=k, init='random', n_init=1, random_state=42)
    rotulos = modelo.fit_predict(X)
    tempo = perf_counter() - inicio
    silhueta = silhouette_score(X, rotulos)
    resultados_sklearn[k] = {
        'rotulos': rotulos,
        'centroides': modelo.cluster_centers_,
        'tempo': tempo,
        'silhueta': silhueta
    }
    print(f"k={k}: Score de Silhueta={silhueta:.3f}, Tempo={tempo:.4f}s")

# -------------------------------------------
# 4. Análise das Diferenças
# -------------------------------------------

print("\nDiferenças entre as implementações:")
for k in valores_k:
    diff_rotulos = (resultados_manual[k]['rotulos'] != resultados_sklearn[k]['rotulos']).mean()
    print(f"\nPara k={k}:")
    print(f"- Diferença nos rótulos: {diff_rotulos:.1%}")
    print(f"- Silhueta (Manual): {resultados_manual[k]['silhueta']:.3f}")
    print(f"- Silhueta (Sklearn): {resultados_sklearn[k]['silhueta']:.3f}")
    print(f"- Tempo (Manual): {resultados_manual[k]['tempo']:.4f}s")
    print(f"- Tempo (Sklearn): {resultados_sklearn[k]['tempo']:.4f}s")

melhor_k = max(valores_k, key=lambda x: resultados_sklearn[x]['silhueta'])
print(f"\nMelhor número de clusters (k) baseado na silhueta: {melhor_k}")

# -------------------------------------------
# 5. Visualização dos Clusters
# -------------------------------------------

def plot_clusters(X_red, rotulos, centroides, titulo):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_red[:, 0], X_red[:, 1] if X_red.shape[1] > 1 else np.zeros_like(X_red[:, 0]),
                         c=rotulos, cmap='viridis', alpha=0.7, edgecolor='k')
    plt.scatter(centroides[:, 0], centroides[:, 1] if centroides.shape[1] > 1 else np.zeros_like(centroides[:, 0]),
                marker='X', s=200, c='red', label='Centroides')
    plt.title(titulo)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2' if X_red.shape[1] > 1 else '')
    plt.legend()
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()

print("\nVisualizando os clusters com PCA:")
for n in [1, 2]:
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    
    modelo = KMeans(n_clusters=melhor_k, random_state=42)
    rotulos_pca = modelo.fit_predict(X_pca)
    
    plot_clusters(X_pca, rotulos_pca, modelo.cluster_centers_,
                 f"Clusters com PCA ({n} componente{'s' if n > 1 else ''})\nk={melhor_k}")