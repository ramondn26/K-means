# K‑Means Customizado e Comparação com Scikit‑Learn

Este repositório apresenta uma implementação do algoritmo K‑Means usando uma abordagem customizada (do zero) e sua comparação com a implementação disponível na biblioteca Scikit‑Learn, utilizando a base de dados Iris.

## O que o código faz

- Preparação dos dados  
  Carrega a base Iris (via load_iris() do Scikit‑Learn) e opcionalmente ignora a variável alvo.

- Clusterização “hardcore”  
  Implementa K‑Means do zero para k = 3 e k = 5:
    • Inicialização aleatória de centróides  
    • Atribuição de clusters por distância Euclidiana  
    • Atualização iterativa de centróides até convergência  
    • Cálculo do Silhouette Score e tempo de execução

- Clusterização com Scikit‑Learn  
  Roda KMeans(n_clusters=k) para os mesmos valores de k, calcula silhouette e mede tempo.

- Experimento com PCA  
  Para o melhor k (pelo Silhouette do Scikit‑Learn), aplica PCA com 1 e 2 componentes e plota:
    • Pontos coloridos por cluster  
    • Centrôides marcados no gráfico

- Análise Comparativa  
  Exibe tabela final comparando:
    • Silhouette Score  
    • Tempo de execução  
    • (Opcional) Diferença de rótulos entre as duas abordagens

## Pré‑requisitos

- Python 3.6+  
- Bibliotecas:
  numpy
  pandas
  matplotlib
  scikit-learn

> Observação: usa load_iris() do Scikit‑Learn; não é necessário CSV.  
> Para usar CSV local, ajuste:
> import pandas as pd  
> data = pd.read_csv("Iris.csv")  
> X = data.iloc[:, :-1].values

## Instalação

1. Clone:
   git clone https://github.com/ramondn26/K-means 
   cd K-means

2. (Opcional) Ambiente virtual:
   python3 -m venv venv  
   source venv/bin/activate  # Linux/macOS  
   venv\Scripts\activate     # Windows

3. Instale:
   pip install numpy pandas matplotlib scikit-learn

## Execução

python kmeans.py

O script exibirá:
- Silhouette e tempo para customizado (k=3,5)  
- Silhouette e tempo para Scikit‑Learn  
- Tabela comparativa final  
- Gráficos PCA (1 e 2 comp.)

## Estrutura de Diretórios

README.md  
kmeans.py

## Observações

- Implementação manual serve para aprendizado; Scikit‑Learn é mais otimizado para produção.  
- k ideal (pelo Silhouette) costuma ser 3 na Iris.  
- Pode estender com métricas ARI, NMI, análise de memória ou pureza dos clusters.
