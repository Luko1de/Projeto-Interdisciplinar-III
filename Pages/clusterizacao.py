import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns

# Título do App
st.title('Clusterização de Filmes')

# Explicação inicial
st.markdown("""
Para fazer o processo de clusterização do dataset foi utilizando o algoritmo K-Means.
O objetivo é identificar grupos de filmes semelhantes com base em suas características.
""")

# Carregar o dataset
st.markdown("### Carregando o Dataset")
df = pd.read_parquet(r"C:\Users\Aline\OneDrive - MSFT\UFRPE - OD\pisi3-2024.1\Projeto-Interdisciplinar-III - v03\utils\movies_cleaned.parquet")
st.write("Dataset carregado com sucesso! Exibindo as primeiras linhas:")
st.dataframe(df.head())

# Amostragem para facilitar a clusterização
st.markdown("### Amostragem")
df_amostra = df.sample(n=7000, random_state=42)
st.write(f"Foi selecionada uma amostra de {len(df_amostra)} filmes para facilitar o processo de clusterização.")

# Remover colunas desnecessárias
st.markdown("### Remoção de Colunas Desnecessárias")
df_amostra = df_amostra.drop(['poster_path', 'backdrop_path', 'production_companies', 'status'], axis=1)
st.write("As colunas irrelevantes para a clusterização foram removidas.")

# One-Hot Encoding para os gêneros
st.markdown("### Codificação dos Gêneros")
mlb = MultiLabelBinarizer()
generos_binarios = mlb.fit_transform(df_amostra['genres'].str.split('-'))
df_encoded = pd.DataFrame(generos_binarios, columns=mlb.classes_, index=df_amostra.index)
df_amostra = pd.concat([df_amostra, df_encoded], axis=1)
st.write("Os gêneros dos filmes foram codificados em uma forma numérica (One-Hot Encoding).")

# Selecionar e normalizar as colunas relevantes
st.markdown("### Normalização dos Dados")
features = df_amostra[['popularity', 'vote_average'] + list(mlb.classes_)]
scaler = StandardScaler()
features[['popularity', 'vote_average']] = scaler.fit_transform(features[['popularity', 'vote_average']])
st.write("As colunas 'popularity' e 'vote_average' foram normalizadas para facilitar a clusterização.")

# Determinar o número de clusters utilizando Elbow e Silhouette
st.markdown("### Determinando o Número Ideal de Clusters")
range_n_clusters = list(range(2, 11))
silhouette_avg = []

for num_clusters in range_n_clusters:
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    silhouette_avg.append(silhouette_score(features, cluster_labels))

plt.figure(figsize=(12, 6))
plt.plot(range_n_clusters, silhouette_avg, marker='o')
plt.xlabel('Número de Clusters', fontsize=14)
plt.ylabel('Silhouette Score', fontsize=14)
plt.title('Método da Silhueta para Seleção do Número de Clusters', fontsize=16)
plt.grid(True)
st.pyplot(plt)
st.write("O gráfico acima mostra o Método da Silhueta, que ajuda a identificar o número ideal de clusters.")

# Selecionar o melhor número de clusters
num_clusters = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]

# Criar e ajustar o modelo KMeans com o número de clusters ideal
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_amostra['cluster'] = kmeans.fit_predict(features)
st.write(f"O número ideal de clusters é {num_clusters}. Os filmes foram agrupados de acordo com este valor.")

# Visualizar os clusters usando Plotly
st.markdown("### Visualização dos Clusters")
fig = px.scatter(df_amostra, x='popularity', y='vote_average', color='cluster', title='Clusters de Filmes')
fig.update_layout(
    title='Clusters de Filmes',
    title_x=0.5,
    title_font_size=24,
    xaxis_title='Popularidade',
    yaxis_title='Média de Votos',
    xaxis_title_font_size=16,
    yaxis_title_font_size=16,
    font=dict(size=14)
)
st.plotly_chart(fig)
st.write("O gráfico acima mostra a distribuição dos filmes nos diferentes clusters.")

# Clusterização com as outras colunas numéricas e os gêneros
st.markdown("### Clusterização com Outras Colunas Numéricas")
colunas_numericas = ['popularity', 'vote_average', 'runtime']
dados_clusterizacao = np.concatenate((df_amostra[colunas_numericas], generos_binarios), axis=1)

inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(dados_clusterizacao)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Soma dos Quadrados Intra-Clusters')
st.pyplot(plt)
st.write("O gráfico do cotovelo ajuda a visualizar o ponto em que a soma dos quadrados intra-cluster começa a estabilizar, sugerindo o número ideal de clusters.")

silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(dados_clusterizacao)
    silhouette_scores.append(silhouette_score(dados_clusterizacao, labels))

plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Análise da Silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Coeficiente de Silhueta')
st.pyplot(plt)
st.write("O gráfico da silhueta fornece uma análise adicional sobre o número de clusters, similar ao gráfico anterior.")

# Criar o modelo KMeans com o número de clusters escolhido
st.markdown("### Resultados Finais da Clusterização")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_amostra['cluster'] = kmeans.fit_predict(dados_clusterizacao)

# Analisar e visualizar os clusters
st.write("Contagem de filmes por cluster:")
st.write(df_amostra.groupby('cluster')['title'].count())

plt.figure(figsize=(16, 12))
sns.scatterplot(x='runtime', y='vote_average', hue='cluster', data=df_amostra, palette='viridis', s=100)
plt.title('Clusters de Filmes')
plt.xlabel('Tempo de execução (Padronizado)')
plt.ylabel('Votação (Padronizada)')
st.pyplot(plt)
st.write("O gráfico final mostra a distribuição dos clusters com base em 'runtime' e 'vote_average'.")
