import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

# Título do aplicativo
st.title('Clusterização Interativa de Filmes e Análise de Silhueta')

# Carregar dados
@st.cache_data
def load_data():
    data = pd.read_csv('C:/Users/lucas/Projetos 3/movies_cleaned.csv')
    data_sample = data.sample(n=5000, random_state=42)
    return data_sample

df = load_data()

# Seleção de amostra
st.sidebar.header("Configurações da Amostra")
sample_size = st.sidebar.slider("Tamanho da amostra", min_value=1000, max_value=5000, value=1000, step=500)
df_sample = df.sample(n=sample_size, random_state=42)

# Seleção de colunas para a clusterização
st.sidebar.header("Seleção de Colunas")
all_columns = df_sample.select_dtypes(include=[np.number]).columns.tolist()
selected_columns = st.sidebar.multiselect("Escolha as colunas para o cluster", all_columns, default=all_columns)

# Normalizar os dados selecionados
@st.cache_data
def preprocess_data(df_sample, selected_columns):
    scaler = StandardScaler()
    df_selected = df_sample[selected_columns].fillna(0)  # Preenche valores ausentes com zero
    df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=selected_columns)
    return df_scaled

df_scaled = preprocess_data(df_sample, selected_columns)

# Aplicar KMeans
st.sidebar.header("Configurações do KMeans")
n_clusters = st.sidebar.slider("Número de Clusters", min_value=2, max_value=10, value=3)

@st.cache_data
def apply_kmeans(df_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_scaled['Cluster'] = kmeans.fit_predict(df_scaled)
    return df_scaled, kmeans

df_scaled, kmeans = apply_kmeans(df_scaled, n_clusters)

# Visualizar os clusters
st.subheader("Clusterização dos Dados")
fig = px.scatter_matrix(df_scaled, dimensions=selected_columns, color='Cluster',
                        title="Matriz de Dispersão dos Clusters",
                        labels={col: col for col in selected_columns})
st.plotly_chart(fig)

# Exibir centroides
st.subheader("Centroide dos Clusters")
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=selected_columns)
st.write(centroids)

# Avaliar a silhueta
st.subheader("Pontuação de Silhueta")
score = silhouette_score(df_scaled[selected_columns], df_scaled['Cluster'])
st.write(f"Pontuação de Silhueta: {score:.2f}")

# Gráfico de Silhueta Detalhado
st.subheader("Gráfico de Silhueta")
sample_silhouette_values = silhouette_samples(df_scaled[selected_columns], df_scaled['Cluster'])
fig, ax = plt.subplots(figsize=(12, 8))
y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[df_scaled['Cluster'] == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_title("Gráfico de Silhueta")
ax.set_xlabel("Coeficiente de Silhueta")
ax.set_ylabel("Cluster")
ax.axvline(x=score, color="red", linestyle="--")
ax.set_yticks([])
ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
st.pyplot(fig)

# Estatísticas por Cluster
st.subheader("Estatísticas por Cluster")
cluster_stats = df_sample.groupby(df_scaled['Cluster']).agg({
    'popularity': 'mean',
    'budget': 'mean',
    'revenue': 'mean',
    'runtime': 'mean',
    'vote_average': 'mean',
    'vote_count': 'mean'
}).reset_index()

fig = px.bar(cluster_stats, 
             x='Cluster', 
             y=['popularity', 'budget', 'revenue', 'runtime', 'vote_average', 'vote_count'], 
             title='Estatísticas dos Clusters',
             labels={'value': 'Média', 'Cluster': 'Cluster'},
             height=500, 
             width=800)
fig.update_layout(barmode='stack')
st.plotly_chart(fig)

# Visualização de outliers por cluster
st.subheader("Detecção e Remoção de Outliers")
numeric_cols = ['popularity', 'budget', 'revenue', 'runtime', 'vote_average', 'vote_count']
coluna_selecionada = st.selectbox('Selecione a coluna para verificar os outliers:', numeric_cols)

# Calculando os limites de IQR
Q1 = df_scaled[coluna_selecionada].quantile(0.25)
Q3 = df_scaled[coluna_selecionada].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Opção de remover outliers
if st.button('Remover Outliers'):
    df_filtrado = df_scaled[(df_scaled[coluna_selecionada] >= limite_inferior) & (df_scaled[coluna_selecionada] <= limite_superior)]
    fig = px.box(df_filtrado, x='Cluster', y=coluna_selecionada, color='Cluster',
                 title=f'Comparação de {coluna_selecionada} por Cluster (Sem Outliers)',
                 color_discrete_sequence=px.colors.qualitative.Plotly)
else:
    fig = px.box(df_scaled, x='Cluster', y=coluna_selecionada, color='Cluster',
                 title=f'Comparação de {coluna_selecionada} por Cluster (Com Outliers)',
                 color_discrete_sequence=px.colors.qualitative.Plotly)

# Exibir o gráfico
st.plotly_chart(fig)


# Exploração dos Dados Antes da Normalização
st.subheader("Exploração dos Dados Antes da Normalização")
df_new_encoded_before_normalization = df_sample.copy()
for col in numeric_cols:
    fig = px.histogram(df_new_encoded_before_normalization, 
                       x=col, 
                       title=f'Distribuição de {col.replace("_", " ").title()} Antes da Normalização',
                       labels={col: col.replace('_', ' ').title(), 'count': 'Contagem'},
                       height=500, 
                       width=900)
    st.plotly_chart(fig)
