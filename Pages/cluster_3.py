import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Título do aplicativo
st.title('Clusterização Interativa de Filmes')

# Carregar dados
@st.cache_data
def load_data():
    # Corrigir a leitura do arquivo CSV
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

# Plot 2D usando Plotly
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

# Visualização dos clusters com seaborn
st.subheader("Visualização dos Clusters em um Gráfico de Dispersão")
sns.pairplot(df_scaled[selected_columns + ['Cluster']], hue='Cluster')
st.pyplot(plt)
