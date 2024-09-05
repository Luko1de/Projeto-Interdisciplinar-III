import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
import streamlit as st
import pandas as pd

st.title('Exploração do Dataset de Filmes')

# Caminho do arquivo CSV local
file_path = r'C:\Users\Administrador\Desktop\movies.csv'

# Leitura do arquivo CSV
df = pd.read_csv(file_path)

# Exibir o DataFrame completo
st.write("Amostra de 5.000 registros do dataset:")
df_sample = df.sample(n=5000)
st.dataframe(df_sample)

df_sample_no_nulls = df_sample.dropna()

# Exibir as dimensões do DataFrame
st.write("Dimensões do dataset amostrado:", df_sample.shape)

# Seleção das colunas desejadas
df_new = df_sample_no_nulls[['title','genres', 'original_language','popularity','release_date', 'budget', 'revenue', 'runtime','vote_average', 'vote_count','production_companies','recommendations','overview']]
# Conversão de 'release_date' para datetime
# Crie uma cópia do DataFrame, se necessário
# Convertendo 'release_date' para datetime e lidando com erros
# Converta 'release_date' para datetime e trate erros
df_new['release_date'] = pd.to_datetime(df_new['release_date'], errors='coerce')

# Remover linhas com datas inválidas
df_new = df_new.dropna(subset=['release_date'])

# Verifique o tipo de dado da coluna
print(df_new['release_date'].dtype)

# Extração do ano de lançamento
df_new['release_year'] = df_new['release_date'].dt.year





@st.cache_data
def get_sample_data(df_new, sample_size=5000):
    return df.sample(n=sample_size)


# Substitua `df_new` pelo DataFrame que você estiver usando
df_sample = get_sample_data(df_new)

# Colunas numéricas que deseja analisar
numerical_cols = ['popularity', 'budget', 'revenue', 'runtime', 'vote_average', 'vote_count']


    
def preprocess_data(df_new):
    # Preencher valores ausentes nas colunas numéricas
    df_new.loc[:, 'budget'] = df_new['budget'].fillna(df_new['budget'].mean())
    df_new.loc[:, 'revenue'] = df_new['revenue'].fillna(df_new['revenue'].mean())
    df_new.loc[:, 'runtime'] = df_new['runtime'].fillna(df_new['runtime'].mean())
    df_new.loc[:, 'popularity'] = df_new['popularity'].fillna(df_new['popularity'].median())
    df_new.loc[:, 'vote_average'] = df_new['vote_average'].fillna(df_new['vote_average'].mean())
    df_new.loc[:, 'vote_count'] = df_new['vote_count'].fillna(df_new['vote_count'].median())

    # Preencher valores ausentes nas colunas categóricas
    df_new.loc[:, 'original_language'] = df_new['original_language'].fillna(df_new['original_language'].mode()[0])
    df_new.loc[:, 'genres'] = df_new['genres'].fillna("Unknown")
    df_new.loc[:, 'production_companies'] = df_new['production_companies'].fillna("Unknown")
    df_new.loc[:, 'recommendations'] = df_new['recommendations'].fillna("Unknown")
    df_new.loc[:, 'overview'] = df_new['overview'].fillna("No overview available")
    
    return df_new

import streamlit as st
import pandas as pd

# Supondo que você já tenha carregado seu DataFrame df_new
# df_new = pd.read_csv('path/to/your/file.csv')

def adjust_budget_for_inflation(row):
    """
    Ajuste o orçamento para a inflação com base no ano de lançamento.
    """
    base_year = 2023  # Ano base para o reajuste
    inflation_rate = 0.03  # Taxa de inflação média anual (ajuste conforme necessário)
    
    year_diff = base_year - row['release_year']
    adjusted_budget = row['budget'] * (1 + inflation_rate) ** year_diff
    return adjusted_budget

# Aplicar a função para ajustar o orçamento
df_new['adjusted_budget'] = df_new.apply(adjust_budget_for_inflation, axis=1)

# Visualizar o resultado no console
print(df_new[['title', 'budget', 'release_year', 'adjusted_budget']].head())

# Configuração do Streamlit

import streamlit as st
import plotly.express as px
import pandas as pd



# Ajustar orçamento para a inflação (se necessário)
def adjust_budget_for_inflation(row):
    base_year = 2023  # Ano base para o reajuste
    inflation_rate = 0.03  # Taxa de inflação média anual (ajuste conforme necessário)
    year_diff = base_year - row['release_year']
    adjusted_budget = row['budget'] * (1 + inflation_rate) ** year_diff
    return adjusted_budget

df_new['adjusted_budget'] = df_new.apply(adjust_budget_for_inflation, axis=1)

# Criar o gráfico de dispersão
fig = px.scatter(df_new, 
                 x="release_year", 
                 y="adjusted_budget", 
                 color="budget",
                 hover_data=["title", "budget", "adjusted_budget"],
                 labels={"adjusted_budget": "Adjusted Budget", "release_year": "Release Year"},
                 title="Adjusted Budget vs. Release Year",
                 color_continuous_scale=px.colors.sequential.Plasma)  # Usar paleta Plasma

fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Adjusted Budget (in USD)",
    hovermode="x unified"
)

# Configuração do Streamlit
st.title("Gráfico de Orçamento Ajustado vs. Ano de Lançamento:")
st.plotly_chart(fig)

@st.cache_data
def processar_generos(df_new):
    # Etapa 1: Contar a frequência dos gêneros
    all_genres = []
    for genres_str in df_new['genres']:
        if isinstance(genres_str, str):
            genres_list = genres_str.strip('[]').replace("'", "").split(', ')
            all_genres.extend(genres_list)
    
    genre_counts = Counter(all_genres)
    
    # Etapa 2: Codificar os gêneros usando MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df_new['genres'].apply(lambda x: x.strip('[]').replace("'", "").split(', ') if isinstance(x, str) else []))
    genres_df_new = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    
    # Etapa 3: Selecionar os 20 principais gêneros
    top_20_genres_names = [genre for genre, count in genre_counts.most_common(20)]
    top_20_genres_df_new = genres_df_new[top_20_genres_names]
    
    # Etapa 4: Concatenar com o DataFrame original
    df_new_encoded = pd.concat([df_new, top_20_genres_df_new], axis=1)
    
    # Etapa 5: Processar gêneros menos frequentes
    def process_genres(genres):
        if pd.isna(genres):  # Verifica se o valor é NaN
            return []
        else:
            return [genre for genre in genres.strip('[]').replace("'", "").split(', ') if genre not in top_20_genres_names]

    df_new_encoded['other_genres'] = df_new_encoded['genres'].apply(process_genres)
    df_new_exploded = df_new_encoded.explode('other_genres')
    df_new_one_hot = pd.get_dummies(df_new_exploded['other_genres'])
    df_new_encoded = df_new_encoded.join(df_new_one_hot.groupby(df_new_exploded.index).sum(), rsuffix='_onehot')
    df_new_encoded = df_new_encoded.drop(columns=['other_genres'])

    return df_new_encoded, top_20_genres_names

df_new_encoded, top_20_genres_names = processar_generos(df_new)


# Supõe-se que df_new_encoded já esteja carregado e processado
df_new_encoded['release_year'] = df_new_encoded['release_year'].fillna(0).astype(int)

# Criação da coluna 'decade'
df_new_encoded['decade'] = (df_new_encoded['release_year'] // 10 * 10).astype(int)

# Criação das categorias
df_new_encoded['budget_category'] = pd.cut(df_new_encoded['adjusted_budget'], bins=[0, 10000000, 50000000, 100000000, np.inf],
                                           labels=['Baixo (0-10)', 'Médio (10-50)', 'Alto (50-100)', 'Muito Alto (>100)'])

df_new_encoded['revenue_category'] = pd.cut(df_new_encoded['revenue'], bins=[0, 50000000, 200000000, 500000000, np.inf],
                                            labels=['Baixa (0-50)', 'Média (50-200)', 'Alta (200-500)', 'Muito Alta (>500)'])

df_new_encoded['popularity_category'] = pd.cut(df_new_encoded['popularity'], bins=[0, 10, 20, 50, np.inf],
                                               labels=['Baixa (0-10)', 'Média (10-20)', 'Alta (20-50)', 'Muito Alta (>50)'])

df_new_encoded['runtime_category'] = pd.cut(df_new_encoded['runtime'], bins=[0, 60, 120, 180, np.inf],
                                            labels=['Curto (0-60)', 'Médio (60-120)', 'Longo (120-180)', 'Muito Longo (>180)'])

df_new_encoded['vote_average_category'] = pd.cut(df_new_encoded['vote_average'], bins=[0, 5, 7, 8.5, 10],
                                                labels=['Baixa (0-5)', 'Média (5-7)', 'Alta (7-8.5)', 'Muito Alta (8.5-10)'])

df_new_encoded['vote_count_category'] = pd.cut(df_new_encoded['vote_count'], bins=[0, 100, 500, 1000, np.inf],
                                              labels=['Baixo (0-100)', 'Médio (100-500)', 'Alto (500-1000)', 'Muito Alto (>1000)'])

# Título da seção

# Criar e exibir gráficos de contagem para cada categoria
categorias = [
    ('budget_category', 'Distribuição do Orçamento'),
    ('revenue_category', 'Distribuição da Receita'),
    ('popularity_category', 'Distribuição da Popularidade'),
    ('runtime_category', 'Distribuição da Duração'),
    ('vote_average_category', 'Distribuição da Avaliação Média'),
    ('vote_count_category', 'Distribuição da Contagem de Votos')
]

# for categoria_col, title in categorias:
#     if categoria_col in df_new_encoded.columns:
#         fig = criar_grafico_contagem(df_new_encoded, categoria_col, title)
#         st.pyplot(fig)
#     else:
#         st.write(f"Coluna {categoria_col} não encontrada no DataFrame.")
    
def process_data(df_new_encoded):
    # Inicializar o MultiLabelBinarizer
    mlb_language = MultiLabelBinarizer()

    # Ajustar e transformar os idiomas originais
    language_encoded = mlb_language.fit_transform(df['original_language'].apply(lambda x: [x] if isinstance(x, str) else []))

    # Criar um novo DataFrame com os idiomas originais codificados
    language_df = pd.DataFrame(language_encoded, columns=mlb_language.classes_)

    # Concatenar o DataFrame original com os idiomas originais codificados
    df_new_encoded = pd.concat([df_new_encoded, language_df], axis=1)
    
    # Contar a frequência de cada idioma original
    language_counts = df_new_encoded['original_language'].value_counts()

    # Obter as 20 principais línguas originais
    top_20_languages = language_counts.head(20).index.tolist()

    # Criar uma função para codificar os idiomas originais
    def encode_language(language):
        if language in top_20_languages:
            return language
        else:
            return 'other'

    # Aplicar a função para criar uma nova coluna com os idiomas codificados
    df_new_encoded['encoded_language'] = df_new_encoded['original_language'].apply(encode_language)

    # Aplicar o one-hot encoding na coluna 'encoded_language'
    df_new_encoded = pd.get_dummies(df_new_encoded, columns=['encoded_language'], prefix=['language'])

    # Criar uma função para processar a coluna 'original_language'
    def process_language(language):
        if pd.isna(language):  # Verifica se o valor é NaN
            return []
        else:
            if language not in top_20_languages:
                return [language]
            else:
                return []

    # Cria uma nova coluna 'other_languages' para armazenar os idiomas menos frequentes
    df_new_encoded['other_languages'] = df_new_encoded['original_language'].apply(process_language)

    # Explode a coluna 'other_languages' para criar uma linha para cada idioma
    df_exploded_language = df_new_encoded.explode('other_languages')

    # Realiza o one-hot encoding na coluna 'other_languages'
    df_new_encoded_one_hot_language = pd.get_dummies(df_exploded_language['other_languages'])

    # Junta os dados one-hot encoded de volta ao DataFrame original com sufixos
    df_new_encoded = df_new_encoded.join(df_new_encoded_one_hot_language.groupby(df_exploded_language.index).sum(), rsuffix='_onehot')

    # Remove a coluna 'other_languages' original, se não for mais necessária
    df_new_encoded = df_new_encoded.drop(columns=['other_languages'])
    
    return df_new_encoded

df_new_encoded_before_normalization = df_new_encoded.copy()


def process_and_scale_data(df_new_encoded):
    # Inicializar os escalonadores
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # Normalizar as colunas
    df_new_encoded[['popularity', 'budget', 'adjusted_budget', 'revenue', 'runtime', 'vote_average', 'vote_count']] = minmax_scaler.fit_transform(
        df_new_encoded[['popularity', 'budget', 'revenue', 'runtime', 'vote_average', 'vote_count']]
    )

    # Padronizar as colunas
    df_new_encoded[['popularity', 'budget', 'adjusted_budget', 'revenue', 'runtime', 'vote_average', 'vote_count']] = standard_scaler.fit_transform(
        df_new_encoded[['popularity', 'budget', 'adjusted_budget', 'revenue', 'runtime', 'vote_average', 'vote_count']]
    )
    
    return df_new_encoded



    


numeric_cols = df_new_encoded.select_dtypes(include=['number']).columns
categorical_cols = df_new_encoded.select_dtypes(include=['object']).columns

df_new_encoded[numeric_cols] = df_new_encoded[numeric_cols].fillna(df_new_encoded[numeric_cols].mean())

# Preencher valores ausentes
for col in categorical_cols:
    mode = df_new_encoded[col].mode()[0]  # Pega a moda
    df_new_encoded[col].fillna(mode, inplace=True)
    
X = df_new_encoded[numeric_cols]

# Encontre o número ideal de clusters usando o método do cotovelo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotar o gráfico do cotovelo usando o Streamlit
st.title('Método do Cotovelo para Seleção de Clusters')

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss)
ax.set_title('Método do Cotovelo')
ax.set_xlabel('Número de Clusters')
ax.set_ylabel('WCSS')

st.plotly_chart(fig)

n_clusters = 4

# Aplicar o KMeans com o número de clusters escolhido
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)

# Adicionar a coluna de clusters ao DataFrame
df_new_encoded['cluster'] = kmeans.labels_

# Selecionar as colunas para o gráfico de dispersão
x_col = 'popularity'
y_col = 'vote_average'

# Criar o gráfico de dispersão com a clusterização
fig = px.scatter(df_new_encoded, x=x_col, y=y_col, color='cluster',
                  title=f'Clusterização com base em {x_col} e {y_col}',color_continuous_scale=px.colors.sequential.Plasma)

# Exibir o gráfico no Streamlit
st.title("Visualização de Clusterização")
st.plotly_chart(fig)














silhouette_avg = silhouette_score(X, kmeans.labels_)
st.write(f"Silhouette Score: {silhouette_avg}")

# Calcular os coeficientes de silhueta para cada amostra
sample_silhouette_values = silhouette_samples(X, kmeans.labels_)

# Criar o gráfico de silhueta
fig, ax = plt.subplots(figsize=(12, 8))

# Plotar o gráfico de silhueta para cada cluster
y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[kmeans.labels_ == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_title("Gráfico de Silhueta")
ax.set_xlabel("Coeficiente de Silhueta")
ax.set_ylabel("Cluster")

# Linha vertical indicando o valor médio da silhueta
ax.axvline(x=silhouette_avg, color="red", linestyle="--")

ax.set_yticks([])
ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Mostrar o gráfico no Streamlit
st.pyplot(fig)





st.title("Exploração dos resultados")

import streamlit as st
import plotly.express as px
import pandas as pd

# Supondo que você já tenha carregado seu DataFrame df_new_encoded
# df_new_encoded = pd.read_csv('path/to/your/file.csv')

# Agrupar o DataFrame por cluster e calcular a média de algumas colunas
cluster_stats = df_new_encoded.groupby('cluster').agg({
    'popularity': 'mean',
    'budget': 'mean',
    'revenue': 'mean',
    'runtime': 'mean',
    'vote_average': 'mean',
    'vote_count': 'mean'
}).reset_index()

# Criar um gráfico de barras empilhadas interativo
fig = px.bar(cluster_stats, 
             x='cluster', 
             y=['popularity', 'budget', 'revenue', 'runtime', 'vote_average', 'vote_count'], 
             title='Estatísticas dos Clusters',
             labels={'value': 'Média', 'cluster': 'Cluster'},
             height=500,  # altura ajustada
             width=800,color_discrete_sequence=px.colors.qualitative.Plotly)   # largura ajustada

# Configurar para barras empilhadas
fig.update_layout(barmode='stack')

# Configuração do Streamlit
st.write("Gráfico das Estatísticas dos Clusters:")
st.plotly_chart(fig)


genre_counts_by_cluster = df_new_encoded.groupby('cluster')[top_20_genres_names].sum()

# Transpor o DataFrame para que os gêneros sejam as linhas e os clusters as colunas
genre_counts_by_cluster = genre_counts_by_cluster.T.reset_index()

# Criar o gráfico de barras empilhadas interativo
fig = px.bar(genre_counts_by_cluster, 
             x='index', 
             y=genre_counts_by_cluster.columns[1:], 
             title='Gêneros por Cluster',
             labels={'index': 'Gêneros', 'value': 'Quantidade'},
             height=500,  # Altura ajustada
             width=900,   # Largura ajustada
             color_discrete_sequence=px.colors.qualitative.Plotly)  # Manter padrão de cor

# Configurar para barras empilhadas
fig.update_layout(barmode='stack')

# Ajustar a rotação dos rótulos do eixo x
fig.update_xaxes(tickangle=-40)

# Configuração do Streamlit
st.write("Gráfico de Gêneros por Cluster:")
st.plotly_chart(fig)

import streamlit as st
import plotly.express as px
import pandas as pd

# Supondo que você já tenha carregado seu DataFrame df_new_encoded
# df_new_encoded = pd.read_csv('path/to/your/file.csv')

# Lista de colunas categóricas
categorical_cols = ['budget_category', 'revenue_category', 'popularity_category', 'runtime_category', 'vote_average_category', 'vote_count_category']

# Iterar sobre cada coluna categórica e criar gráficos de barras empilhadas
for col in categorical_cols:
    fig = px.histogram(df_new_encoded, 
                       x=col, 
                       color='cluster', 
                       barmode='stack',  # Barras empilhadas por cluster
                       title=f'Distribuição de {col.replace("_", " ").title()} por Cluster',
                       labels={col: col.replace('_', ' ').title(), 'count': 'Contagem'},
                       color_discrete_sequence=px.colors.qualitative.Plotly,  # Manter padrão de cor
                       height=500,  # Altura ajustada
                       width=900)   # Largura ajustada

    # Exibir o gráfico no Streamlit
    st.write(f"Gráfico de Distribuição para {col.replace('_', ' ').title()}:")
    st.plotly_chart(fig)

import streamlit as st
import plotly.express as px
import pandas as pd

# Supondo que você já tenha carregado seu DataFrame df_new_encoded
# df_new_encoded = pd.read_csv('path/to/your/file.csv')

# Lista de colunas numéricas
numeric_cols = ['popularity', 'adjusted_budget', 'revenue', 'runtime', 'vote_average', 'vote_count']

# Iterar sobre cada coluna numérica e criar boxplots
for col in numeric_cols:
    fig = px.box(df_new_encoded, 
                 x='cluster', 
                 y=col, 
                 color='cluster',
                 title=f'Comparação de Distribuições de {col.replace("_", " ").title()} por Cluster')
    
    # Exibir o gráfico no Streamlit
    st.write(f"Boxplot para {col.replace('_', ' ').title()}:")
    st.plotly_chart(fig)

import streamlit as st
import plotly.express as px
import pandas as pd

# Supondo que você já tenha carregado seu DataFrame df_new_encoded
# df_new_encoded = pd.read_csv('path/to/your/file.csv')

# Criar o gráfico de dispersão
fig = px.scatter(df_new_encoded, 
                 x='release_year', 
                 y='popularity', 
                 color='cluster',
                 title='Ano de Lançamento em Relação aos Clusters')

# Exibir o gráfico no Streamlit
st.write("Gráfico de Dispersão: Ano de Lançamento em Relação aos Clusters")
st.plotly_chart(fig)


import streamlit as st
import plotly.express as px
import pandas as pd

# Supondo que você já tenha carregado seu DataFrame df_new_encoded
# df_new_encoded = pd.read_csv('path/to/your/file.csv')

# Agrupar por ano e cluster para calcular a média das colunas numéricas
trend_stats = df_new_encoded.groupby(['release_year', 'cluster']).agg({
    'popularity': 'mean',
    'budget': 'mean',
    'revenue': 'mean',
    'runtime': 'mean',
    'vote_average': 'mean',
    'vote_count': 'mean'
}).reset_index()

# Criar o gráfico de linha para visualizar a tendência ao longo dos anos por cluster
fig = px.line(trend_stats, 
              x='release_year', 
              y='popularity',  # Altere essa coluna para outras variáveis que deseja explorar
              color='cluster', 
              title='Tendência de Popularidade por Cluster ao Longo dos Anos',
              labels={'release_year': 'Ano', 'popularity': 'Média de Popularidade'},
              color_discrete_sequence=px.colors.qualitative.Plotly,
              height=500, 
              width=900)

# Exibir o gráfico no Streamlit
st.write("Gráfico de Tendência de Popularidade por Cluster ao Longo dos Anos")
st.plotly_chart(fig)


































