import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import plotly.express as px

st.title('Exploração do Dataset de Filmes')

# Opção de upload de arquivo CSV
file_path = st.file_uploader("Carregar arquivo CSV", type=["csv"])
@st.cache_data
def load_data():
    # Corrigir a leitura do arquivo CSV
    data = pd.read_csv('C:/Users/lucas/Projetos 3/movies_cleaned.csv')
    data_sample = data.sample(n=5000, random_state=42)
    return data_sample

# Carregar o dataset
df = load_data()
    # Exibir o DataFrame completo
st.write("Amostra de 5.000 registros do dataset:")
df_sample = df.sample(n=5000, random_state=42)
st.dataframe(df_sample)

df_sample_no_nulls = df_sample.dropna()

    # Seleção das colunas desejadas
df_new = df_sample_no_nulls[['title', 'genres', 'original_language', 'popularity', 'release_date', 
                                  'budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 
                                  'production_companies', 'recommendations', 'overview']]

df_new['release_date'] = pd.to_datetime(df_new['release_date'], errors='coerce')
df_new = df_new.dropna(subset=['release_date'])
    # Extração do ano de lançamento
df_new['release_year'] = df_new['release_date'].dt.year

@st.cache_data
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

df_new = preprocess_data(df_new)

    # Filtros interativos
st.sidebar.header("Filtros")
    
    # Filtro por idioma
languages = df_new['original_language'].unique()
selected_languages = st.sidebar.multiselect("Selecione Idiomas", languages, default=languages)

    # Filtro por gêneros
genres = df_new['genres'].unique()
selected_genres = st.sidebar.multiselect("Selecione Gêneros", genres, default=genres)

    # Filtro por ano de lançamento
release_years = df_new['release_year'].unique()
selected_years = st.sidebar.multiselect("Selecione Anos de Lançamento", release_years, default=release_years)

    # Aplicando os filtros
filtered_data = df_new[
    (df_new['original_language'].isin(selected_languages)) &
    (df_new['genres'].isin(selected_genres)) &
    (df_new['release_year'].isin(selected_years))
    ]

    # Exibir dados filtrados
st.write("Dados Filtrados:")
st.dataframe(filtered_data)

    # Escalonar dados numéricos para clusterização
numerical_cols = ['popularity', 'budget', 'revenue', 'runtime', 'vote_average', 'vote_count']
    
if not filtered_data.empty:
        # Escalar dados
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(filtered_data[numerical_cols])
        
        # Número de clusters
        n_clusters = st.sidebar.slider("Número de Clusters", 1, 10, 3)
        
        # Executar KMeans
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        filtered_data['Cluster'] = kmeans.fit_predict(scaled_data)
        
        # Visualizar clusters
        fig = px.scatter(
            filtered_data,
            x='release_year',
            y='budget',  # Utilize a coluna de orçamento que deseja visualizar
            color='Cluster',
            title='Distribuição de Filmes por Clusters',
            labels={'budget': 'Orçamento (USD)', 'release_year': 'Ano de Lançamento'},
            hover_name='title'
        )
        
        st.plotly_chart(fig)
else:
        st.warning("Nenhum dado disponível após a aplicação dos filtros.")
