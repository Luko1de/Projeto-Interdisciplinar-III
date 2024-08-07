import pandas as pd
import streamlit as st
import plotly.express as px

def show():
    import streamlit as st
import pandas as pd
import plotly.express as px
'C:/Users/lucas/Projetos 3/movies.csv'
st.title("Análise Exploratória de Filmes")

# Adicionar uma opção para o usuário escolher o tamanho da amostra de dados
sample_size = st.slider('Escolha o tamanho da amostra de dados', min_value=1000, max_value=100000, value=10000, step=1000)

# Carregar os dados necessários (com amostragem)
st.write(f"Carregando uma amostra de {sample_size} registros...")
df = pd.read_csv('C:/Users/lucas/Projetos 3/movies.csv', nrows=sample_size)

# Processar a coluna budget e revenue
df['budget'] = df['budget'].astype(str).str.replace(',', '').astype(float)
df['revenue'] = df['revenue'].astype(str).str.replace(',', '').astype(float)
df['popularity'] = df['popularity'].astype(float)
df['vote_average'] = df['vote_average'].astype(float)
df['vote_count'] = df['vote_count'].astype(float)
df['runtime'] = df['runtime'].astype(float)

# Preprocessar a coluna de gêneros
df['genres'] = df['genres'].fillna('')
df['genres_list'] = df['genres'].apply(lambda x: x.split('-'))

# Gráficos de Dispersão para budget e revenue
st.write("### Gráfico de Dispersão do Orçamento (Budget)")
st.write("Visualiza a distribuição do orçamento dos filmes, ajudando a identificar outliers e entender a variação dos valores de orçamento.")
fig_scatter_budget = px.scatter(df, x=df.index, y='budget', title='Dispersão do Orçamento dos Filmes')
st.plotly_chart(fig_scatter_budget, use_container_width=True)

st.write("### Gráfico de Dispersão da Receita (Revenue)")
st.write("Visualiza a distribuição da receita dos filmes, permitindo a identificação de filmes com receitas excepcionalmente altas ou baixas.")
fig_scatter_revenue = px.scatter(df, x=df.index, y='revenue', title='Dispersão da Receita dos Filmes')
st.plotly_chart(fig_scatter_revenue, use_container_width=True)

# Análise da Popularidade por Gênero
st.write("### Popularidade Média por Gênero")
st.write("Insights sobre quais gêneros são mais populares podem ser usados para segmentação de mercado e recomendação de filmes.")
genre_popularity = df.explode('genres_list').groupby('genres_list')['popularity'].mean().reset_index()
genre_popularity = genre_popularity.sort_values(by='popularity', ascending=False)
fig_genre_popularity = px.bar(genre_popularity, x='genres_list', y='popularity', labels={'genres_list': 'Gênero', 'popularity': 'Popularidade Média'}, title='Popularidade Média por Gênero')
st.plotly_chart(fig_genre_popularity, use_container_width=True)

# Análise da Receita por Gênero
st.write("### Receita Média por Gênero")
st.write("Identificar quais gêneros são mais lucrativos pode ajudar na tomada de decisões estratégicas para produções futuras e alocação de recursos.")
genre_revenue = df.explode('genres_list').groupby('genres_list')['revenue'].mean().reset_index()
genre_revenue = genre_revenue.sort_values(by='revenue', ascending=False)
fig_genre_revenue = px.bar(genre_revenue, x='genres_list', y='revenue', labels={'genres_list': 'Gênero', 'revenue': 'Receita Média'}, title='Receita Média por Gênero')
st.plotly_chart(fig_genre_revenue, use_container_width=True)

# Análise da Avaliação Média por Gênero
st.write("### Avaliação Média por Gênero")
st.write("Gêneros com avaliações mais altas podem indicar preferências de qualidade, informando sistemas de recomendação que priorizem filmes bem avaliados.")
genre_vote_average = df.explode('genres_list').groupby('genres_list')['vote_average'].mean().reset_index()
genre_vote_average = genre_vote_average.sort_values(by='vote_average', ascending=False)
fig_genre_vote_average = px.bar(genre_vote_average, x='genres_list', y='vote_average', labels={'genres_list': 'Gênero', 'vote_average': 'Avaliação Média'}, title='Avaliação Média por Gênero')
st.plotly_chart(fig_genre_vote_average, use_container_width=True)

# Análise de Receita e Avaliação por Idioma
st.write("### Receita Média por Idioma")
st.write("Entender o impacto do idioma na receita pode influenciar estratégias de marketing e distribuição.")
language_revenue = df.groupby('original_language')['revenue'].mean().reset_index()
language_revenue = language_revenue.sort_values(by='revenue', ascending=False)
fig_language_revenue = px.bar(language_revenue, x='original_language', y='revenue', labels={'original_language': 'Idioma', 'revenue': 'Receita Média'}, title='Receita Média por Idioma')
st.plotly_chart(fig_language_revenue, use_container_width=True)

st.write("### Avaliação Média por Idioma")
st.write("Compreender o impacto do idioma na avaliação pode informar sobre preferências culturais e de qualidade.")
language_vote_average = df.groupby('original_language')['vote_average'].mean().reset_index()
language_vote_average = language_vote_average.sort_values(by='vote_average', ascending=False)
fig_language_vote_average = px.bar(language_vote_average, x='original_language', y='vote_average', labels={'original_language': 'Idioma', 'vote_average': 'Avaliação Média'}, title='Avaliação Média por Idioma')
st.plotly_chart(fig_language_vote_average, use_container_width=True)

# Análise Temporal de Lançamentos
st.write("### Tendência de Lançamentos ao Longo dos Anos")
st.write("Analisar tendências de lançamentos pode ajudar a identificar períodos de alta atividade na indústria cinematográfica e prever tendências futuras.")
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
release_trend = df.set_index('release_date').resample('Y')['id'].count().reset_index()
fig_release_trend = px.line(release_trend, x='release_date', y='id', labels={'release_date': 'Ano', 'id': 'Número de Filmes Lançados'}, title='Tendência de Lançamentos ao Longo dos Anos')
st.plotly_chart(fig_release_trend, use_container_width=True)

# Análise da Duração dos Filmes
st.write("### Distribuição da Duração dos Filmes")
st.write("Entender a distribuição da duração dos filmes pode informar sobre a preferência do público e influenciar a edição e produção de futuros filmes.")
fig_runtime = px.histogram(df, x='runtime', nbins=50, labels={'runtime': 'Duração (minutos)'}, title='Distribuição da Duração dos Filmes')
st.plotly_chart(fig_runtime, use_container_width=True)

# Análise da Contagem de Votos
st.write("### Distribuição da Contagem de Votos")
st.write("Analisar a contagem de votos pode ajudar a entender o engajamento do público, útil para estratégias de marketing e para ajustar modelos de predição de popularidade.")
fig_vote_count = px.histogram(df, x='vote_count', nbins=50, labels={'vote_count': 'Contagem de Votos'}, title='Distribuição da Contagem de Votos')
st.plotly_chart(fig_vote_count, use_container_width=True)

# Análise de Avaliações
st.write("### Distribuição das Avaliações dos Filmes")
st.write("Compreender a distribuição das avaliações pode ajudar a ajustar modelos de predição de qualidade e a desenvolver sistemas de recomendação que priorizem filmes bem avaliados.")
fig_vote_average = px.histogram(df, x='vote_average', nbins=50, labels={'vote_average': 'Avaliação Média'}, title='Distribuição das Avaliações dos Filmes')
st.plotly_chart(fig_vote_average, use_container_width=True)

# Chamar a função show para exibir a aplicação Streamlit
show()