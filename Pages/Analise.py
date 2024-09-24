import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Carregar os dados
df = pd.read_csv(r'C:\Users\eleve\Downloads/movies.csv')

# Análise Exploratória de Dados Inicial
st.title("Análise Exploratória de Dados Inicial")
st.write("Análise Exploratória de Dados, Leitura dos primeiros registros do dataframe:")
st.write(df.head())
st.write("Verificar informações básicas do dataframe:")
st.write(df.info())
st.write("Verificar estatísticas descritivas do dataframe:")
st.write(df.describe())
st.write("Verificar valores nulos do dataframe:")
st.write(df.isnull().sum())

# Dicionário de dados
data_dict = {
    'id': 'Identificador único do filme',
    'title': 'Título do filme',
    'genres': 'Gêneros do filme',
    'original_language': 'Idioma original do filme',
    'overview': 'Sinopse do filme',
    'popularity': 'Popularidade do filme',
    'production_companies': 'Empresas de produção do filme',
    'release_date': 'Data de lançamento do filme',
    'budget': 'Orçamento do filme',
    'revenue': 'Receita do filme',
    'runtime': 'Duração do filme',
    'status': 'Status do filme',
    'tagline': 'Tagline do filme',
    'vote_average': 'Avaliação média do filme',
    'vote_count': 'Número de votos do filme',
    'credits': 'Créditos do filme',
    'keywords': 'Palavras-chave do filme',
    'poster_path': 'Caminho do pôster do filme',
    'backdrop_path': 'Caminho do backdrop do filme',
    'recommendations': 'Recomendações do filme'
}

# Converter o dicionário de dados em um DataFrame
data_dict_df = pd.DataFrame(list(data_dict.items()), columns=['Coluna', 'Descrição'])

st.write("Dicionário de dados:")
st.table(data_dict_df)

# Processar a coluna genres para criar a matriz de gêneros
# Preenchendo valores nulos em genres com string vazia
df['genres'] = df['genres'].fillna('')

# Dividir os gêneros em uma lista
df['genres_list'] = df['genres'].apply(lambda x: x.split('-'))

# Criar um conjunto com todos os gêneros únicos
all_genres = set(genre for sublist in df['genres_list'] for genre in sublist if genre)

# Criar colunas binárias para cada gênero
for genre in all_genres:
    df[genre] = df['genres_list'].apply(lambda x: 1 if genre in x else 0)

# Criar o mapa de calor
st.write("Mapa de Calor de Gêneros dos Filmes:")

# Calcular a correlação entre os gêneros
genre_corr = df[list(all_genres)].corr()

# Criar o mapa de calor interativo com Plotly
fig_heatmap = px.imshow(genre_corr, 
                        labels=dict(color="Correlação"),
                        x=list(all_genres),
                        y=list(all_genres),
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1)

st.plotly_chart(fig_heatmap, use_container_width=True)

# Gráfico de barras com a quantidade de filmes por gênero
st.write("Quantidade de Filmes por Gênero:")

# Contar a quantidade de filmes por gênero
genre_counts = df[list(all_genres)].sum().sort_values(ascending=False)

# Criar o gráfico de barras interativo com Plotly
fig_bar = px.bar(x=genre_counts.values, 
                 y=genre_counts.index, 
                 orientation='h',
                 labels={'x': 'Quantidade de Filmes', 'y': 'Gênero'},
                 title='Quantidade de Filmes por Gênero')

st.plotly_chart(fig_bar, use_container_width=True)

# Remover vote_average iguais a 0 e vote_count menores que 300
vote_count_threshold = 300
df_votes = df[(df['vote_average'] > 0) & (df['vote_count'] > vote_count_threshold)]

# Distribuição das avaliações
fig_vote_average = px.histogram(df_votes, x='vote_average', nbins=20, labels={'vote_average': 'Avaliação média'}, title='Distribuição das avaliações')
st.plotly_chart(fig_vote_average, use_container_width=True)

# Distribuição da contagem de votos (vote_count)
fig_vote_count = px.histogram(df_votes, x='vote_count', nbins=20, labels={'vote_count': 'Contagem de votos'}, title='Distribuição da contagem de votos')
st.plotly_chart(fig_vote_count, use_container_width=True)

# Relação entre avaliação média e quantidade de votos
fig_vote_scatter = px.scatter(df_votes, x='vote_count', y='vote_average', labels={'vote_count': 'Contagem de votos', 'vote_average': 'Avaliação média'}, title='Relação entre avaliação média e Contagem de votos')
st.plotly_chart(fig_vote_scatter, use_container_width=True)

# Distribuição de notas por gênero
# Transformar DataFrame para long format para Plotly
df_genres = df_votes.melt(id_vars=['title', 'vote_average'], value_vars=list(all_genres), var_name='genre', value_name='presence')
df_genres = df_genres[df_genres['presence'] == 1]

# Criar box plot
fig_box_genre = px.box(df_genres, x='genre', y='vote_average', labels={'genre': 'Gênero', 'vote_average': 'Avaliação média'}, title='Distribuição de notas por gênero')
st.plotly_chart(fig_box_genre, use_container_width=True)

# distrubuição de notas por decada
vote_count_threshold = 150
df_dates = df[(df['vote_average'] > 0) & (df['vote_count'] > vote_count_threshold)]
# Converter release_date para datetime
df_dates['release_date'] = pd.to_datetime(df_dates['release_date'], errors='coerce')

# Adicionar coluna de década
df_dates['decade'] = (df_dates['release_date'].dt.year // 10) * 10

# Criar box plot
fig_box_decade = px.box(df_dates, x='decade', y='vote_average', labels={'decade': 'Década de lançamento', 'vote_average': 'Avaliação média'}, title='Distribuição de notas por década de lançamento')
st.plotly_chart(fig_box_decade, use_container_width=True)

# Filmes mais bem avaliados
st.write("Filmes mais bem avaliados (top 10):")
top_rated_movies = df_votes[['title', 'vote_count', 'vote_average']].sort_values(by='vote_average', ascending=False).head(10)
st.write(top_rated_movies)

# Filmes com maior número de votos
st.write("Filmes com maior número de votos (top 10):")
most_voted_movies = df_votes[['title', 'vote_count', 'vote_average']].sort_values(by='vote_count', ascending=False).head(10)
st.write(most_voted_movies)

# Preenchendo valores nulos em runtime 0
df['runtime'] = df['runtime'].fillna(0)
df_runtime = df[(df['runtime'] > 0)]

# Distribuição da duração dos filmes
fig_runtime = px.histogram(df_runtime, x='runtime', nbins=20, labels={'runtime': 'Duração (minutos)'}, title='Distribuição da duração dos filmes')
fig_runtime.update_traces(xbins=dict(
    start = 0,
    end = 500,
    size = 20
))
st.plotly_chart(fig_runtime, use_container_width=True)

# Filmes com maior duração
st.write("Filmes com maior duração (top 10):")
most_voted_movies = df_runtime[['title', 'runtime']].sort_values(by='runtime', ascending=False).head(10)
st.write(most_voted_movies)

#Analise do budget
st.title("Análise do Orçamento e Gêneros")

df_budget = pd.read_csv(r'C:\Users\eleve\Downloads/movies.csv', nrows=10000)

# Processar a coluna budget e revenue
df_budget['budget'] = df_budget['budget'].astype(str).str.replace(',', '').astype(float)
df_budget['revenue'] = df_budget['revenue'].astype(str).str.replace(',', '').astype(float)
df_budget['popularity'] = df_budget['popularity'].astype(float)
df_budget['vote_average'] = df_budget['vote_average'].astype(float)
df_budget['vote_count'] = df_budget['vote_count'].astype(float)
df_budget['runtime'] = df_budget['runtime'].astype(float)

# Preprocessar a coluna de gêneros
df_budget['genres'] = df_budget['genres'].fillna('')
df_budget['genres_list'] = df_budget['genres'].apply(lambda x: x.split('-'))
df_budget.dropna(subset='budget')
# Processar a coluna budget e revenue

# Criar um conjunto com todos os gêneros únicos
all_genres = set(genre for sublist in df_budget['genres_list'] for genre in sublist if genre)

# Manter apenas os gêneros mais frequentes para simplificar
top_genres = pd.Series([genre for sublist in df_budget['genres_list'] for genre in sublist]).value_counts().head(10).index
df_top_genres = df_budget[df_budget['genres_list'].apply(lambda x: any(genre in x for genre in top_genres))]

# Criar colunas binárias para os principais gêneros
for genre in top_genres:
    df_top_genres[genre] = df_top_genres['genres_list'].apply(lambda x: 1 if genre in x else 0)

# Calcular o saldo (revenue - budget) por gênero
genre_saldo = {}
for genre in top_genres:
    genre_data = df_top_genres[df_top_genres[genre] == 1]
    saldo = (genre_data['revenue'] - genre_data['budget']).sum()
    genre_saldo[genre] = saldo

# Converter o dicionário para DataFrame para visualização
genre_saldo_df = pd.DataFrame(list(genre_saldo.items()), columns=['Gênero', 'Saldo']).sort_values(by='Saldo', ascending=False)

# Criar o gráfico de barras para mostrar o saldo por gênero
st.write("Saldo total (receita - orçamento) por gênero dos filmes:")

fig_bar = px.bar(genre_saldo_df, 
                    x='Gênero', 
                    y='Saldo',
                    labels={'Saldo': 'Saldo (Receita - Orçamento)', 'Gênero': 'Gênero'},
                    title='Saldo Total por Gênero')

st.plotly_chart(fig_bar, use_container_width=True)

# Gráficos de Dispersão para budget e revenue
st.write("### Gráfico de Dispersão do Orçamento (Budget)")
st.write("Visualiza a distribuição do orçamento dos filmes, ajudando a identificar outliers e entender a variação dos valores de orçamento.")
fig_scatter_budget = px.scatter(df_budget, x=df_budget.index, y='budget', title='Dispersão do Orçamento dos Filmes')
st.plotly_chart(fig_scatter_budget, use_container_width=True)

st.write("### Gráfico de Dispersão da Receita (Revenue)")
st.write("Visualiza a distribuição da receita dos filmes, permitindo a identificação de filmes com receitas excepcionalmente altas ou baixas.")
fig_scatter_revenue = px.scatter(df_budget, x=df_budget.index, y='revenue', title='Dispersão da Receita dos Filmes')
st.plotly_chart(fig_scatter_revenue, use_container_width=True)

# Análise da Popularidade por Gênero
st.write("### Popularidade Média por Gênero")
st.write("Insights sobre quais gêneros são mais populares podem ser usados para segmentação de mercado e recomendação de filmes.")
genre_popularity = df_budget.explode('genres_list').groupby('genres_list')['popularity'].mean().reset_index()
genre_popularity = genre_popularity.sort_values(by='popularity', ascending=False)
fig_genre_popularity = px.bar(genre_popularity, x='genres_list', y='popularity', labels={'genres_list': 'Gênero', 'popularity': 'Popularidade Média'}, title='Popularidade Média por Gênero')
st.plotly_chart(fig_genre_popularity, use_container_width=True)

# Análise da Receita por Gênero
st.write("### Receita Média por Gênero")
st.write("Identificar quais gêneros são mais lucrativos pode ajudar na tomada de decisões estratégicas para produções futuras e alocação de recursos.")
genre_revenue = df_budget.explode('genres_list').groupby('genres_list')['revenue'].mean().reset_index()
genre_revenue = genre_revenue.sort_values(by='revenue', ascending=False)
fig_genre_revenue = px.bar(genre_revenue, x='genres_list', y='revenue', labels={'genres_list': 'Gênero', 'revenue': 'Receita Média'}, title='Receita Média por Gênero')
st.plotly_chart(fig_genre_revenue, use_container_width=True)

# Análise da Avaliação Média por Gênero
st.write("### Avaliação Média por Gênero")
st.write("Gêneros com avaliações mais altas podem indicar preferências de qualidade, informando sistemas de recomendação que priorizem filmes bem avaliados.")
genre_vote_average = df_budget.explode('genres_list').groupby('genres_list')['vote_average'].mean().reset_index()
genre_vote_average = genre_vote_average.sort_values(by='vote_average', ascending=False)
fig_genre_vote_average = px.bar(genre_vote_average, x='genres_list', y='vote_average', labels={'genres_list': 'Gênero', 'vote_average': 'Avaliação Média'}, title='Avaliação Média por Gênero')
st.plotly_chart(fig_genre_vote_average, use_container_width=True)

# Análise de Receita e Avaliação por Idioma
st.write("### Receita Média por Idioma")
st.write("Entender o impacto do idioma na receita pode influenciar estratégias de marketing e distribuição.")
language_revenue = df_budget.groupby('original_language')['revenue'].mean().reset_index()
language_revenue = language_revenue.sort_values(by='revenue', ascending=False)
fig_language_revenue = px.bar(language_revenue, x='original_language', y='revenue', labels={'original_language': 'Idioma', 'revenue': 'Receita Média'}, title='Receita Média por Idioma')
st.plotly_chart(fig_language_revenue, use_container_width=True)

st.write("### Avaliação Média por Idioma")
st.write("Compreender o impacto do idioma na avaliação pode informar sobre preferências culturais e de qualidade.")
language_vote_average = df_budget.groupby('original_language')['vote_average'].mean().reset_index()
language_vote_average = language_vote_average.sort_values(by='vote_average', ascending=False)
fig_language_vote_average = px.bar(language_vote_average, x='original_language', y='vote_average', labels={'original_language': 'Idioma', 'vote_average': 'Avaliação Média'}, title='Avaliação Média por Idioma')
st.plotly_chart(fig_language_vote_average, use_container_width=True)

# Análise Temporal de Lançamentos
st.write("### Tendência de Lançamentos ao Longo dos Anos")
st.write("Analisar tendências de lançamentos pode ajudar a identificar períodos de alta atividade na indústria cinematográfica e prever tendências futuras.")
df_budget['release_date'] = pd.to_datetime(df_budget['release_date'], errors='coerce')
release_trend = df_budget.set_index('release_date').resample('Y')['id'].count().reset_index()
fig_release_trend = px.line(release_trend, x='release_date', y='id', labels={'release_date': 'Ano', 'id': 'Número de Filmes Lançados'}, title='Tendência de Lançamentos ao Longo dos Anos')
st.plotly_chart(fig_release_trend, use_container_width=True)

# Análise da Duração dos Filmes
st.write("### Distribuição da Duração dos Filmes")
st.write("Entender a distribuição da duração dos filmes pode informar sobre a preferência do público e influenciar a edição e produção de futuros filmes.")
fig_runtime = px.histogram(df_budget, x='runtime', nbins=50, labels={'runtime': 'Duração (minutos)'}, title='Distribuição da Duração dos Filmes')
st.plotly_chart(fig_runtime, use_container_width=True)

# Análise da Contagem de Votos
st.write("### Distribuição da Contagem de Votos")
st.write("Analisar a contagem de votos pode ajudar a entender o engajamento do público, útil para estratégias de marketing e para ajustar modelos de predição de popularidade.")
fig_vote_count = px.histogram(df_budget, x='vote_count', nbins=50, labels={'vote_count': 'Contagem de Votos'}, title='Distribuição da Contagem de Votos')
st.plotly_chart(fig_vote_count, use_container_width=True)

# Análise de Avaliações
st.write("### Distribuição das Avaliações dos Filmes")
st.write("Compreender a distribuição das avaliações pode ajudar a ajustar modelos de predição de qualidade e a desenvolver sistemas de recomendação que priorizem filmes bem avaliados.")
fig_vote_average = px.histogram(df_budget, x='vote_average', nbins=50, labels={'vote_average': 'Avaliação Média'}, title='Distribuição das Avaliações dos Filmes')
st.plotly_chart(fig_vote_average, use_container_width=True)

# #Palavras chaves mais comuns
st.write("Wordcloud das palavras chaves mais comuns")

# Carregar os dados
df.dropna(subset=['keywords'], axis=0, inplace=True)
df_keywords = df['keywords']
all_keywords = "-".join(kw for kw in df_keywords)
stopwords = set(STOPWORDS)
stopwords.update(" ", "and", "of", "the")
wordcloud  = WordCloud(stopwords=stopwords, background_color='white', width=800, height=600).generate(all_keywords)

fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()
plt.imshow(wordcloud)
st.pyplot(fig)
st.title("Análise do ano de lançamento")

#ver a coluna separa do ano de lançamento
df['release_date'].head()

#formatar a data 
# Converter para datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
st.write(df['release_date'])
# Converter para datetime
#print(df['release_date'].dtype)  # Verifique o tipo de dados
st.write(df['release_date'].unique())  # Verifique os valores únicos


#ver quantas datas faltam 
#Criar uma nova coluna 'release_year' que contém apenas o ano de 'release_date'
df['release_year'] = df['release_date'].dt.year
df['release_year'] = df['release_year'].astype('Int64')
st.write(df['release_year'])
# Exibir as primeiras linhas do DataFrame para verificar o resultado
print(df[['release_date', 'release_year']])

# Contar o número de valores faltosos na coluna 'release_year'
missing_years_count = df['release_year'].isna().sum()

st.write(f"Quantidade de anos faltando: {missing_years_count}")

# Substituir os anos faltando por 0
df['release_year'] = df['release_year'].fillna(0)

# Verificar se ainda existem valores faltando
remaining_missing_years = df['release_year'].isna().sum()
st.write(f"Quantidade de anos faltando após a substituição: {remaining_missing_years}")

# Encontrar o primeiro ano de lançamento não nulo
primeiro_ano = df['release_year'][df['release_year'] != 0].min()

# Encontrar o último ano de lançamento
ultimo_ano = df['release_year'].max()

st.write("Primeiro ano de lançamento (não nulo):", primeiro_ano)
st.write("Último ano de lançamento:", ultimo_ano)

#quantidade de anos abaixo de 1900
anos_1800_1900 = df[(df['release_year'] >= 1800) & (df['release_year'] <= 1900)].sort_values(by='release_year')
anos_1800_1900.shape[0]

st.write(df.groupby('release_year').size())



from collections import Counter

# Filtrar o DataFrame para filmes lançados antes de 1900
filmes_antes_1900 = df[df['release_year'] < 1900]

# Contar a frequência de cada gênero
contagem_generos = Counter(filmes_antes_1900['genres'].dropna())

# Encontrar os gêneros mais frequentes
generos_mais_frequentes = contagem_generos.most_common(5)  # Top 5 gêneros mais frequentes

# Imprimir os gêneros mais frequentes (ou uma mensagem indicando que não há gêneros)
if generos_mais_frequentes:
    st.write("Gêneros mais frequentes de filmes lançados antes de 1900:")
    for genero, contagem in generos_mais_frequentes:
        st.write(f"{genero}: {contagem}")
else:
    print("Não há informações de gênero disponíveis para filmes lançados antes de 1900.")

# st.write("ver nomes dos filmes antes de 1900")

# print("Nomes dos filmes antes de 1900")
# # Filtrar o DataFrame para filmes lançados antes de 1900
# filmes_antes_1900 = df[df['release_year'] < 1900]

# # Selecionar a coluna 'title' (nomes dos filmes)
# nomes_filmes_antes_1900 = filmes_antes_1900['title']

# # Imprimir os nomes dos filmes
# print("Nomes dos filmes lançados antes de 1900:")
# for nome in nomes_filmes_antes_1900:
#     st.write(nome)

# st.write("Nomes do filmes depois de 2024")

# # Filtrar o DataFrame para filmes lançados depois de 2024
# filmes_depois_2024 = df[df['release_year'] > 2024]

# # Selecionar a coluna 'title' (nomes dos filmes)
# nomes_filmes_depois_2024 = filmes_depois_2024['title']

# # Imprimir os nomes dos filmes
# print("Nomes dos filmes lançados depois de 2024:")
# for nome in nomes_filmes_depois_2024:
#   print(nome)



# # Filtrar o DataFrame para filmes lançados depois de 2024
# filmes_depois_2024 = df[df['release_year'] > 2024]

# # Contar o número de filmes
# quantidade_filmes_depois_2024 = filmes_depois_2024.shape[0]

# # Imprimir o resultado
# st.write("Quantidade de filmes lançados depois de 2024:", quantidade_filmes_depois_2024)

one_hot_ano = pd.get_dummies(df, columns=['release_year'])
print(one_hot_ano)

st.write("Filmes mais populares por ano")
df['decade'] = (df['release_year'] // 10) * 10
popularidade_por_ano = df.groupby(df['decade'])['popularity'].mean()

# Ordenar por popularidade média decrescente
popularidade_por_ano_ordenada = popularidade_por_ano.sort_values(ascending=False).astype(int)


# Exibir os anos com maior popularidade média
st.write(popularidade_por_ano_ordenada.head(10))

# plt.figure(figsize=(12, 6))
# popularidade_por_ano_ordenada.plot(kind='bar')
# plt.xlabel('Ano de Lançamento')
# plt.ylabel('Popularidade Média')
# plt.title('Popularidade Média de Filmes por Ano de Lançamento')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

st.write("Quantidade de filmes por ano")
contagem_anos = df['release_year'].value_counts()

# Ordenar os anos por frequência decrescente
contagem_anos_ordenada = contagem_anos.sort_values(ascending=False)

# Exibir os anos mais frequentes
st.write(contagem_anos_ordenada.head(10))


st.write("Filmes mais populares por época")
epocas = {
    "Filmes da antiguidade(1800 até 1900)": (1800, 1900),
    "Filmes Antigos(1900 até 950)": (1900, 1950),
    "Filmes de 1900": (1950, 2000),
    "Filmes anos 2000": (2000, 2012),
    "Filmes atuais": (2012, 2024)
}

# Encontrar os filmes mais populares por época
for epoca, (inicio, fim) in epocas.items():
    # Filtrar os filmes da época
    filmes_epoca = df[(df['release_date'].dt.year >= inicio) & (df['release_date'].dt.year <= fim)]

    # Ordenar os filmes por popularidade decrescente
    filmes_epoca_ordenados = filmes_epoca.sort_values(by='popularity', ascending=False)

    # Exibir os 3 filmes mais populares da época
    print(f"\n--- {epoca} ---")
    st.write(filmes_epoca_ordenados[['title', 'popularity']].head(3))
 