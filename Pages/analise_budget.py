import pandas as pd
import streamlit as st
import plotly.express as px

def show():
    st.title("Análise do Orçamento e Gêneros")

    # Adicionar uma opção para o usuário escolher o tamanho da amostra de dados
    sample_size = st.slider('Escolha o tamanho da amostra de dados', min_value=1000, max_value=100000, value=10000, step=1000)

    # Carregar os dados necessários (com amostragem)
    st.write(f"Carregando uma amostra de {sample_size} registros...")
    df = pd.read_csv('C:/Users/lucas/Projetos 3/movies.csv', usecols=['genres', 'budget', 'revenue'], nrows=sample_size)

    # Processar a coluna budget e revenue
    df['budget'] = df['budget'].astype(str).str.replace(',', '').astype(float)
    df['revenue'] = df['revenue'].astype(str).str.replace(',', '').astype(float)

    # Preprocessar a coluna de gêneros
    df['genres'] = df['genres'].fillna('')
    df['genres_list'] = df['genres'].apply(lambda x: x.split('-'))

    # Criar um conjunto com todos os gêneros únicos
    all_genres = set(genre for sublist in df['genres_list'] for genre in sublist if genre)
    
    # Manter apenas os gêneros mais frequentes para simplificar
    top_genres = pd.Series([genre for sublist in df['genres_list'] for genre in sublist]).value_counts().head(10).index
    df_top_genres = df[df['genres_list'].apply(lambda x: any(genre in x for genre in top_genres))]

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

    # Criar gráficos de dispersão para budget e revenue
    st.write("Gráfico de Dispersão do Orçamento (Budget):")
    fig_scatter_budget = px.scatter(df, x=df.index, y='budget', title='Dispersão do Orçamento dos Filmes')
    st.plotly_chart(fig_scatter_budget, use_container_width=True)

    st.write("Gráfico de Dispersão da Receita (Revenue):")
    fig_scatter_revenue = px.scatter(df, x=df.index, y='revenue', title='Dispersão da Receita dos Filmes')
    st.plotly_chart(fig_scatter_revenue, use_container_width=True)

# Chamar a função show para exibir a aplicação Streamlit
show()
