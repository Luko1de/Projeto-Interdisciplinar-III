import streamlit as st
import pandas as pd
import plotly.express as px
import itertools

# Função para carregar os dados
@st.cache_data
def load_data():
    # Corrigir a leitura do arquivo CSV
    data = pd.read_csv('C:/Users/lucas/Projetos 3/movies_cleaned.csv')
    data_sample = data.sample(n=5000, random_state=42)
    return data_sample

# Carregar o dataset
df = load_data()

# Título do Dashboard
st.title('Analisador Exploratório de Dados (AED) Interativo')

# Seção de configurações
st.header('Configurações do Dashboard')

# Filtrar dados por colunas específicas
st.subheader('Filtros de Colunas')

# Selecionar colunas para filtragem
filter_columns = st.multiselect(
    'Escolha as colunas que deseja filtrar',
    options=df.columns.tolist(),
)

# Mostrar os filtros para cada coluna selecionada
filtered_df = df.copy()
for col in filter_columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        min_val, max_val = df[col].min(), df[col].max()
        selected_range = st.slider(f'Selecione o intervalo de {col}', min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]
    elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
        selected_categories = st.multiselect(f'Selecione as categorias para {col}', df[col].unique())
        if selected_categories:
            filtered_df = filtered_df[filtered_df[col].isin(selected_categories)]

# Exibir DataFrame filtrado
st.subheader('Dados Filtrados')
st.dataframe(filtered_df)

# Selecionar colunas para análise
st.subheader('Análise de Dados')
selected_columns = st.multiselect(
    'Escolha as colunas para análise',
    options=filtered_df.columns.tolist(),
)

# Funções para criar gráficos com Plotly
def create_histograms(data, columns):
    for col in columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            st.subheader(f'Histograma de {col}')
            fig = px.histogram(data, x=col, nbins=20)
            st.plotly_chart(fig)

def create_scatter_plots(data, columns):
    if len(columns) >= 2:
        pairs = list(itertools.combinations(columns, 2))
        for (x, y) in pairs:
            st.subheader(f'Gráfico de Dispersão: {x} vs {y}')
            fig = px.scatter(data, x=x, y=y, width=800, height=600)
            st.plotly_chart(fig)
    else:
        st.warning('Por favor, selecione ao menos duas colunas para o gráfico de dispersão.')

def create_boxplots(data, columns):
    for col in columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            st.subheader(f'Boxplot de {col}')
            fig = px.box(data, y=col)
            st.plotly_chart(fig)

# Gerar gráficos de acordo com a escolha do usuário
if selected_columns:
    chart_types = st.multiselect(
        'Escolha o(s) tipo(s) de gráfico(s) que deseja visualizar',
        ('Histograma', 'Gráfico de Dispersão', 'Boxplot')
    )
    
    # Gerar gráficos baseados na seleção do usuário
    if 'Histograma' in chart_types:
        create_histograms(filtered_df, selected_columns)
    if 'Gráfico de Dispersão' in chart_types:
        create_scatter_plots(filtered_df, selected_columns)
    if 'Boxplot' in chart_types:
        create_boxplots(filtered_df, selected_columns)
else:
    st.warning('Por favor, selecione ao menos uma coluna para análise.')
