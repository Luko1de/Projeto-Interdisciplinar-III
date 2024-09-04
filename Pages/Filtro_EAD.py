import streamlit as st
import pandas as pd
import plotly.express as px
import itertools

# Carregar dados
@st.cache_data
def load_data():
    data = pd.read_parquet('utils/movies_cleaned.parquet')
    data_sample = data.sample(n=5000, random_state=42)
    return data_sample

# Carregar o dataset
df = load_data()

# Título do Dashboard
st.title('Dashboard Interativo de Análise de Filmes')

# Caixa de seleção para escolher colunas
st.header('Configurações do Dashboard')

# Selecionar colunas para análise
selected_columns = st.multiselect(
    'Escolha as colunas para análise',
    options=df.columns.tolist(),
)

# Exibir DataFrame filtrado
st.subheader('Dados Filtrados')
if selected_columns:
    st.dataframe(df[selected_columns])

# Funções para criar gráficos com Plotly
def create_histograms(data, columns):
    for col in columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            st.subheader(f'Histograma de {col}')
            fig = px.histogram(data, x=col, nbins=20)
            st.plotly_chart(fig)

def create_scatter_plots(data, columns):
    if len(columns) >= 2:
        # Criar gráficos para todas as combinações de pares de colunas
        pairs = list(itertools.combinations(columns, 2))
        for (x, y) in pairs:
            st.subheader(f'Gráfico de Dispersão: {x} vs {y}')
            # Define o tamanho fixo da figura
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

# Mostrar os gráficos de acordo com a escolha do usuário
if selected_columns:
    # Multiseleção para escolher os tipos de gráficos
    chart_types = st.multiselect(
        'Escolha o(s) tipo(s) de gráfico(s) que deseja visualizar',
        ('Histograma', 'Gráfico de Dispersão', 'Boxplot')
    )
    
    # Gerar gráficos baseados na seleção do usuário
    if 'Histograma' in chart_types:
        create_histograms(df, selected_columns)
    if 'Gráfico de Dispersão' in chart_types:
        create_scatter_plots(df, selected_columns)
    if 'Boxplot' in chart_types:
        create_boxplots(df, selected_columns)
else:
    st.warning('Por favor, selecione ao menos uma coluna para análise.')
