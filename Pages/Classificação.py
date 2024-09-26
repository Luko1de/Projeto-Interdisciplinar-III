import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Função para carregar o modelo a partir de um arquivo
def carregar_modelo(arquivo):
    with open(arquivo, 'rb') as file:
        modelo = pickle.load(file)
    return modelo

# Função para carregar os dados
def carregar_dados():
    caminho_dados = 'C:/Users/Aline/OneDrive - MSFT/UFRPE - OD/pisi3-2024.1/Projeto-Interdisciplinar-III - v03/utils/dados.pkl'
    
    if caminho_dados.endswith('.parquet'):
        return pd.read_parquet(caminho_dados)
    elif caminho_dados.endswith('.pkl'):
        return pd.read_pickle(caminho_dados)
    else:
        st.error("Formato de arquivo não suportado. Use .parquet ou .pkl.")
        return None

# Dicionário de features
features_dict = {
    "vetores": ['actors_vector', 'prodcia_vector'],
    "numericas": ['budget', 'revenue', 'runtime', 'vote_average'],
    "generos": ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
                'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
                'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']
}

all_features = features_dict["vetores"] + features_dict["numericas"] + features_dict["generos"]

# Carregar os modelos
random_forest_model = carregar_modelo('Pages/rf_model.pkl')
SVM_model = carregar_modelo('Pages/svm_model.pkl')
xgboost_model = carregar_modelo('Pages/xgb_model.pkl')

algorithms = {
    "Random Forest": random_forest_model,
    "SVM": SVM_model,
    "XGBoost": xgboost_model
}

# Carregar os dados
df = carregar_dados()

# Verificar se os dados foram carregados corretamente
if df is not None:
    # Filtrar os dados diretamente no DataFrame principal `df`

    # Obter companhias de produção e atores únicos
    cia_producao_opcoes = df['production_companies'].explode().unique()
    atores_opcoes = df['credits'].explode().unique()

    # Resultados previamente calculados
    results = {
        "Random Forest": {"Acurácia": 0.96, "Acurácia Média": 0.96},
        "SVM": {"Acurácia": 0.72, "Acurácia Média": 0.71}, 
        "XGBoost": {"Acurácia": 0.99, "Acurácia Média": 0.98} 
    }

    # Interface Streamlit
    st.title("Previsão de Sucesso de Filmes 🎬")
    st.write("Escolha os parâmetros e o algoritmo para prever se o filme fará sucesso!")

    # Seleção de múltiplos gêneros, companhias de produção e atores
    generos = st.multiselect("Escolha o(s) gênero(s) do filme:", features_dict["generos"], default=None)
    cia_producao = st.multiselect("Escolha a(s) companhia(s) de produção:", cia_producao_opcoes, default=None)
    atores = st.multiselect("Escolha os atores:", atores_opcoes, default=None)
    orcamento = st.slider("Escolha o intervalo do orçamento do filme (em milhões):", min_value=1, max_value=300, step=1)
    algoritmo_escolhido = st.selectbox("Escolha o algoritmo de Machine Learning:", list(algorithms.keys()))

    # Exibir os resultados
    st.write(f"**Resultados para {algoritmo_escolhido}:**")
    st.write(f"- Acurácia: {results[algoritmo_escolhido]['Acurácia']:.2f}")
    st.write(f"- Acurácia Média (Cross-Validation): {results[algoritmo_escolhido]['Acurácia Média']:.2f}")

    # Função para filtrar os dados com base nos parâmetros do usuário
    def filtrar_dados(generos, cia_producao, atores, orcamento):
        # Filtro para gêneros
        if generos:
            filtro_generos = df[generos].sum(axis=1) > 0  # Pelo menos 1 gênero deve ser verdadeiro
        else:
            filtro_generos = pd.Series([True] * len(df))

        # Filtro para companhias de produção
        if cia_producao:
            filtro_cias = df['production_companies'].apply(lambda x: any(cia in x for cia in cia_producao))
        else:
            filtro_cias = pd.Series([True] * len(df))

        # Filtro para atores
        if atores:
            filtro_atores = df['credits'].apply(lambda x: any(ator in x for ator in atores))
        else:
            filtro_atores = pd.Series([True] * len(df))

        # Filtro para orçamento
        filtro_orcamento = df['budget'] <= orcamento * 1000000

        # Aplicando todos os filtros
        dados_filtrados = df[filtro_generos & filtro_cias & filtro_atores & filtro_orcamento]
        return dados_filtrados

    # Botão para calcular a previsão
    if st.button("Calcular Previsão"):
        if not generos or not cia_producao or not atores:
            st.warning("Por favor, selecione pelo menos um gênero, uma companhia de produção e um ator.")
        else:
            dados_filtrados = filtrar_dados(generos, cia_producao, atores, orcamento)
            
            if len(dados_filtrados) == 0:
                st.write("Nenhum dado corresponde aos critérios escolhidos.")
            else:
                # Extrair apenas as colunas relevantes para o modelo
                dados_para_prever = dados_filtrados[all_features]
                
                # Puxar o modelo selecionado e fazer previsões
                modelo = algorithms[algoritmo_escolhido]
                predicao = modelo.predict(dados_para_prever)
                probabilidade = modelo.predict_proba(dados_para_prever)[:, 1]
            
                for i in range(len(predicao)):
                    if predicao[i] == 1:
                        st.success(f"Previsão: O filme será um sucesso! 🎉 (Probabilidade: {probabilidade[i]:.2f})")
                    else:
                        st.error(f"Previsão: O filme pode não ser um sucesso. 😕 (Probabilidade: {probabilidade[i]:.2f})")
else:
    st.write("Erro ao carregar os dados. Verifique o caminho ou formato do arquivo.")

