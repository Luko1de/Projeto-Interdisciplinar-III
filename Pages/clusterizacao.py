import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import FunctionTransformer
import nltk
from nltk.stem import WordNetLemmatizer
import time

# Função para aplicar lemmatization
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Função para aplicar transformação de dados
def process_data(df):
    df['keywords'] = df['keywords'].astype(str).fillna('')
    df['genres'] = df['genres'].astype(str).fillna('')
    df['keywords'] = df['keywords'].apply(lambda x: ' '.join(x.split('-')))
    df['genres'] = df['genres'].apply(lambda x: ' '.join(x.split('-')))
    df['keywords'] = df['keywords'].apply(lemmatize_text)
    df['genres'] = df['genres'].apply(lemmatize_text)
    return df

# Função para criar e ajustar o modelo KMeans e calcular a inércia e o índice de Silhueta
def calculate_metrics(X, k_range):
    inertia = []
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        
        # Calcular o índice de Silhueta, somente para k > 1
        if k > 1:
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(X, labels, metric='euclidean')
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(None)
    
    return inertia, silhouette_scores

# Título do aplicativo
st.markdown("<h2 style='text-align: left; color: white;'>Método Elbow e Índice de Silhueta para Determinação de Número de Clusters</h2>", unsafe_allow_html=True)

# Marcar o início do tempo
start_time = time.time()

# Carregar e processar os dados
df = pd.read_csv(r'C:\Users\Aline\OneDrive - MSFT\UFRPE - OD\pisi3-2024.1\Projeto-Interdisciplinar-III - v03\utils\movies_cleaned.csv')
df = process_data(df)

# Combinar palavras-chave e gêneros
df_combined = df['keywords'] + ' ' + df['genres']

# Utilizar o TfidfVectorizer para transformar as palavras-chave e gêneros em numeração
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(df_combined)

# Faixa de números de clusters para testar
k_range = range(1, 11)

# Calcular inércia e índice de Silhueta
st.write("Calculando a inércia e o índice de Silhueta para cada número de clusters.")
inertia, silhouette_scores = calculate_metrics(X_tfidf, k_range)

# Plotar o gráfico Elbow
st.write("Exibindo o gráfico Elbow para entender o número de clusters por palavras-chave e gênero")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_range, inertia, marker='o')
ax.set_title('Método Elbow')
ax.set_xlabel('Número de Clusters (k)')
ax.set_ylabel('Inércia')
st.pyplot(fig)


st.write("Com base no gráfico Elbow, não foi possível identificar o número ideal de clusters, por isto devemos calcular o índice do método da silhueta.")

# Plotar o gráfico de Índice de Silhueta
st.write("Exibindo o gráfico de Índice de Silhueta para cada número de clusters.")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_range[1:], silhouette_scores[1:], marker='o', color='orange', label='Índice de Silhueta')
ax.set_title('Índice de Silhueta para Número de Clusters')
ax.set_xlabel('Número de Clusters (k)')
ax.set_ylabel('Índice de Silhueta')
ax.legend()
st.pyplot(fig)

# Marcar o fim do tempo e calcular a duração
end_time = time.time()
execution_time = end_time - start_time
st.write(f"Tempo de execução: {execution_time:.2f} segundos")