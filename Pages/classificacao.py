import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import Perceptron


# Carregar o dataset
@st.cache_data
def load_data():
    file_path = r'C:/Users/Administrador/Desktop/movies.csv'
    df = pd.read_csv(file_path)
    return df

# Função principal da aplicação
def main():
    st.title('Avaliação do Modelo de Classificação')

    df = load_data()
    st.write("### Dataset")
    st.write(df.head())
    
    # Adicionar opção de seleção da coluna alvo
    target_column = st.selectbox('Selecione a coluna alvo', df.columns)

    if target_column:
        df['success'] = (df['revenue'] > df['revenue'].median()).astype(int)
        X = df.drop(['success'], axis=1, errors='ignore')  # Remover a coluna 'success' se estiver presente
        y = df['success']

        # Codificar colunas categóricas
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le

        # Preencher valores ausentes
        X = X.fillna(X.mean())

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar o modelo
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Exibir resultados
        st.write("### Métricas do Modelo")
        st.write(f"Acurácia: {accuracy:.2f}")
        st.write(f"Precisão: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-score: {f1:.2f}")

        # Visualizar a matriz de confusão
        st.write("### Matriz de Confusão")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Matriz de Confusão')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        st.pyplot(fig)



    perceptron_model = Perceptron(random_state=42)

# Ajuste o modelo aos dados de treinamento
    perceptron_model.fit(X_train, y_train)

    # Faça previsões nos dados de teste
    y_pred_perceptron = perceptron_model.predict(X_test)

    # Avalie o desempenho
    accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
    classification_report_perceptron = classification_report(y_test, y_pred_perceptron)

    # Calcule a matriz de confusão
    cm_perceptron = confusion_matrix(y_test, y_pred_perceptron)

    # Exiba a acurácia e o relatório de classificação
    st.write("Acurácia do Perceptron:", accuracy_perceptron)
    st.text("Relatório de Classificação do Perceptron:\n" + classification_report_perceptron)

    # Exiba a matriz de confusão
    st.write("Matriz de Confusão do Perceptron:")

    # Visualize a matriz de confusão
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_perceptron, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('Matriz de Confusão do Perceptron')
    plt.xlabel('Previsto')
    plt.ylabel('Real')

    # Exiba o gráfico no Streamlit
    st.pyplot(fig)
    
    rf_model = RandomForestClassifier(random_state=42)

# Ajuste o modelo aos dados de treinamento
    rf_model.fit(X_train, y_train)

    # Faça previsões nos dados de teste
    y_pred_rf = rf_model.predict(X_test)

    # Avalie o desempenho
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    classification_report_rf = classification_report(y_test, y_pred_rf)

    # Calcule a matriz de confusão
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    # Exiba a acurácia e o relatório de classificação
    st.write("Acurácia do Random Forest:", accuracy_rf)
    st.text("Relatório de Classificação do Random Forest:\n" + classification_report_rf)

    # Exiba a matriz de confusão
    st.write("Matriz de Confusão do Random Forest:")

    # Visualize a matriz de confusão
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('Matriz de Confusão do Random Forest')
    plt.xlabel('Previsto')
    plt.ylabel('Real')

    # Exiba o gráfico no Streamlit
    st.pyplot(fig)
    

# import streamlit as st
# from imblearn.over_sampling import RandomOverSampler
# from collections import Counter

# # Simulação de dados para o exemplo
# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification

# # Gerar dados de exemplo
# X, y = make_classification(n_classes=3, class_sep=2, weights=[0.1, 0.3, 0.6], n_informative=3, n_redundant=1, flip_y=0, n_features=5, n_clusters_per_class=1, n_samples=1000, random_state=42)

# # Dividir em conjuntos de treinamento e teste
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Inicialize o RandomOverSampler
# ros = RandomOverSampler(random_state=42)

# # Faça o oversampling dos dados de treinamento
# X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# # Função para exibir as informações no Streamlit
# def display_oversampling_results(X_train, y_train, X_train_resampled, y_train_resampled):
#     # Contar as classes antes e depois do oversampling
#     original_counts = Counter(y_train)
#     resampled_counts = Counter(y_train_resampled)

#     st.write("Distribuição das classes antes do oversampling:")
#     st.write(pd.DataFrame.from_dict(original_counts, orient='index', columns=['Número de Amostras']))

#     st.write("Distribuição das classes após o oversampling:")
#     st.write(pd.DataFrame.from_dict(resampled_counts, orient='index', columns=['Número de Amostras']))

#     st.write(f"Número total de amostras antes do oversampling: {len(y_train)}")
#     st.write(f"Número total de amostras após o oversampling: {len(y_train_resampled)}")

#     # Exemplo de visualização
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))

#     sns.histplot(y_train, ax=ax[0], discrete=True)
#     ax[0].set_title('Distribuição das Classes Antes do Oversampling')

#     sns.histplot(y_train_resampled, ax=ax[1], discrete=True)
#     ax[1].set_title('Distribuição das Classes Após o Oversampling')

#     st.pyplot(fig)

# # Chamar a função no Streamlit
# st.title("Oversampling com RandomOverSampler")

# display_oversampling_results(X_train, y_train, X_train_resampled, y_train_resampled)
