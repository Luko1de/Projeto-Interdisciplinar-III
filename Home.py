import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Home",
    page_icon=":clapper:",
    layout="centered",
    initial_sidebar_state="expanded"
    )

st.markdown("<h1 style='text-align: center; color: white;'>Projeto Interdisciplinar III</h1>", unsafe_allow_html=True)
st.write("Olá, bem-vindo(a) ao MovieBox. O MovieBox é um projeto interdisciplinar que tem como objetivo criar um sistema de recomendação de filmes, utilizando técnicas de aprendizado de máquina. O sistema foi desenvolvido por alunos da Universidade Federal Rural de Pernambuco, para a cadeira Projetos Interdisciplinar III, do curso de Sistemas de Informação.")
