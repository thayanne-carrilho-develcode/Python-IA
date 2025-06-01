import streamlit as st # type: ignore
import json
import requests # type: ignore

#titulo da aplicação
st.title('Modelo de Predição de Salário')

#Inputs

st.write('Quantos meses o profissional está na empresa?')
tempo_na_empresa = st.slider("Meses", min_value=1, max_value=120, value=60, step=1)

st.write('Qual o nivel do profissional na empresa?')
nivel_na_empresa = st.slider("Nivel", min_value=1, max_value=10, value=5, step=1)

#Preparar dados pra API
input_features = {'tempo_na_empresa' : tempo_na_empresa,
                  'nivel_na_empresa': nivel_na_empresa }

# Criar um botão e capturar um evento deste botão para disparar a API
if st.button('Estimar salário'):
    res = requests.post(url = 'http://127.0.0.1:8000/predict', data=json.dumps(input_features))
    
    res_json = json.loads(res.text)
    salario_em_reais = round(res_json['salario_em_reais'], 2)
    st.subheader(f'O salario estimado é de R$ {salario_em_reais}')