#importando bibliotecas
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#Importando dados limpos
df = pd.read_csv('dados/df_13_22.csv')

#separando dados
X = df[['FG%', '3P%', '3PA', '3PM', 'BLKA', 'DREB']]
Y = df['Playoffs']

#split treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.33, 
                                                    stratify=Y,
                                                    random_state = 42)

#configurando detalhes da pagina
st.set_page_config(page_icon="🏀", page_title="NBA-Tera")

#titulo
st.header("Projeto NBA - Tera")

#informações
st.write("""
Prevendo classificação para os playoffs.\n
App que utiliza machine learning para realizar previsões se um time vai ou não para os playoffs da NBA.\n
Fonte: https://www.nba.com/stats/teams
""")

#Cabeçalho
st.subheader('Informações sobre os dados')

#nome do time
time_input = st.sidebar.text_input('Digite o nome do time')
st.write("Time :", time_input)

#dados de input
def get_team_data():
    fg_p = st.sidebar.slider('Fiel Goal (%)', 0.0, 100.0, 40.0)
    p3_p = st.sidebar.slider('3 Point Field Goal (%)', 0.0, 100.0, 35.0)
    p3_a = st.sidebar.slider('3 Point Field Goals Attempted', 1000, 4500, 2000)
    pm3 =  st.sidebar.slider('3 Point Field Goals Made', 0, 2000, 800)
    blka = st.sidebar.slider('Blocks Against', 0, 1000, 380)
    dreb = st.sidebar.slider('Defensive Rebounds', 1500, 3500, 2500)

    team_data = {'FG%' : fg_p,
                '3P%':p3_p,
                '3PA':p3_a,
                '3PM':pm3,
                'BLKA':blka,
                'DREB':dreb
                }

    features = pd.DataFrame(team_data, index=[0])
    return features

team_input_variables = get_team_data()

#mostrando dados do time
st.subheader('Dados do time ')
st.write(team_input_variables)

#treinando o modelo
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

#avaliando o modelo
y_pred_proba = logistic_regression.predict_proba(X_test)[:,1]
y_pred = (y_pred_proba >= 0.5).astype(int)
presicion = precision_score(y_test, y_pred, pos_label=1).round(2)
recall = recall_score(y_test, y_pred, pos_label=1).round(2)
f1_scor = metrics.f1_score(y_test, y_pred, pos_label=1).round(2)

st.subheader('Médidas do modelo')
st.write('Precision :', presicion)
st.write('Recall :', recall)
st.write('F1 score :', f1_scor)

#prevendo com base nas seleções
prediction = logistic_regression.predict(team_input_variables)
st.subheader('Previsão')
st.write('Caso 0 o time não classifica para os playofss caso 1 o time se classifica')
st.write(prediction)