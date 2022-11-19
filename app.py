#importando bibliotecas
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#tirando alertas
st.set_option('deprecation.showPyplotGlobalUse', False)

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

container = st.container()
#titulo
container.title("⛹️‍♂️ Projeto NBA - Tera")
container.image('utils/nba.jpg')

#titulo sidebar
st.sidebar.title('Preencha para a previsão:')

#informações
st.write("""
Prevendo classificação para os playoffs.\n
App que utiliza machine learning para realizar previsões se um time vai ou não para os playoffs da NBA.\n
Fonte: https://www.nba.com/stats/teams
""")

#como utilizar
st.info('Edite os dados do seu time na coluna à esquerda e veja abaixo se ele se classifica ou não para os playoffs.', icon= 'ℹ')

#Cabeçalho
st.header('Informações sobre os dados')

#nome do time
time_input = st.sidebar.text_input('Digite o nome do time')
#aviso para preencher o campo do nome
if len(time_input) == 0:
    st.sidebar.warning ('Campo obrigatório')

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
st.subheader('Dados do time:')

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric(label= team_input_variables.columns.values[0], value= team_input_variables['FG%'])
col2.metric(label= team_input_variables.columns.values[1], value= team_input_variables['3P%'])
col3.metric(label= team_input_variables.columns.values[2], value= team_input_variables['3PA'])
col4.metric(label= team_input_variables.columns.values[3], value= team_input_variables['3PM'])
col5.metric(label= team_input_variables.columns.values[4], value= team_input_variables['BLKA'])
col6.metric(label= team_input_variables.columns.values[5], value= team_input_variables['DREB'])

#treinando o modelo
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

#avaliando o modelo
y_pred_proba = logistic_regression.predict_proba(X_test)[:,1]
y_pred = (y_pred_proba >= 0.5).astype(int)
presicion = precision_score(y_test, y_pred, pos_label=1).round(2)
recall = recall_score(y_test, y_pred, pos_label=1).round(2)
f1_scor = metrics.f1_score(y_test, y_pred, pos_label=1).round(2)


st.subheader('Medidas do modelo')
st.markdown("##### O modelo utilizado neste app possui as seguintes métricas:")
st.write('Precision :', presicion)
st.write('Recall :', recall)
st.write('F1 score :', f1_scor)
with st.expander("ℹ️ - Sobre as métricas", expanded=False):
	st.write(
        """     
	-   Precision: Dentre todas as classificações de classe Positivo que o modelo fez, quantas estão corretas. Neste caso, das previsões de times que iriam para os playoffs, quantos realmente foram.
	-   Recall: Porcentagem de dados classificados como positivos comparado com a quantidade real de positivos. Neste caso, do total de times que foram aos playoffs, quantos ao todo a previsão acertou.
	-   F1 Score: métrica que une precision e recall a fim de trazer um número único que determine a qualidade geral do modelo.
	    """
	)
	st.markdown("")

#botão para rodar somente quando trigado
st.header('Previsão')
if st.sidebar.button('Prever', help = 'Click aqui para prevermos se o time se classifica ou não para os playoffs', type = 'primary', disabled = len(time_input) == 0):
    #prevendo com base nas seleções
    prediction = logistic_regression.predict(team_input_variables)
    if prediction == 0:
        st.error(f'❌ Infelizmente {time_input} não vai para os playoffs ')
    else:
        st.success(f'Parabéns {time_input}  vai para os playoffs', icon="✅")