#Passo 1: Entendimento do desafio
#Passo 2: Entendimento da área\Empresa
#Passo 3: Extração/obtenção de dados
from re import X
from statistics import linear_regression
from sys import displayhook
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tabela= pd.read_csv('D:/pyton/intensivão python/day 4\/advertising.csv')

#Passo 4: Ajuste de dados(tratamento/limpeza)
#Passo 5: Analise explorátoria
tabela.corr()

from sklearn.model_selection import train_test_split

#Passo 6: Modelagem + algoritímos 
x= tabela[['TV', 'Radio','Jornal']]
y=tabela['Vendas']


x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.1)
#Importou a inteligência artificial
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#cria a inteligência artificial 
modelo_regressaolinear=LinearRegression()
modelo_arvorededecisao=RandomForestRegressor()
#Treina a inteligência artificial
modelo_regressaolinear.fit(x_treino,y_treino)
modelo_arvorededecisao.fit(x_treino,y_treino )

from sklearn.metrics import r2_score
#Previsão dos testes
previsao_regressaolinear= modelo_regressaolinear.predict(x_teste)
previsao_arvorededecisao= modelo_arvorededecisao.predict(x_teste)

(r2_score(y_teste, previsao_regressaolinear))
(r2_score(y_teste, previsao_arvorededecisao))
#Passo 7: Interpretação de resultados
tabela_auxiliar= pd.DataFrame()
tabela_auxiliar['y teste']= y_teste
tabela_auxiliar['arvore de decisão']= previsao_arvorededecisao
tabela_auxiliar['regresssão linear']=previsao_regressaolinear

#Passo 8: Previsão de vendas
nova_tabela = pd.read_csv("D:/pyton/intensivão python/day 4/novos.csv")

previsao = modelo_arvorededecisao.predict(nova_tabela)
print(previsao)
  