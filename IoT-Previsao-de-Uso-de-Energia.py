#!/usr/bin/env python
# coding: utf-8

# # 1. Problema de Negócio

# O conjunto de dados foi coletado por um 
# período de 10 minutos por cerca de 5 meses. As condições de temperatura e 
# umidade da casa foram monitoradas com uma rede de sensores sem fio ZigBee. 
# Cada nó sem fio transmitia as condições de temperatura e umidade em torno 
# de 3 min. Em seguida, a média dos dados foi calculada para períodos de 10 minutos. 
# 
# Os dados de energia foram registrados a cada 10 minutos com medidores de 
# energia de barramento m. O tempo da estação meteorológica mais próxima do 
# aeroporto (Aeroporto de Chievres, Bélgica) foi baixado de um conjunto de dados 
# públicos do Reliable Prognosis (rp5.ru) e mesclado com os conjuntos de dados 
# experimentais usando a coluna de data e hora. Duas variáveis aleatórias foram 
# incluídas no conjunto de dados para testar os modelos de regressão e filtrar os 
# atributos não preditivos (parâmetros).
# 
# O nosso objetivo é prever o uso de energia armazenado na variavel 'Appliances', dessa forma iremos construir um modelo de Regressão.
# 
# -- Objetivos
# - R^2 superior a 70%
# - RMSE inferior a 25
# - MAE inferior a 15
# - Acuracia superior a 80%
# - Relatar economia total de energia.

# | Feature     | Descrição                                          | Unidade        |
# |-------------|----------------------------------------------------|----------------|
# | date        | Data no formato ano-mês-dia hora:minutos:segundos. |                |
# | Appliances  | Consumo de energia. Variavel Target.               | Wh (Watt-Hora) |
# | lights      | Consumo de energia de luminárias.                  | Wh (Watt-Hora) |
# | T1          | Temperatura na Cozinha.                            | Celsius        |
# | RH1         | Umidade Relativa na Cozinha.                       | %              |
# | T2          | Temperatura na Sala de Estar.                      | Celsius        |
# | RH2         | Umidade Relativa na Sala de Estar.                 | %              |
# | T3          | Temperatura na Lavanderia.                         | Celsius        |
# | RH3         | Umidade Relativa na Lavanderia.                    | %              |
# | T4          | Temperatura no Escritório.                         | Celsius        |
# | RH4         | Umidade Relativa no Escritório.                    | %              |
# | T5          | Temperatura no Banheiro.                           | Celsius        |
# | RH5         | Umidade Relativa no Banheiro.                      | %              |
# | T6          | Temperatura Externa Lado Norte.                    | Celsius        |
# | RH6         | Umidade Relativa Externa Lado Norte.               | %              |
# | T7          | Temperatura na Sala de Passar Roupa.               | Celsius        |
# | RH7         | Umidade Relativa na Sala de Passar Roupa.          | %              |
# | T8          | Temperatura no Quarto do Adolescente.              | Celsius        |
# | RH8         | Umidade Relativa no Quarto do Adolescente.         | %              |
# | T9          | Temperatura no Quarto dos Pais.                    | Celsius        |
# | RH9         | Umidade Relativa no Quarto dos Pais.               | %              |
# | T_out       | Temperatura Externa.                               | Celsius        |
# | Press_mm_hg | Pressão.                                           | mm/hg          |
# | RH_out      | Umidade Relativa Externa.                          | %              |
# | Windspeed   | Velocidade do Vento.                               | m/s            |
# | Visibility  | Visibilidade.                                      | km             |
# | Tdewpoint   | Ponto de Saturação.                                | Celsius        |
# | rv1         | Variável Randômica.                                |                |
# | rv2         | Variável Randômica.                                |                |
# | NSM         | Segundos até a meioa noite                         |                |
# | WeekStatus  | Indicativo de Dia da Semana ou Final de Semana.    |                |
# | Day_of_week | Indicativo de Segunda à Domingo.                   |                |

# # 2. Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sweetviz as sv
import statsmodels.api as sm
import statsmodels.formula.api as smf
import shap
import graphviz

from warnings import simplefilter
from matplotlib.colors import ListedColormap
from math import ceil
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import normaltest, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from smogn import smoter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from holidays import Belgium
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
from sklearn.feature_selection import RFE
from sklearn.tree import export_graphviz
from catboost import CatBoostRegressor
from catboost import Pool, cv
from pickle import dump, load


# In[2]:


# Versões dos pacotes usados neste jupyter notebook
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Herikc Brecher" --iversions')


# ## 2.1 Ambiente

# In[3]:


simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme()


# In[4]:


seed_ = 194
np.random.seed(seed_)


# # 3. Carregamento dos Dados

# In[5]:


# Carregamento do dataset de treino e teste
dtTreino = pd.read_csv('../data/training.csv')
dtTeste = pd.read_csv('../data/testing.csv')


# In[6]:


dtTreino.head()


# In[7]:


dtTeste.head()


# In[8]:


dtFull = pd.concat([dtTreino, dtTeste], axis = 0)


# In[9]:


dtFull.head()


# In[10]:


print(dtTreino.shape, dtTeste.shape, dtFull.shape)


# # 4. Analise Exploratoria

# In[11]:


dtFull.head()


# Possuimos ao todo 19375 observações, unindo o conjunto de treino e teste.

# In[12]:


dtFull.describe()


# A unica feature que aparenta estar no formato errado é a coluna 'Date', essa que é 'datetime' foi carregada como 'object'.

# In[13]:


dtFull.dtypes


# In[14]:


# Copiando para um dataset onde iremos processar os dados
dtProcessado = dtFull.copy()

# Convertendo a coluna 'date' para 'datetime'
dtProcessado['date'] = pd.to_datetime(dtProcessado['date'], format='%Y-%m-%d %H:%M:%S')


# In[15]:


dtProcessado.dtypes


# Agora os dados estão no formato correto, e não tivemos perda de informação.

# In[16]:


dtProcessado.head()


# In[17]:


# Verificando se possui valor missing/NA
print(dtProcessado.isna().sum())


# Colunas como 'date', 'rv1' e 'rv2' possuem valores unicos para cada observação, sendo 1:1. Iremos verificar depois se essas informações são relevantes para o modelo, pois isso pode causar problemas.

# In[18]:


# Verificando valores unicos
print(dtProcessado.nunique())


# In[19]:


# Verificando se possui valores duplicados
print(sum(dtProcessado.duplicated()))


# Para melhor interpretação dos dados, iremos separa eles em variaveis qualitativas e quantitativas.

# In[20]:


qualitativas = ['WeekStatus', 'Day_of_week']
quantitativas = dtProcessado.drop(['WeekStatus', 'Day_of_week', 'date'], axis = 1).columns


# In[21]:


dtProcessado[qualitativas].head()


# In[22]:


dtProcessado[quantitativas].head()


# # 4.2 Geração de plots e insights

# Analisando o grafico abaixo é perceptivel que o consumo de energia nos 'Weekend' são proporcionais aos 'Weekday'. Já que a 'Weekday' representa exatatemente 28.5% de uma semana. Por acaso esse também é o valor do consumo de energia em %.

# In[23]:


# Consumo de energia entre dias da semana e finais de semana
fig = plt.figure(figsize = (15, 10))
plt.pie(dtProcessado.groupby('WeekStatus').sum()['Appliances'], labels = ['Weekday', 'Weekend'], autopct = '%1.1f%%')

plt.savefig('../analises/pizza_energia_weekday_weekend.png')
plt.show()


# É perceptivel que ao longo do periodo da coleta dos dados mantemos oscilações comuns no consumo de energia, provavel que se de por eventos climaticos ao longo do periodo.

# In[24]:


plt.plot(dtProcessado['date'], dtProcessado['Appliances'])


# In[25]:


def scatter_plot_conjunto(data, columns, target):
    # Definindo range de Y
    y_range = [data[target].min(), data[target].max()]
    
    for column in columns:
        if target != column:
            # Definindo range de X
            x_range = [data[column].min(), data[column].max()]
            
            # Scatter plot de X e Y
            scatter_plot = data.plot(kind = 'scatter', x = column, y = target, xlim = x_range, ylim = y_range,                                    c = ['black'])
            
            # Traçar linha da media de X e Y
            meanX = scatter_plot.plot(x_range, [data[target].mean(), data[target].mean()], '--', color = 'red', linewidth = 1)
            meanY = scatter_plot.plot([data[column].mean(), data[column].mean()], y_range, '--', color = 'red', linewidth = 1)


# É perceptivel que as variaveis 'T*' como 'T1', 'T2'... possuem baixa correlação com a variavel target. Onde possuimos concentrações maiores para valores médios, porém ao aumentarem ou diminuirem muito passam a diminuir a 'Appliances'. Já variaveis 'RH_*' possuem uma correlação um pouco maior.

# In[26]:


scatter_plot_conjunto(dtProcessado, quantitativas, 'Appliances')


# ## 4.3 Distribuição dos Dados

# Iremos verificar se os nossos dados possuem uma distribuição Gaussiana ou não. Dessa forma iremos entender quais metodos estatisticos utilizar. Distribuições Gaussianas utilizam de métodos estatisticos paramétricos. Já o contrário utiliza de métodos estatisticos não paramétricos. É importante entender qual método utilizar para não termos uma vissão errada sobre os dados.

# In[27]:


def quantil_quantil_teste(data, columns):
    
    for col in columns:
        print(col)
        qqplot(data[col], line = 's')
        plt.show()


# Olhando os graficos abaixo, possuimos algumas variaveis que não seguem a reta Gaussiana, indicando dados não normalizados, porém para termos certeza, iremos trazer isso para uma representação numerica, onde podemos ter uma maior certeza.

# In[28]:


quantil_quantil_teste(dtProcessado, quantitativas)


# In[29]:


def testes_gaussianos(data, columns, teste):
    
    for i, col in enumerate(columns):
        print('Teste para a variavel', col)
        alpha = 0.05
        
        if teste == 'shapiro':
            stat, p = shapiro(data[col])
        elif teste == 'normal':
            stat, p = normaltest(data[col])           
        elif teste == 'anderson':
            resultado = anderson(data[col])
            print('Stats: %.4f' % resultado.statistic)
            
            for j in range(len(resultado.critical_values)):
                sl, cv = resultado.significance_level[j], resultado.critical_values[j]
                
                if resultado.statistic < cv:
                    print('Significancia = %.4f, Valor Critico = %.4f, os dados parecem Gaussianos. Falha ao rejeitar H0.' % (sl, cv))
                else:
                    print('Significancia = %.4f, Valor Critico = %.4f, os dados não parecem Gaussianos. H0 rejeitado.' % (sl, cv))
            
        if teste != 'anderson':         
            print('Stat = ', round(stat, 4))
            print('p-value = ', round(p, 4))
            #print('Stats = %4.f, p = %4.f' % (stat, p))

            if p > alpha:
                print('Os dados parecem Gaussianos. Falha ao rejeitar H0.')
            else:
                print('Os dados não parecem Gaussianos. H0 rejeitado.')
            
        print('\n')


# # 4.3.1 Teste normal de D'Agostino

# O teste Normal de D'Agostino avalia se os dados são Gaussianos utilizando estatisticas resumidas como: Curtose e Skew.

# Aparentemente os nossos dados não seguem o comportamento Gaussiano, dessa forma iremos ter que tomar medidas estatisticas para amenizar o impacto na hora da modelagem preditiva.

# In[30]:


testes_gaussianos(dtProcessado, quantitativas, teste = 'normal')


# Analisando abaixo o boxplot das variaveis quantitativas, percebemos que algumas variaveis possuem muitos outliers e irão necessitar um tratamento.
# 
# Sendo alguas delas: 'Appliances', 'T1', 'RH_1', 'Visibility', 'RH_5'. Sendo alguns outliers somente para valores maximos e outros para valores minimos.

# In[31]:


# Plot para variaveis quantitativas

fig = plt.figure(figsize = (16, 32))

for i, col in enumerate(quantitativas):
    plt.subplot(10, 3, i + 1)
    dtProcessado.boxplot(col)
    plt.tight_layout()


# Visualizando rapidamente o heatmap, percebemos que existem valores muito proximo de preto e outros muito proximo de branco, valores esses fora da diagonal principal, indicando fortes indicios de multicolinearidade, o que para modelos de regressão são prejudiciais. 
# 
# Um segundo ponto são as variaveis 'rv1' e 'rv2' que possuem correlação 1, de acordo com o nosso dicionario de dados essas variaveis são randomicas, então irão ser removidas do dataset de qualquer maneira. Já o NSM é uma variavel sequencial, que também irá ser removida.

# In[32]:


fig = plt.figure(figsize = (32, 32))

sns.heatmap(dtProcessado[quantitativas].corr(method = 'pearson'), annot = True, square = True)
plt.show()


# Apartir do Sweetviz confirmamos que possuimos muitas variaveis com alta correlação, o que irá gerar Multicolinearidade, para tentarmos amenizar o impacto iremos utilizar de autovetores. 
# 
# Observação: O report foi analisado e anotado insights, porém para melhor compreensão passo a passo dos dados, iremos realizar a analise de forma manual ao longo do notebook.

# In[33]:


# Gerando relatorio de analise do Sweetviz
relatorio = sv.analyze(dtProcessado)
relatorio.show_html('eda_report.html')


# In[34]:


# Remoção de variaveis desnecessárias a primeira vista

dtProcessado = dtProcessado.drop(['rv1', 'rv2'], axis = 1)
quantitativas = quantitativas.drop(['rv1', 'rv2'])


# # 4.4 Avaliando MultiColinearidade

# In[35]:


dtProcessado_Temp = dtProcessado.copy()
dtProcessado_Temp = dtProcessado_Temp.drop(['date', 'Appliances'], axis = 1)

# Capturando variaveis independentes e dependentes
X = dtProcessado_Temp[quantitativas.drop('Appliances')]

# Gerando matriz de correlação e recombinando
corr = np.corrcoef(X, rowvar = 0)
eigenvalues, eigenvectors = np.linalg.eig(corr)


# In[36]:


menor = 999
index = 0
for i, val in enumerate(eigenvalues):
    if val < menor:
        menor = val
        index = i


# In[37]:


print('Menor valor do eigenvalues:', menor, 'Index:', index)


# In[38]:


menorEigenVector = abs(eigenvectors[:, 19])


# In[39]:


for i, val in enumerate(eigenvectors[:, 19]):
    print('Variavel', i,':', abs(val))


# In[40]:


colunas = dtProcessado_Temp.columns


# Analisando as variaveis de indice 11, 19, 21 e 24, aparentam possuir multicolinearidade devido ao seu alto valor absoluto. Porém a sua correlação é baixa, dessa forma iremos aprofundar mais a analise para tomarmos alguma decisão.

# In[41]:


colunas[[11, 19, 21, 24]]


# A variavel 'RH_5' não apresenta um comportamento nitido de correlação com as demais variaveis no scatter_plot. Porém, apresenta uma tendencia pequena de aumento nos valores de 'RH_5' apartir de uma determinada crescente nas variaveis independentes.

# In[42]:


scatter_plot_conjunto(dtProcessado_Temp, ['RH_5', 'RH_9', 'Press_mm_hg', 'Visibility'], 'RH_5')


# Para 'RH_9' temos o mesmo detalhe, não apresenta um comportamento nitido de correlação com as demais variaveis no scatter_plot. Porém, apresenta uma tendencia pequena de aumento nos valores de 'RH_9' apartir de uma determinada crescente nas variaveis independentes. 

# In[43]:


scatter_plot_conjunto(dtProcessado_Temp, ['RH_5', 'RH_9', 'Press_mm_hg', 'Visibility'], 'RH_9')


# Para 'RH_9' temos o mesmo detalhe, não apresenta um comportamento nitido de correlação com as demais variaveis no scatter_plot.

# In[44]:


scatter_plot_conjunto(dtProcessado_Temp, ['RH_5', 'RH_9', 'Press_mm_hg', 'Visibility'], 'Press_mm_hg')


# Para 'Visibility' temos o mesmo detalhe, não apresenta um comportamento nitido de correlação com as demais variaveis no scatter_plot. Porém, apresenta uma tendencia pequena de aumento nos valores de 'Visibility' apartir de uma determinada crescente nas variaveis independentes.

# In[45]:


scatter_plot_conjunto(dtProcessado_Temp, ['RH_5', 'RH_9', 'Press_mm_hg', 'Visibility'], 'Visibility')


# Analisando as variaveis que apontam possuir alguma Multicolinearidade, até o momento não conseguimos identificar com alta confiança se isso se é verdade. Iremos utilizar VIF para verificar o impacto das variaveis de maneira mais automatizada e acertiva. 
# 
# É esperado que valores de VIF = 1 ou proximos a 1 não possua correlação com outras variaveis independentes. Caso VIF ultrapasse valores como 5 ou até 10, possuimos fortes indicios de multicolinearidade entre as variaveis com tal valor.

# In[46]:


def calcular_VIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif


# In[47]:


dtProcessado_Temp = dtProcessado.copy()
dtProcessado_Temp = dtProcessado_Temp.drop(['Appliances'], axis = 1)

# Capturando variaveis independentes
X = dtProcessado_Temp[quantitativas.drop('Appliances')]


# Analisando abaixo é perceptivel que a unica variavel com valor baixo para VIF é 'lights'. Assim iremos necessitar de um grande tratamento sobre as variaveis.

# In[48]:


calcular_VIF(X)


# Abaixo realizamos a primeira tentativa removendo variaveis com VIF > 2000. Porém ainda possuimos alto indice de MultiColinearidade. Iremos remover variaveis com VIF > 1000.

# In[49]:


X_temp = X.drop(['T1', 'T2', 'T9', 'Press_mm_hg'], axis = 1)
calcular_VIF(X_temp)


# Ainda com a remoção de VIF > 1000 estamos com fortes indicios de MultiColinearidade, iremos aumentar o nosso range para VIF > 250.

# In[50]:


X_temp = X.drop(['T1', 'T2', 'T9', 'Press_mm_hg', 'RH_1', 'RH_3', 'RH_4', 'T7'], axis = 1)
calcular_VIF(X_temp)


# Após uma remoção massiva de variaveis continuamos com alta taxa de MultiColinearidade, iremos remover mais algumas variaveis. Porém, é esperado que iremos fazer mais testews nas variaveis para verificar seu valor para a predição.

# In[51]:


X_temp = X.drop(['T1', 'T2', 'T9', 'Press_mm_hg', 'RH_1', 'RH_3', 'RH_4', 'T7', 'T3', 'T4', 'T5', 'T8', 'RH_9', 'RH_2',                'RH_7', 'RH_8'], axis = 1)
calcular_VIF(X_temp)


# Após removermos 21 variaveis das 25 variaveis quantitativas, conseguimos reduzir o VIF para um valor aceitavel, porém é necessário verificar o impacto dessa remoção e se a tecnica para sua remoção foi utilizada da maneira correta.

# In[52]:


X_temp = X.drop(['T1', 'T2', 'T9', 'Press_mm_hg', 'RH_1', 'RH_3', 'RH_4', 'T7', 'T3', 'T4', 'T5', 'T8', 'RH_9', 'RH_2',                'RH_7', 'RH_8', 'RH_5', 'T_out', 'Visibility', 'RH_out', 'T6'], axis = 1)
calcular_VIF(X_temp)


# Iremos verificar o valor das variaveis utilizando tanto o dataset original quanto com as variaveis removidas apartir do calculo de VIF. Para isso iremos utilizar um modelo base de regressão lienar do StatsModels.

# In[53]:


# Carregando todas variaveis com exceção da 'Target', iremos adicionar a constante exigida pelo modelo
X = dtProcessado_Temp.copy().drop('date', axis = 1)[quantitativas.drop('Appliances')]
Xc = sm.add_constant(X)

y = dtProcessado['Appliances'].values


# In[54]:


# Criando e treinando modelo
modelo = sm.OLS(y, Xc)
modelo_v1 = modelo.fit()


# Analisando abaixo percemos que o nosso modelo representa somente 16.5% da variancia dos dados, R-squared. Também verificamos que o valor F esta muito alto, sendo inviavel utilizar pare predição. Nosso AIC e BIC já indicam um valor muito alto, o que esta nos sinalizando MultiColinearidade.
# 
# Também possuimos variaveis como 'T1', 'RH_4', 'T5', 'RH_5', 'T7' e 'Press_mm_hg' com valores de p > 0.05, indicando que não possui relação com a predição de variaveis. 
# 
# Possuimos um 'Omnibus' muito alto, visto que o ideal seria 0. Já Skew e Kurtosis possuem valores relativamente normais para dos que não foram tratados. Já o Durbin-Watson está com um valor relativamente proximo do normal (entre 1 e 2), porém esta indicando que os nossos dados podem estar concentrados, a medida que o ponto de dados aumenta o erro relativo aumenta. Por ultimo, estamos com um 'Conditiom Number' extremamente alto, indicando mais ainda a nossa multicolienaridade.

# In[55]:


# Visualizando resumo do modelo
modelo_v1.summary()


# Abaixo iremos criar o modelo novamente porém a redução de variaveis implicadas pelo p value e VIF.

# In[56]:


# Carregando variaveis com exceção
Xc = sm.add_constant(X_temp)

y = dtProcessado['Appliances'].values


# In[57]:


# Criando e treinando modelo
modelo = sm.OLS(y, Xc)
modelo_v2 = modelo.fit()


# Treinando o modelo com a remoção de variaveis, notamos que tivemos uma grande redução no R-Squared, trazendo uma significancia de apenas 6% da variancia dos nossos dados. Devemos tentar escolher variaveis melhores para o nosso modelo, consequentemente levou ao aumento do valor F e diminuição do 'Log-Likelihood'.
# 
# Outros valores permaneceram com resultados semelhantes, com exceção de 'Conditiom Number' que reduziu drasticamente ao ponto de não sofrermos mais multicolinearidade.
# 
# Iremos ter que avaliar melhor quais variaveis utilizar para o nosso modelo, afim de reduzir a MultiColinearidade sem perder variaveis de valor.

# In[58]:


# Visualizando resumo do modelo
modelo_v2.summary()


# ## 4.5 Simetria dos Dados

# ### 4.5.1 Skewness

# Esperamos valores de Skewness proximo de 0 para uma simetria perfeita.
# 
# ![image.png](attachment:image.png)

# - Se skewness é menor que −1 ou maior que +1, a distribuição é 'highly skewed'.
# 
# - Se skewness esta entre −1 e −½ ou entre +½ e +1, a distribuição é 'moderately skewed'.
# 
# - Se skewness esta entre −½ e +½, a distribuição é aproximadaente simetrica.

# Olhando primeiramente para o Skewness, possuimos variaveis com alta simetria o que é muito bom para os algoritmos de Machine Learnign em geral. Porém possuimo a variavel 'lights' com um simetria muito acima de 0. Já as outras variaveis possuem um Skewness aceitavel, valores maiores que 0.5 ou menores que -0.5 indicam que a simetria já começa a se perder, porém ainda é aceitavel.

# In[59]:


print(dtProcessado[quantitativas].skew(), '\nSoma:', sum(abs(dtProcessado[quantitativas].skew())))


# ### 4.5.2 Histograma

# In[60]:


def hist_individual(data, columns, width = 10, height = 15):
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    
    columns_adjust = ceil(len(columns) / 3)
    
    for i, column in enumerate(columns):
        ax = fig.add_subplot(columns_adjust, 3, i + 1)
        data[column].hist(label = column)
        plt.title(column)
        
    plt.tight_layout()  
    plt.show()


# Abaixo iremos verificar o histograma das variaveis. Porém para visualizarmos de uma melhor forma iremos separar em grupos de plots abaixo. Fica perceptivel que 'Appliances' e'lights' não possuem simetria, devido a sua alta concentração entre 0 e 10. Porém variaveis como 'T1' e 'T4' possuem alta simetria.

# In[61]:


hist_individual(dtProcessado, quantitativas[0:9])


# In[62]:


hist_individual(dtProcessado, quantitativas[9:18])


# In[63]:


hist_individual(dtProcessado, quantitativas[18:27])


# ### 4.5.3 Exceço de Kurtosis

# ![image.png](attachment:image.png)

# Mesokurtic -> Kurtosis ~= 0: Distribuição normal.
# 
# Leptokurtic -> Kurtosis > 0: Valores proximos a media ou dos extremos.
# 
# Platykurtic -> Kurtosis < 0: Valores muito espalhados.

# É perceptivel que variaveis como 'Appliances', 'lights' e 'RH_5' claramente estão distantes de uma distribuição normal, porém outras variaveis se aproximam de uma distribuição Gaussiana, com valores maiores que 3 e menores que 4. Também é perceptivel qque possuimos muitas variaveis com o comportamento de uma 'Platykurtic', ou seja valores muito espalhados. 

# In[64]:


print(dtProcessado[quantitativas].kurtosis(), '\nSoma:', sum(abs(dtProcessado[quantitativas].kurtosis())))


# ## 4.6 Analise Temporal

# ### 4.6.1 Pre-Processamento colunas temporais

# Para realizarmos uma analise eficiente das variaveis temporais, iremos transforma-las, adicionando coluna de 'Month', 'Day', 'Hour', e convertendo coluna 'Day_of_week' e 'WeekStatus' para numericas.

# In[65]:


# Renomeando coluna WeekStatus para Weekend
dtProcessado = dtProcessado.rename(columns = {'WeekStatus': 'Weekend'})


# In[66]:


# Dia de semana = 0, final de semana = 1
dtProcessado['Day_of_week'] = dtProcessado['date'].dt.dayofweek
dtProcessado['Weekend'] = 0

dtProcessado.loc[(dtProcessado['Day_of_week'] == 5) | (dtProcessado['Day_of_week'] == 6), 'Weekend'] = 1


# In[67]:


# Criando colunan de Mês, Dia e Hora
dtProcessado['Month'] = dtProcessado['date'].dt.month
dtProcessado['Day'] = dtProcessado['date'].dt.day
dtProcessado['Hour'] = dtProcessado['date'].dt.hour


# In[68]:


dtProcessado.head()


# ### 4.6.2 Analise Temporal de Gasto Energia

# Abaixo percebemos que o gasto energetico por dia da semana tende a iniciar alto na Segunda / 0 em 115 Wh, passando por um queda para 80-85 Wh até Quinta, voltando a subir até os 105 Wh na Sexta e Sabado. Por ultimo, voltamos a uma queda por volta dos 85 Wh no Domingo.
# 
# Apartir desse cenario, podemos visualizar que a Segunda passa a ser um dia onde as pessoas gastam maior energia, talvez por estar começando a semana com maior foco em atividades que leval a gasto de energia eletrica. Com uma queda ao longo da semana, que volta a subir proximo ao final de semana, onde temos dias de descanso que passam a acontecer em casa, e por ultimo no domingo onde tende a ser dias para saida de familia.
# 
# Claro que o cenario acima é somente uma hipotese, porém representar a realidade de algumas pessoas, para um melhor entendimento poderia ser feito uma pesquisa do estilo de vida do cidadões de onde foi retirado o dataset.

# In[69]:


fig, ax = plt.subplots(figsize = (10, 5))
dtProcessado.groupby('Day_of_week').mean()['Appliances'].plot(kind = 'bar')

ax.set_title('Média de Watt-Hora por Dia')
ax.set_ylabel('Watt-Hora')
ax.set_xlabel('Dia da Semana')

plt.savefig('../analises/barra_dia_semana_media_wh.png')
plt.plot()


# In[70]:


fig, ax = plt.subplots(figsize = (10, 5))
dtProcessado.groupby('Day_of_week').sum()['Appliances'].plot(kind = 'bar')

ax.set_title('Soma de Watt-Hora por Dia')
ax.set_ylabel('Watt-Hora')
ax.set_xlabel('Dia da Semana')

plt.savefig('../analises/barra_dia_semana_soma_wh.png')
plt.plot()


# É analisado que o gasto de hora começa a subir aproximadamente as 6 da manhã até as 11 horas da manhã chegar em um pico de 130 Wh, depois temos uma queda até os 100 Wh e voltamos a subir pro volta das 15 horas da tarde, até chegar ao pico de 180 Wh as 18 horas, apartir desse momento vamos caindo o nivel de energia até chegar abaixo dos 60 Wh as 23 horas.

# In[71]:


fig, ax = plt.subplots(figsize = (10, 5))
dtProcessado.groupby('Hour').mean()['Appliances'].plot(kind = 'line')

ax.set_title('Media de Watt-Hora por Hora')
ax.set_ylabel('Watt-Hora')
ax.set_xlabel('Hora do Dia')

plt.savefig('../analises/linha_hora_media_wh.png')
plt.plot()


# In[72]:


fig, ax = plt.subplots(figsize = (10, 5))
dtProcessado.groupby('Hour').sum()['Appliances'].plot(kind = 'line')

ax.set_title('Soma de Watt-Hora por Hora')
ax.set_ylabel('Watt-Hora')
ax.set_xlabel('Hora do Dia')

plt.savefig('../analises/linha_hora_soma_wh.png')
plt.plot()


# In[73]:


# Criando copia do data set
dtProcessado_temporal = dtProcessado.copy()

# Set da data como index
dtProcessado_temporal.index = dtProcessado_temporal['date']
dtProcessado_temporal = dtProcessado_temporal.drop('date', axis = 1)


# In[74]:


dtProcessado_temporal.head()


# In[75]:


# Calculando media por data
dtProcessado_Dia = dtProcessado_temporal['Appliances'].resample('D').mean()

# Calculando media até a data atual
media_momentanea = pd.Series(                        [np.mean(dtProcessado_Dia[:x]) for x in range(len(dtProcessado_Dia))]                        )

media_momentanea.index = dtProcessado_Dia.index


# Percebe-se que o gasto de energia vem oscilando bastante entre os meses, porém mantem uma média constante devido ao alto volume de dados. Talvez a coluna 'Mês' e 'Dia' possuam uma representatividade interessante para o modelo.

# In[76]:


fig, ax = plt.subplots(figsize = (15, 5))
plt.plot(dtProcessado_Dia, label = 'Gasto Energetico Diario')
plt.plot(media_momentanea, label = 'Media de Gasto Energetico')
plt.legend()
plt.xticks(rotation = 90)

plt.savefig('../analises/linha_media_wh_data.png')
ax.set_title('Gasto Médio de Energia Diário em Watt-Hora');


# # 5. Pre-Processamento

# ## 5.1 Removendo Colunas Desnecessárias

# Abaixo iremos remover as colunas que mostraram se sem valor durante a analise exploratoria.

# In[77]:


dtProcessado = dtProcessado.drop(['date'], axis = 1)


# In[78]:


dtProcessado.head()


# ## 5.2 Detectando Outliers

# In[79]:


def boxplot_individuais(data, columns, width = 15, height = 8):
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
    fig.set_figheight(8)
    fig.set_figwidth(15)
    
    columns_adjust = ceil(len(columns) / 3)
    
    for i, column in enumerate(columns):
        ax = fig.add_subplot(columns_adjust, 3, i + 1)
        sns.boxplot(x = data[column])
        
    plt.tight_layout()  
    plt.show()


# ![image.png](attachment:image.png)

# É perceptivel que com exceção das variaveis: 'RH_4', 'RH_6', 'T7' e 'T9', todas as outras variaveis possuem outliers. Alguns possuem somentne acima do limite inferior, outras apenas do limite superior. Ainda poussimos os casos de variaveis que possuem em ambos os limites.
# 
# Para tratar os outliers iremos utilizar a tecnnica de IQR, iremos mover os dados abaixo do limite inferior para o limite inferior, já para o limite superior iremos mover os dados acima do mesmo para o limite superior.

# In[80]:


boxplot_individuais(dtProcessado, quantitativas[0:9])


# In[81]:


boxplot_individuais(dtProcessado, quantitativas[9:18])


# In[82]:


boxplot_individuais(dtProcessado, quantitativas[18:27])


# ## 5.3 Tratando Outliers

# In[83]:


def calcular_limites_IQR(column):
    # Calcular Q1 e Q3 do array
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    return limite_inferior, limite_superior

def aplicar_IQR_coluna(column, superior, inferior):
    limite_inferior, limite_superior = calcular_limites_IQR(column)
    
    if inferior:
        column = [limite_inferior if x < limite_inferior else x for x in column]
        
    if superior:      
        column = [limite_superior if x > limite_superior else x for x in column]
    
    return column

def aplicar_IQR(data, columns = [], superior = True, inferior = True):
    
    if len(columns) == 0:
        especificar = False
    else:
        especificar = True
    
    for i, column in enumerate(data.columns):
        if especificar:
            if column in columns:
                data[column] = aplicar_IQR_coluna(data[column], superior, inferior)
        else:
            data[column] = aplicar_IQR_coluna(data[column], superior, inferior)
            
    return data


# Dataset antes da aplicação do IQR para correção de outliers.

# In[84]:


dtProcessado.describe()


# In[85]:


dtProcessado_IQR = dtProcessado.copy()
dtProcessado_IQR = aplicar_IQR(dtProcessado_IQR, columns = dtProcessado_IQR.columns.copy().drop(['lights',                                                                    'Weekend', 'Day_of_week', 'Month', 'Day', 'Hour']))


# Dataset após aplicação do IQR para correção de outliers. Percebe-se que valores minimos e maximos passaram a ser muito mais realistas, também é perceptivel mudanças na média. Considerando que temos mais de 19 mil registros, uma mudança na média passa a ser muito significativo.
# 
# Observação: Não foi aplicado IQR em 'lights' por uma baixa concentração de outliers, também ocorre que ao aplicar IQR em 'lights', todos os valores são zerados.

# In[86]:


dtProcessado_IQR.describe()


# ## 5.4 Feature Scaling

# ### 5.4.1 Aplicando Normalização

# Para os algoritmos que iremos utilizar como SVM, XGBoost e Regressão Logística Multilinear a normalização se mostra mais relevante. Como nossas variaveis 

# In[87]:


# Normalização dos dados
scaler = StandardScaler()
processado_IQR_normalizado = dtProcessado_IQR.copy()
processado_IQR_normalizado[quantitativas.drop('Appliances')] = scaler.fit_transform(                                                                    dtProcessado_IQR[quantitativas.drop('Appliances')])


# In[88]:


'''
# Normalização dos dados
scaler = StandardScaler()
processado_IQR_normalizado = dtProcessado_IQR.copy()
processado_IQR_normalizado[quantitativas] = scaler.fit_transform(dtProcessado_IQR[quantitativas])
'''


# Após realizar a normalização dos dados, iremos revisitar algumas metricas como skewness, kurtose e boxplot stats.

# In[89]:


dtProcessado_IQR_normalizado = pd.DataFrame(processado_IQR_normalizado.copy(), columns = dtProcessado_IQR.columns)


# In[90]:


dtProcessado_IQR_normalizado.head()


# ### 5.4.2 Analisando Dados Pós Normalização

# Verificando novamente o skewness, tivemos um aumento consideravel na simetria dos dados, conseguimos reduzir o nosso Skewness total pela metade, o que deve levar a melhores resultados para os algoritmos. Iremos realizar a analise de suas vantagens posteriormente na aplicação dos algoritmos.

# In[91]:


print(dtProcessado_IQR_normalizado.skew()[quantitativas],      '\nSoma:', sum(abs(dtProcessado_IQR_normalizado[quantitativas].skew())))


# Verificando novamente a Kurtosis possuimos uma perspectiva muito melhor, conseguimos ajustar as respectivas kurtosis para proximo de 3, trazendo uma distribuição normal Gaussiana, isso se da pela normalização dos dados. A soma ideal da Kurtosis para as nossas 27 variaveis seria 0, chegamos em um valor bem proximo.

# In[92]:


print(dtProcessado_IQR_normalizado[quantitativas].kurtosis(),      '\nSoma:', sum(abs(dtProcessado_IQR_normalizado[quantitativas].kurtosis())))


# Percebemos uma melhora significativa no histograma abaixo das variaveis, com exceção de lights que manteve um skewness alto.

# In[93]:


hist_individual(dtProcessado_IQR_normalizado, quantitativas[0:9])


# In[94]:


hist_individual(dtProcessado_IQR_normalizado, quantitativas[9:18])


# In[95]:


hist_individual(dtProcessado_IQR_normalizado, quantitativas[18:27])


# Já em relação aos outliers, com a aplicação de IR e correçõs nas escalas dos dados conseguimos uma redução perceptivel.
# 
# Observação: Não foi aplicado correção de outliers por IQR na variavel 'lights'.

# In[96]:


boxplot_individuais(dtProcessado_IQR_normalizado, quantitativas[0:9])


# In[97]:


boxplot_individuais(dtProcessado_IQR_normalizado, quantitativas[9:18])


# In[98]:


boxplot_individuais(dtProcessado_IQR_normalizado, quantitativas[18:27])


# ## 5.5 Incremento nas Features

# Aqui iremos acrescentar mais uma variavel no nosso dataset que irá merecer uma analise na etapa de Feature Selection. Iremos acrescentar uma Feature do tipo booleana para os feriados no de coleta dos dados.

# In[100]:


feriados = []

# Criando lista com todos feriados do ano em que o dataset foi gerado
for data in Belgium(years = [2016]).items():
    feriados.append(data)


# In[101]:


# Converter para dataframe e renomear colunas
dtferiados = pd.DataFrame(feriados)
dtferiados.columns = ['data', 'feriado']


# In[102]:


dtferiados.head()


# In[103]:


# Criar uma copia do dataset original para recuperar a coluna 'date', desconsiderando horario
dtTemp = dtFull.copy()
dtTemp['date'] = pd.to_datetime(dtTemp['date'], format='%Y-%m-%d %H:%M:%S').dt.date


# In[104]:


def isHoliday(row):
    
    # Verifica se a data da linha atual esta no dataframe de feriados
    holiday = dtferiados.apply(lambda x: 1 if (row['date'] == x['data']) else 0, axis = 1)
    
    holiday = sum(holiday)
    
    if holiday > 0:
        holiday = 1
    else:
        holiday = 0
    
    return holiday


# In[105]:


# Preenche a coluna feriados do dataframe temporario
dtTemp['Holiday'] = dtTemp.apply(isHoliday, axis = 1)


# In[106]:


# Copia a coluna de feriados do dataframe temporario para o novo
dtProcessado_incremento = dtProcessado_IQR_normalizado.copy()
dtProcessado_incremento['Holiday'] = dtTemp['Holiday'].copy().values


# Como verificado abaixo criamos uma variavel boolean para os dias que forem feriado, onde pode ocorrer um aumento do consumo de energia. 

# In[107]:


dtProcessado_incremento.head()


# Por ultimo iremos remover a variavel 'lights' por não fazer sentido estar no modelo, visto que a mesma apresenta o consumo de Wh das fontes luz da residencia, assim nos indicando um pouco do consumo de energia.

# In[109]:


dtFinal = dtProcessado_incremento.drop('lights', axis = 1)


# # 6. Feature Selecting

# Após uma densa etapa de analise exploratoria e pre-processamento iremos iniciar a etapa de seleção de variaveis, onde iremos ter que trabalhar densamente para eliminar multicolinearidade escolher variaveis que trazem valor para o nosso problema.
# 
# Sobre a regressão Lasso e Ridge, iremos utilizar a Lasso com uma das alternativas para medir a importancia das variaveis. Já a ressão Ridge, iremos utilizar durante a modelagem preditiva para tentar aumentar a importancia das variaveis corretas.

# In[110]:


dtFinal.columns


# ## 6.1 Select From Model - Random Forest

# Iremos utilizar o SelectModel para secionarmos as variaveis baseadas em sua importância, posteriormente iremos realizar o plot de importância por variavel.

# In[111]:


X_fs = dtFinal.drop(['Appliances'], axis = 1)
y_fs = dtFinal['Appliances'].values


# In[112]:


seleciona_fs = SelectFromModel(RandomForestRegressor())
seleciona_fs.fit(X_fs, y_fs)


# In[113]:


variaveis = X_fs.columns[seleciona_fs.get_support()]


# In[114]:


print(variaveis)


# ## 6.2 Random Forest - Feature Importance

# Agora iremos utilizar o Random Forest na sua forma pura, sem hiperparametros. Essa forma é um pouco perigosa pois pode gerar vies do modelo, por isso iremos testar posteriormente com um modelo diferente.

# In[115]:


modelo_fs_v1 = RandomForestRegressor()
modelo_fs_v1.fit(X_fs, y_fs)


# In[116]:


index_ordenado_fs_v1 = modelo_fs_v1.feature_importances_.argsort()


# Analisando, possuimos a variavel 'NSM' com a maior importancia muito a frente, seguido por 'Hour' e 'Lights'. O SelectModel analisou que as melhores variaveis seriam as: 'lights', 'T3', 'RH_3', 'T8', 'Press_mm_hg', 'NSM' e 'Hour'.
# 
# Dessa forma escolhendo as 7 variaveis com maior importancia.

# In[117]:


plt.barh(dtFinal.drop(['Appliances'], axis = 1).columns[index_ordenado_fs_v1],         modelo_fs_v1.feature_importances_[index_ordenado_fs_v1])


# ## 6.3 Regressão LASSO

# Iremos utilizar a Regressão LASSO para minimizar variaveis, assim podemos diminuir a nossa dimensionalidade e multicolinearidade, de forma que o modelo se torne mais generalizado.

# In[118]:


# Função para calcular o RMSE
def rmse_cv(modelo, x, y):
    rmse = np.sqrt(-cross_val_score(modelo, 
                                    x, 
                                    y, 
                                    scoring = "neg_mean_squared_error", 
                                    cv = 5))
    return(rmse)


# In[119]:


# Criando modelo LASSO, com lista de alphas e executanndo em CV
modelo_fs_v2 = LassoCV(alphas = [10, 1, 0.1, 0.01, 0.001])
modelo_fs_v2.fit(X_fs, y_fs)


# In[120]:


# Calculando RMSE de todos os CV
rmse = rmse_cv(modelo_fs_v2, X_fs, y_fs)


# In[121]:


# Print valor medio, maximo, minimo
print(rmse.mean(), max(rmse), min(rmse))


# In[122]:


# Coeficientes LASSO
coef = pd.Series(modelo_fs_v2.coef_, index = X_fs.columns)


# In[123]:


coef.sort_values().tail(15)


# In[124]:


# Plotando importancia das variaveis
imp_coef_fs = pd.concat([coef.sort_values().head(15), coef.sort_values().tail(15)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef_fs.plot(kind = "barh")
plt.title("Coeficientes Modelo LASSO")


# ## 6.4 Recursive Feature Elimination (RFE) - Linear SVR

# Como quarto metodo, iremos utilizar RFE, onde em geral apresenta bons resultados para combater multicolinearidade que é o nosso principal problema nesse dataset. Porém, o seu tempo de execução pode ser muito grande.

# In[125]:


# Criando modelo de SVM para regressão
modelo_v4 = LinearSVR(max_iter = 3000)
rfe = RFE(modelo_v4, n_features_to_select = 8)


# In[126]:


# Treinando RFE
rfe.fit(X_fs, y_fs)


# In[127]:


print('Features Selecionadas: %s' % rfe.support_)
print("Feature Ranking: %s" % rfe.ranking_)


# In[128]:


variaveis_v4 = [X_fs.columns[i] for i, col in enumerate(rfe.support_) if col == True]


# In[129]:


print(variaveis_v4)


# In[130]:


X_fs[variaveis_v4].head()


# ## 6.5 Analisando Seleção

# In[131]:


def avalia_modelo(modelo, x, y):  
    preds = modelo.predict(x)
    
    erros = abs(preds - y)
    mape = 100 * np.mean(erros / y)
    r2 = 100*r2_score(y, preds)
    acuracia = 100 - mape
    mse = mean_squared_error(y, preds, squared = True)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared = False)
    
    print(modelo,'\n')
    print('R^2                 : {:0.2f}%' .format(r2))
    print('Acuracia            : {:0.2f}%'.format(acuracia))
    print('MAE                 : {:0.2f}'.format(mae))
    print('MSE                 : {:0.2f}'.format(mse))
    print('RMSE                : {:0.2f}\n'.format(rmse))


# ### 6.5.1 Random Forest

# In[132]:


# Selecionando variaveis do RandomForestRegressor
X_sel_fs_v1 = X_fs[variaveis]


# In[133]:


x_train, x_test, y_train, y_test = train_test_split(X_fs, y_fs, test_size = .3, random_state = seed_)


# Primeiramente iremos avaliar o modelo com todas as variaveis utilizadas durante a sua construção.

# In[134]:


# Criando o modelo com todas variaveis
modelo_sel_fs_v1 = RandomForestRegressor()
modelo_sel_fs_v1.fit(x_train, y_train)


# In[135]:


avalia_modelo(modelo_sel_fs_v1, x_test, y_test)


# In[136]:


x_train, x_test, y_train, y_test = train_test_split(X_sel_fs_v1, y_fs, test_size = .3, random_state = seed_)


# In[137]:


# Criando o modelo com variaveis selecionodas pelo RandomForestRegressor
modelo_sel_fs_v2 = RandomForestRegressor()
modelo_sel_fs_v2.fit(x_train, y_train)


# Avaliando o modelo utilizando somente 6 colunas, mantivemos um R^2 de 70% com um aumento para 23 do RMSE. Apesar do modelo ser um pouco pior, aumentamos a nossa generalização em muito, visto que passamos de 31 variaveis para 6 variaveis.

# In[138]:


avalia_modelo(modelo_sel_fs_v2, x_test, y_test)


# In[139]:


i = 20
x_temp = x_test.iloc[i]
x_temp = pd.DataFrame(x_temp).T
y_temp = y_test[i]


# In[140]:


pred = modelo_sel_fs_v2.predict(x_temp)


# In[141]:


print('Previsto:', pred,'Real:', y_temp)


# ### 6.5.2 LASSO

# In[142]:


X_sel_fs_v2 = X_fs[['RH_1', 'T3', 'T6', 'T8', 'RH_3']]


# In[143]:


x_train, x_test, y_train, y_test = train_test_split(X_fs, y_fs, test_size = .3, random_state = seed_)


# Primeiramente iremos avaliar o modelo com todas as variaveis utilizadas durante a sua construção.

# In[144]:


# Criando modelo LASSO, com todas variaveis
modelo_sel_fs_v3 = LassoCV(alphas = [10, 1, 0.1, 0.01, 0.001])
modelo_sel_fs_v3.fit(x_train, y_train)


# In[145]:


avalia_modelo(modelo_sel_fs_v3, x_test, y_test)


# In[146]:


x_train, x_test, y_train, y_test = train_test_split(X_sel_fs_v2, y_fs, test_size = .3, random_state = seed_)


# In[147]:


# Criando modelo LASSO, com variaveis selecionadas
modelo_sel_fs_v4 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
modelo_sel_fs_v4.fit(x_train, y_train)


# Analisando o modelo é perceptivel que ele não conseguiu representar os dados da forma adequada, mantendo um R^2 abaixo de 50%, com acuracia pouco acima de 50%.

# In[148]:


avalia_modelo(modelo_sel_fs_v4, x_test, y_test)


# ### 6.5.3 RFE - Linear SVR

# In[149]:


# Selecionando variaveis do RandomForestRegressor
X_sel_fs_v3 = X_fs[variaveis_v4]


# In[150]:


x_train, x_test, y_train, y_test = train_test_split(X_fs, y_fs, test_size = .3, random_state = seed_)


# In[151]:


# Criando o modelo com todas variaveis
modelo_sel_fs_v3 = LinearSVR(max_iter = 3000)
modelo_sel_fs_v3.fit(x_train, y_train)


# Analisando o modelo com todas variaveis, o deu desempenho não aparenta ser bom visto que manteve um R^2 abaixo de 50% e alto RMSE, apesar disso apresentou uma acuracia superior ao modelo de regressão LASSO.

# In[152]:


avalia_modelo(modelo_sel_fs_v3, x_test, y_test)


# In[153]:


x_train, x_test, y_train, y_test = train_test_split(X_sel_fs_v3, y_fs, test_size = .3, random_state = seed_)


# In[154]:


# Criando o modelo com todas variaveis
modelo_sel_fs_v3 = LinearSVR(max_iter = 3000)
modelo_sel_fs_v3.fit(x_train, y_train)


# In[155]:


avalia_modelo(modelo_sel_fs_v3, x_test, y_test)


# Realizando a analise acima é perceptivel que as variaveis que aparentam trazer mais representatividade para o nosso modelo são as do modelo de Random Forest, esse que sugeriu utilizar as seguintes variaveis:
# 
# 'T3', 'RH_3', 'T8', 'Press_mm_hg', 'NSM' e 'Hour'
# 
# Analisando as variaveis acima:
# 'T3' -> Mede a temperatura em graus celsius na lavanderia, a lavanderia constuma possuir equipamentos que consomem um nivel de energia significativamente maior do que outros eletrodomesticos, assim também aumenta o nivel de calor no comodo.
# 
# 'RH_3' -> Umidade relativa na lavanderia, indicando aumento da umidade no ambiente, também por conta dos eletrodomesticos utilizados no ambiente.
# 
# 'T8' -> Temperatura no quarto do adolescente.
# 
# 'Press_mm_hg' -> Pressão.
# 
# 'NSM' -> Quantos segundos faltam para a meia noite, visto que quando mais proximo da meia noite menor o consumo de energia.
# 
# 'Hour' -> Semelhante a 'NSM' porém indicando a forma do dia de forma mais especifica, visto que a hora influencia diretamente no consumo de energia.

# # 7. Modelagem Preditiva

# ## 7.1 Definindo Ambiente

# In[156]:


# Separando em variaveis preditivas e target 
#X = dtFinal[variaveis]
X = dtFinal[['T3', 'RH_3', 'T8', 'Press_mm_hg', 'NSM', 'Hour']]
y = dtFinal['Appliances'].values


# In[157]:


X.head()


# In[158]:


y


# In[159]:


# Separando em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed_)


# In[160]:


def reportModeloRegressao(modelo, x_teste, y_teste, x_treino = [], y_treino = [], report_treino = False):  
    y_pred = modelo.predict(x_teste)
    
    erros = abs(y_pred - y_teste)
    mape = 100 * np.mean(erros / y_teste)
    r2 = 100*r2_score(y_teste, y_pred)
    r2_ajustado = 1 - (1 - r2) * (len(y_teste) - 1) / (len(y_teste) - x_teste.shape[1] -1)
    acuracia = 100 - mape
    mse = mean_squared_error(y_teste, y_pred, squared = True)
    mae = mean_absolute_error(y_teste, y_pred)
    rmse = mean_squared_error(y_teste, y_pred, squared = False)
    
    print(modelo,'\n')
    print('Dados de teste')
    print('R^2                 : {:0.2f}%' .format(r2))
    print('R^2 Ajustado        : {:0.2f}%' .format(r2_ajustado))
    print('Acuracia            : {:0.2f}%'.format(acuracia))
    print('MAE                 : {:0.2f}'.format(mae))
    print('MSE                 : {:0.2f}'.format(mse))
    print('RMSE                : {:0.2f}\n'.format(rmse))
    
    residuo = abs(y_teste - y_pred)
    plt.scatter(residuo, y_pred)
    plt.xlabel('Residuos')
    plt.ylabel('Previsto')
    plt.show()
    
    if report_treino:
        print('Dados de treino')
        if x_treino.shape[1] > 0 and len(y_treino) > 0: 
            reportModeloRegressao(modelo, x_treino, y_treino)
        else:
            print('X_treino e/ou y_treino possuem tamanho 0.')


# In[161]:


def treinaRegressao_GridSearchCV(modelo, params_, x_treino, y_treino, x_teste, y_teste,                        n_jobs = -1, cv = 5, refit = True, scoring = None, salvar_resultados = False,                       report_treino = False, retorna_modelo = False):
    grid = GridSearchCV(modelo, params_, n_jobs = n_jobs, cv = cv, refit = refit, scoring = scoring)
    
    grid.fit(x_treino, y_treino)
    pred = grid.predict(x_teste)
    modelo_ = grid.best_estimator_

    print(grid.best_params_)
    
    reportModeloRegressao(modelo_, x_teste, y_teste, x_treino, y_treino, report_treino) 
    
    if salvar_resultados:
        resultados_df = pd.DataFrame(grid.cv_results_)
        
        if retorna_modelo:
            return resultados_df, modelo_
        else:
            resultados_df
        
    if retorna_modelo:
        return modelo_


# ## 7.2 SVR

# Primeiramente iremos criar um modelo base utilizando o algoritmo SVM para regressão, conhecido como SVR. Assim, iremos poder ter uma metrica minima para comparar os nossos modelos, posteriormente iremos passar por uma fase de tuning dos hiperparametros, utilizaando GridSearchCV e depois um tuning manual.

# In[162]:


# Modelo base do algoritmo SVM para regressão
modelo_svr = SVR(max_iter = -1)
modelo_svr.fit(x_train, y_train)


# Apesar da nossa acuracia base ser 67%, estamos com um R^2 muito baixo de apenas 20.75%. Iremos tentar diminuor o nosso RMSE na medida que aumentamos o R^2.

# In[163]:


reportModeloRegressao(modelo_svr, x_test, y_test, x_train, y_train, True)


# In[162]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],\n    'C': [0.9, 1.0, 1.1],\n    'gamma': ['scale', 'auto']\n}\n\n# Criação de modelo intenso 01\nmodelo = SVR(max_iter = -1, cache_size = 1000)\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[163]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'kernel': ['rbf'],\n    'C': [0.001, 0.1, 1.0, 10, 100],\n    'gamma': ['auto']\n}\n\n# Criação de modelo intenso 02\nmodelo = SVR(max_iter = -1, cache_size = 1000)\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[164]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'kernel': ['rbf'],\n    'C': [0.1, 1.0, 10, 100, 1000, 10000],\n    'gamma': ['auto']\n}\n\n# Criação de modelo intenso 03\nmodelo = SVR(max_iter = -1, cache_size = 1000)\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[165]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'kernel': ['rbf'],\n    'C': [500, 1000, 2000],\n    'gamma': ['auto']\n}\n\n# Criação de modelo intenso 04\nmodelo = SVR(max_iter = -1, cache_size = 1000)\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[117]:


get_ipython().run_cell_magic('time', '', "# Modelo 05\nmodelo_svr_v5 = SVR(max_iter = -1, cache_size = 1000, kernel = 'rbf', C = 10000, gamma = 'auto')\nmodelo_svr_v5.fit(x_train, y_train)\n\nreportModeloRegressao(modelo_svr_v5, x_test, y_test, x_train, y_train, True)")


# In[124]:


get_ipython().run_cell_magic('time', '', "# Modelo 06\nmodelo_svr_v6 = SVR(max_iter = -1, cache_size = 1000, kernel = 'rbf', C = 10000, gamma = 1) # gamma = 'auto' = 0.166\nmodelo_svr_v6.fit(x_train, y_train)\n\nreportModeloRegressao(modelo_svr_v6, x_test, y_test, x_train, y_train, True)")


# In[126]:


get_ipython().run_cell_magic('time', '', "# Modelo 07\nmodelo_svr_v7 = SVR(max_iter = -1, cache_size = 1000, kernel = 'rbf', C = 10000, gamma = 3) # gamma = 'auto' = 0.166\nmodelo_svr_v7.fit(x_train, y_train)\n\nreportModeloRegressao(modelo_svr_v7, x_test, y_test, x_train, y_train, True)")


# In[127]:


get_ipython().run_cell_magic('time', '', "# Modelo 08\nmodelo_svr_v8 = SVR(max_iter = -1, cache_size = 1000, kernel = 'rbf', C = 10000, gamma = 0.5) # gamma = 'auto' = 0.166\nmodelo_svr_v8.fit(x_train, y_train)\n\nreportModeloRegressao(modelo_svr_v8, x_test, y_test, x_train, y_train, True)")


# ### 7.2.1 Conclusão SVR

# Executando o algoritmo SVR, conseguimos atingir as seguintes metricas sem evitar overfitting:
# 
# - R^2                 : 56.06%
# - R^2 Ajustado        : 56.12%
# - Acuracia            : 76.01%
# - MAE                 : 17.26
# - MSE                 : 797.83
# - RMSE                : 28.25
# 
# Apesar de atingirmos um RMSE relativamente baixo de 28 unidades, não possuimos uma boa acuracia, estando apenas em 75%. Ainda possuimos um R^2 baixo, de apenas 56%.
# 
# O algoritmo SVR necessita de uma alta carga de processamento, chegando a possuir testes em que os resultados demoravam mais de 1 hora para serem gerados. Para melhor compreensão foram exibidos nesse documento somente os algoritmos de maior influência.
# 
# O algoritmo SVR, não apresentou bom desempenho, devido a alta variabilidade nos dados, que não conseguiram ser identificados da forma ideal. Assim, iremos adotar a eestratégia de utilizar algoritmos ensemble, como XGBoost e CatBoost da categoria boosting.

# ### 7.2.2 Executando Melhor Modelo

# In[164]:


get_ipython().run_cell_magic('time', '', "# Modelo Final\nmodelo_svr_final = SVR(max_iter = -1, cache_size = 1000, kernel = 'rbf', C = 10000, gamma = 0.5)\nmodelo_svr_final.fit(x_train, y_train)\n\nreportModeloRegressao(modelo_svr_final, x_test, y_test, x_train, y_train, True)")


# ### 7.2.3 Avaliando SVR

# In[165]:


shap.initjs()


# In[166]:


# Construindo shap
amostras = 20
explainer = shap.Explainer(modelo_svr_final.predict, x_train)
shap_values = explainer(x_test[:amostras])


# In[167]:


# Waterfall Predição 0
shap.plots.waterfall(shap_values[0])


# In[168]:


# Waterfall Predição 10
shap.plots.waterfall(shap_values[10])


# In[169]:


# Force Predição 0
shap.plots.force(shap_values[0])


# In[170]:


# Force Predição 10
shap.plots.force(shap_values[10])


# In[171]:


# Summary Plot
shap.summary_plot(shap_values, x_test[:amostras])


# ## 7.3 CatBoost Regressor

# ### 7.3.1 Definindo Ambiente

# In[172]:


# Separando o conjunto de treino em treino e validação
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = seed_)


# In[173]:


# Definindo variaveis categoricas
categorical_features_index = np.where(x_train.dtypes != np.float)[0]


# ### 7.3.2 Iniciando Modelagem

# In[174]:


get_ipython().run_cell_magic('time', '', "# Modelo Base CatBoost Regressor\nmodelo_cat = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_)\n\nmodelo_cat.fit(x_train, y_train,\n               cat_features = categorical_features_index,\n               eval_set = (x_val, y_val),\n               plot = True, verbose = False);")


# Comparando com o melhor modelo gerado no tuning do algoritmo SVR, o modelo base do Catboost já possui um desempenho superior com maior R^2 Ajustado e menor RMSE e MSE.

# In[175]:


reportModeloRegressao(modelo_cat, x_test, y_test, x_train, y_train, True)


# In[120]:


get_ipython().run_cell_magic('time', '', "# Modelo 01 CatBoost Regressor\nmodelo_cat_v1 = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                                 iterations = 5000, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\nmodelo_cat_v1.fit(x_train, y_train,\n               cat_features = categorical_features_index,\n               eval_set = (x_val, y_val),\n               plot = True, verbose = True);")


# In[121]:


reportModeloRegressao(modelo_cat_v1, x_test, y_test, x_train, y_train, True)


# In[122]:


get_ipython().run_cell_magic('time', '', "# Modelo 03 CatBoost Regressor\nmodelo_cat_v3 = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                                 iterations = 5000, metric_period = 50, od_type = 'Iter', od_wait = 20,\\\n                                 learning_rate = 0.01)\n\nmodelo_cat_v3.fit(x_train, y_train,\n               cat_features = categorical_features_index,\n               eval_set = (x_val, y_val),\n               plot = True, verbose = False);")


# In[123]:


reportModeloRegressao(modelo_cat_v3, x_test, y_test, x_train, y_train, True)


# In[125]:


get_ipython().run_cell_magic('time', '', "# Modelo 04 CatBoost Regressor\nmodelo_cat_v4 = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                                 iterations = 5000, metric_period = 50, od_type = 'Iter', od_wait = 20,\\\n                                 learning_rate = 0.1)\n\nmodelo_cat_v4.fit(x_train, y_train,\n               cat_features = categorical_features_index,\n               eval_set = (x_val, y_val),\n               plot = True, verbose = False);")


# In[126]:


reportModeloRegressao(modelo_cat_v4, x_test, y_test, x_train, y_train, True)


# In[131]:


get_ipython().run_cell_magic('time', '', "# Modelo 05 CatBoost Regressor\nmodelo_cat_v5 = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                                 iterations = 20000, metric_period = 50, od_type = 'Iter', od_wait = 20,\\\n                                 learning_rate = 0.01)\n\nmodelo_cat_v5.fit(x_train, y_train,\n               cat_features = categorical_features_index,\n               eval_set = (x_val, y_val),\n               plot = True, verbose = False);\n\nreportModeloRegressao(modelo_cat_v5, x_test, y_test, x_train, y_train, True)")


# In[138]:


# Separando em variaveis preditivas e target 
#X = dtFinal[variaveis]
X = dtFinal[['T3', 'RH_3', 'T8', 'Press_mm_hg', 'NSM', 'Hour']]
y = dtFinal['Appliances'].values

# Separando em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed_)


# In[141]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [5, 6, 7, 8, 9],\n    'learning_rate': [0.01, 0.05, 0.1, 0.2],\n    'iterations' : [5000]\n}\n\n# Criação de modelo intenso 06\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[142]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [7, 8, 9, 10],\n    'learning_rate': [0.04, 0.05, 0.06, 0.07],\n    'iterations' : [5000]\n}\n\n# Criação de modelo intenso 07\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[143]:


# Separando em variaveis preditivas e target 
#X = dtFinal[variaveis]
X = dtFinal[['T3', 'RH_3', 'T8', 'Press_mm_hg', 'NSM', 'Hour']]
y = dtFinal['Appliances'].values

# Separando em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed_)

# Separando o conjunto de treino em treino e validação
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = seed_)


# In[144]:


get_ipython().run_cell_magic('time', '', "# Modelo 08 CatBoost Regressor\nmodelo_cat_v8 = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                                 iterations = 5000, metric_period = 50, od_type = 'Iter', od_wait = 20,\\\n                                 learning_rate = 0.05, depth = 10)\n\nmodelo_cat_v8.fit(x_train, y_train,\n               cat_features = categorical_features_index,\n               eval_set = (x_val, y_val),\n               plot = True, verbose = False);\n\nreportModeloRegressao(modelo_cat_v8, x_test, y_test, x_train, y_train, True)")


# In[145]:


# Separando em variaveis preditivas e target 
#X = dtFinal[variaveis]
X = dtFinal[['T3', 'RH_3', 'T8', 'Press_mm_hg', 'NSM', 'Hour']]
y = dtFinal['Appliances'].values

# Separando em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed_)


# In[147]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [8, 9, 10],\n    'learning_rate': [0.04, 0.05, 0.06],\n    'grow_policy': ['Depthwise', 'Lossguide'],\n    'iterations' : [5000]\n}\n\n# Criação de modelo intenso 09\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[148]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [9, 10],\n    'learning_rate': [0.02, 0.03, 0.04],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000]\n}\n\n# Criação de modelo intenso 10\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[149]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [10, 11],\n    'learning_rate': [0.025, 0.03, 0.035],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000]\n}\n\n# Criação de modelo intenso 11\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[151]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'learning_rate': [0.024, 0.025, 0.026],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000]\n}\n\n# Criação de modelo intenso 12\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[153]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [9000, 10000, 11000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000]\n}\n\n# Criação de modelo intenso 13\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[154]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.2, 0.22, 0.025, 0.27],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000]\n}\n\n# Criação de modelo intenso 14\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[155]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    # Não foi adicionado a score function 'Cosine', pois essa é a default utilizada no modelo 14\n    'score_function': ['L2', 'NewtonCosine', 'NewtonL2']\n}\n\n# Criação de modelo intenso 15\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[118]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.5]\n}\n\n# Criação de modelo intenso 16\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[117]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.4, 2.6, 2.8]\n}\n\n# Criação de modelo intenso 17\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[119]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.5],\n    'subsample': [0.7, 0.9, 1.0] # Default subsample = 0.8\n}\n\n# Criação de modelo intenso 18\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[117]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.5],\n    'subsample': [0.8],\n    'bootstrap_type': ['Bayesian', 'Bernoulli', 'No'] # Default para CPU = MVS\n}\n\n# Criação de modelo intenso 19\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[118]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [3.0],\n    'subsample': [0.8],\n    'bootstrap_type': ['Bernoulli']\n}\n\n# Criação de modelo intenso 20\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[120]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.5],\n    'subsample': [0.8],\n    'bootstrap_type': ['Bernoulli'],\n    'random_strength': [0.8, 1.2, 1.4, 1.6] # Default = 1\n}\n\n# Criação de modelo intenso 21\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[121]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.5],\n    'subsample': [0.8],\n    'bootstrap_type': ['Bernoulli'],\n    'random_strength': [0.9, 1.0, 1.1] # Default = 1\n}\n\n# Criação de modelo intenso 22\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[122]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.5],\n    'subsample': [0.8],\n    'bootstrap_type': ['Bernoulli'],\n    'random_strength': [1.0],\n    'min_data_in_leaf': [3, 6, 9] # Default = 1\n}\n\n# Criação de modelo intenso 22\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[117]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.5],\n    'subsample': [0.8],\n    'bootstrap_type': ['Bernoulli'],\n    'random_strength': [1.0],\n    'min_data_in_leaf': [1, 2] # Default = 1\n}\n\n# Criação de modelo intenso 23\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[130]:


get_ipython().run_cell_magic('time', '', "\nfeature_weights = [[1, 1, 1, 0.9, 1.1, 1.1], [0.9, 0.95, 1.05, 1, 1.2, 1.1]]\n\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.5],\n    'subsample': [0.8],\n    'bootstrap_type': ['Bernoulli'],\n    'random_strength': [1.0],\n    'min_data_in_leaf': [1],\n    'feature_weights': feature_weights\n}\n\n# Criação de modelo intenso 24\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 50, od_type = 'Iter', od_wait = 20)\n\ntreinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test, scoring = 'neg_root_mean_squared_error',\\\n                            report_treino = True)")


# In[138]:


# Separando em variaveis preditivas e target 
#X = dtFinal[variaveis]
X = dtFinal[['T3', 'RH_3', 'T8', 'Press_mm_hg', 'NSM', 'Hour']]
y = dtFinal['Appliances'].values

# Separando em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed_)

# Definindo variaveis categoricas
categorical_features_index = np.where(x_train.dtypes != np.float)[0]


# In[144]:


get_ipython().run_cell_magic('time', '', "cv_dataset = Pool(data=x_train,\n                  label=y_train,\n                  cat_features=categorical_features_index)\n\nparams = {\n    'depth': 11,\n    'langevin': True,\n    'diffusion_temperature': 10000,\n    'learning_rate': 0.025,\n    'grow_policy': 'Depthwise',\n    'iterations' : 5000,\n    'score_function': 'Cosine',\n    'l2_leaf_reg': 2.5,\n    'subsample': 0.8,\n    'bootstrap_type': 'Bernoulli',\n    'random_strength': 1.0,\n    'min_data_in_leaf': 1,\n    'loss_function': 'RMSE',\n    'loss_function': 'RMSE',\n    'eval_metric': 'RMSE',\n    'random_seed': seed_,\n    'verbose': False,\n    'metric_period': 10,\n    'od_type': 'Iter',\n    'od_wait': 10\n}\n\nscores = cv(cv_dataset,\n            params,\n            fold_count = 5, \n            plot = True,\n            seed = seed_)")


# In[137]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': 11,\n    'langevin': True,\n    'diffusion_temperature': 10000,\n    'learning_rate': 0.025,\n    'grow_policy': 'Depthwise',\n    'iterations' : 5000,\n    'score_function': 'Cosine',\n    'l2_leaf_reg': 2.5,\n    'subsample': 0.8,\n    'bootstrap_type': 'Bernoulli',\n    'random_strength': 1.0,\n    'min_data_in_leaf': 1,\n    'loss_function': 'RMSE',\n    'loss_function': 'RMSE',\n    'eval_metric': 'RMSE',\n    'random_seed': seed_,\n    'verbose': False,\n    'metric_period': 1,\n    'od_type': 'Iter',\n    'od_wait': 10\n}\n\n# Criação de modelo 25\nmodelo_cat_v25 = CatBoostRegressor(**params)\n\nmodelo_cat_v25.fit(x_train, y_train,\n               cat_features = categorical_features_index,\n               eval_set = (x_val, y_val),\n               plot = True, verbose = False)\n\nreportModeloRegressao(modelo_cat_v25, x_test, y_test, x_train, y_train, True)")


# ### 7.3.3 Conclusão CatBoostRegressor

# Após uma série de intensos treinamento com o algoritmo CatBoostRegressor, foi atingido as seguintes metricas:
# 
# - R^2                 : 73.05%
# - R^2 Ajustado        : 73.12%
# - Acuracia            : 79.44%
# - MAE                 : 14.35
# - MSE                 : 489.44
# - RMSE                : 22.12
# 
# Apesar de um intenso treinamento a acuracia que tinhamos como meta não foi alcançada por 0.56%. Apesar do objetivo não ser atingido, obtivemos metricas excelentes comparadas a outros algoritmos que atuaram no mesmo problema. O modelo final ainda obteve overfitting, apesar disso conseguiu seguir uma boa metrica de teste.  

# ### 7.3.4 Execução Melhor Modelo

# In[176]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n    'depth': [11],\n    'langevin': [True],\n    'diffusion_temperature': [10000],\n    'learning_rate': [0.025],\n    'grow_policy': ['Depthwise'],\n    'iterations' : [5000],\n    'score_function': ['Cosine'],\n    'l2_leaf_reg': [2.5],\n    'subsample': [0.8],\n    'bootstrap_type': ['Bernoulli'],\n    'random_strength': [1.0],\n    'min_data_in_leaf': [1]\n}\n\n# Criação de modelo Final\nmodelo = CatBoostRegressor(loss_function = 'RMSE', eval_metric = 'RMSE', random_seed = seed_,\\\n                           verbose = False, metric_period = 1, od_type = 'Iter', od_wait = 10)\n\nmodelo_cat_final = treinaRegressao_GridSearchCV(modelo, params, x_train, y_train, x_test, y_test,\\\n                                            scoring = 'neg_root_mean_squared_error',\\\n                                            report_treino = True, retorna_modelo = True)")


# ### 7.3.5 Salvando Modelo

# In[177]:


# Salvando modelo de machine learning em formato Pickle
pickle_out = open('../modelos/modelo_final.pkl', mode = 'wb')
dump(modelo_cat_final, pickle_out)
pickle_out.close()


# In[178]:


# Salvando Scale
dump(scaler, open('../modelos/scaler.pkl', mode = 'wb'))


# In[179]:


# Carregando modelo
with open('../modelos/modelo_final.pkl', 'rb') as f:
    modelo_cat_final = load(f)


# ### 7.3.6 Avaliando CatBoostRegressor

# In[180]:


def dependence_plot_unique(columns, shap_values_, x):
    for col in columns:
        shap.dependence_plot(col, shap_values_, x)


# In[181]:


# Construindo shap
amostras = 1000
x_shap = x_train[:amostras]
y_shap =  y_train[:amostras]

explainer = shap.TreeExplainer(modelo_cat_final)
shap_values = explainer.shap_values(Pool(x_shap, y_shap))


# Abaixo temos o impacto da predição 0 e 13, assim conseguimos ver como o resultado é afetado por cada variavel. 

# In[182]:


shap.initjs()
n = 0
shap.force_plot(explainer.expected_value, shap_values[n,:], x_shap.iloc[n,:])


# In[183]:


shap.initjs()
n = 13
shap.force_plot(explainer.expected_value, shap_values[n,:], x_shap.iloc[n,:])


# Abaixo possuimos a distribiução dos dados e seus respectivos impactos ao longo de n observações, assim conseguimos entender de forma simples o impacto de cada variavel e ainda realizar filtros.

# In[184]:


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, x_shap)


# É perceptivel que variaveis como 'NSM' e 'Hour' possuem um comportamento similar ao consumo de energia por hora, isso se da pois ambas variaveis estão relacionadas a tempo, onde impacta diretamente no consumo de energia.

# In[185]:


dependence_plot_unique(x_train.columns, shap_values, x_shap)


# É perceptivel que a maioria das variaveis possuem um impacto significativo e uma direção homogenea em relação ao impacto no valor resultante da função f(x) do nosso modelo. Essas tendencia se tornna maior em variaveis como 'NSM', 'Hour', 'T3' e 'T8'. Já em 'Press_mm_hg' se mostra menor em relação as outras.

# In[186]:


shap.summary_plot(shap_values,x_shap)


# In[187]:


modelo_cat_final.plot_tree(
    tree_idx=0
)


# # 8. Conclusão

# Por ultimo na etapa de conclusão, o CatBoostRegressor conseguiu ter um desempenho formidavel em relação a outros algoritmos como SVM e XGBoost.
# 
# Observação: Alguns testes foram realizados em outros notebooks para menor peso do notebook final.
# 
# O algoritmo CatBoostRegressor apresentou overfitting, que foi possivel controlar em determinadas etapas, porém com o alto custo computacional gerado a cada modelo, acabou ficando inviavel um maior controle. Ainda assim, conseguimos realizar um aumento nas metricas para o modelo, apresentando metricas relativamente altas.
# 
# -- Objetivos
# 
# 1. R^2 superior a 70% ✔
# 2. RMSE inferior a 25 ✔
# 3. MAE inferior a 15 ✔
# 4. Acuracia superior a 80%
# 5. Relatar economia total de energia ✔
# 
# Observação: O objetivo numero 5 será coberto no relatório final.

# In[188]:


# Calculando Previsões
pred = modelo_cat_final.predict(x_test)


# Analisando abaixo na linha tracejada em vermelho temos o consumo real de energia, já em azul possuimos o consumo estimado pelo modelo. Analisando previamente, percebemos que o modelo consegue diversas vezes capturar com precisão o consumo de energia naquele momento, já para momentos que não captura com exatidão, consegue prever a tendencia correta.
# 
# Dessa forma é interessante perceber que o algoritmo possui comportamentos em sua maioria corretos. Ainda iremos observar o valor previsto com o limite inferior e superior.

# In[189]:


fig = plt.figure(figsize = (20, 8))

amostras = 50
plt.plot(y_test[:amostras], label = 'Target', color = 'red', linestyle = '--')
plt.plot(pred[:amostras], label = 'CatBoost', color = 'blue')
plt.title('Consumo de Energia')
plt.xlabel('Observações')
plt.ylabel('Wh')

plt.legend()
plt.savefig('../analises/linha_real_previsto.png')
plt.show()


# In[190]:


# Calculando com intervalo de 95% de confiança
soma_erro = np.sum((y_test - pred)**2)
stdev = np.sqrt( 1 / (len(y_test) - 2) * soma_erro)

intervalo = 1.95 * stdev
lower, upper = pred - intervalo, pred + intervalo


# Observando abaixo ainda possuimos o limite inferior em verde e o superior em amarelo. O limite foi calculado uitilizando um intervalo de 95% de confiança. Dessa forma é possível visualizar que as predições em nenhum momento estiveram fora dos limites de confiança. Assim, podemos estimar que o nosso modelo possui valores com confiança acima de 95%.

# In[191]:


fig = plt.figure(figsize = (20, 8))

amostras = 50
plt.plot(y_test[:amostras], label = 'Target', color = 'red', linestyle = '--')
plt.plot(lower[:amostras],label='Limite Inferior', linestyle='--', color='g')
plt.plot(upper[:amostras],label='Limite Superior', linestyle='--', color='y')
plt.plot(pred[:amostras], label = 'CatBoost', color = 'blue')
plt.title('Previsão de Energia com Limite Inferior e Superior')
plt.xlabel('Observações')
plt.ylabel('Wh')

plt.legend()
plt.savefig('../analises/linha_real_previsto_limites.png')
plt.show()


# In[192]:


# Somando consumo de energia real e previsto
soma_energia_real_wh = sum(y_test)
soma_energia_pred_wh = sum(pred)

# Convertendo de Wh para kWh
soma_energia_real_kwh = soma_energia_real_wh / 1000
soma_energia_pred_kwh = soma_energia_pred_wh / 1000

soma_energia_pred = [soma_energia_real_kwh, soma_energia_pred_kwh]


# In[193]:


# Preco kWh Belgium - Local dos dados coletados
# Fonte dados atualizados em 01.12.2020: https://www.globalpetrolprices.com/Belgium/
kwh_casa_eur = 0.265
kwh_casa_usd = 0.315


# In[194]:


low = min(soma_energia_pred)
high = max(soma_energia_pred)


# In[195]:


def adicionaLabels(x, y):
    for i in range(len(x)):
        plt.text(i, round(y[i], 4), round(y[i], 4), ha = 'center')


# Abaixo conseguimos perceber que o consumo de energia total previsto foi 0.8 kWh abaixo do consumo real, dessa forma se aproximando muito do valor real, sendo perceptível que o modelo conseguiu prever o consumo de energia considerando os limites inferiores e superiores. Dessa forma podendo gerar economia de energia e maior eficiência na rede elétrica.

# In[196]:


# Grafico de barras do consumo previsto x real de energia
fig = plt.figure(figsize = (13, 8))

plt.bar(['Consumo Real', 'Consumo Previsto'], soma_energia_pred)
plt.ylabel('kWh')
plt.title('Consumo de Energia')
plt.ylim([ceil(low-0.5*(high-low)) - 1, ceil(high+0.5*(high-low)) + 1])
adicionaLabels(['Consumo Real', 'Consumo Previsto'], soma_energia_pred)

plt.savefig('../analises/barra_consumo_real_previsto.png')
plt.show()


# In[197]:


# Calculando custo eletrico em EUR
custo_eletrico_real_eur = soma_energia_real_kwh * kwh_casa_eur
custo_eletrico_pred_eur = soma_energia_pred_kwh * kwh_casa_eur

custo_eletrico_eur = [custo_eletrico_real_eur, custo_eletrico_pred_eur]

# Calculando limites do eixo y
low = min(custo_eletrico_eur)
high = max(custo_eletrico_eur)


# Abaixo é perceptivel que o custo eletrico em Euro possui uma diferença de apenas € 0.20, assim chegando a minimizar o custo em relação ao esperado.

# In[198]:


# Grafico de barras do consumo previsto x real de energia
fig = plt.figure(figsize = (13, 8))

plt.bar(['Custo Real', 'Custo Previsto'], custo_eletrico_eur)
plt.ylabel('EUR')
plt.title('Custo Eletrico')
plt.ylim([ceil(low-0.5*(high-low)) - 1, ceil(high+0.5*(high-low)) + 1])
adicionaLabels(['Custo Real', 'Custo Previsto'], custo_eletrico_eur)

plt.savefig('../analises/barra_custo_real_previsto_euro.png')
plt.show()


# In[199]:


# Calculando custo eletrico em USD
custo_eletrico_real_usd = soma_energia_real_kwh * kwh_casa_usd
custo_eletrico_pred_usd = soma_energia_pred_kwh * kwh_casa_usd

custo_eletrico_usd = [custo_eletrico_real_usd, custo_eletrico_pred_usd]

# Calculando limites do eixo y
low = min(custo_eletrico_usd)
high = max(custo_eletrico_usd)


# Abaixo é perceptivel que o custo eletrico em Dolar possui uma diferença de apenas $ 0.85, assim chegando a minimizar o custo em relação ao esperado.

# In[200]:


# Grafico de barras do consumo previsto x real de energia
fig = plt.figure(figsize = (13, 8))

plt.bar(['Custo Real', 'Custo Previsto'], custo_eletrico_usd)
plt.ylabel('USD')
plt.title('Custo Eletrico')
plt.ylim([ceil(low-0.5*(high-low)) - 1, ceil(high+0.5*(high-low)) + 1])
adicionaLabels(['Custo Real', 'Custo Previsto'], custo_eletrico_usd)

plt.savefig('../analises/barra_custo_real_previsto_dolar.png')
plt.show()


# In[ ]:





# In[ ]:




