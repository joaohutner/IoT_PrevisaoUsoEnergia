# Script de criação do dashboard
# https://dash.plotly.com/dash-html-components

# Imports
import traceback
import pandas as pd
import plotly.express as px
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import requests

# Módulos customizados
from modulos import constant, app_element

# Gera o layout
def get_layout():
    try:
        '''
        # Conectando ao banco de dados sqlite
        banco = sqlite3.connect('../database/banco.db', check_same_thread = False)
        cursor = banco.cursor()
        
        # Extraindo dados do banco de dados
        cursor.execute("SELECT * FROM previsao_energia")
        
        temp_list = cursor.fetchall() 
        return_list = []
        
        # Convertendo para lista de dicionarios
        for temp in temp_list:
            return_list.append({
                "Data": temp[0],
                "Hour": temp[1],
                "Press_mm_hg": temp[2],
                "Temperatura_Interna": temp[3],
                "Umidade_Interna": temp[4],
                "Previsao_Energia": temp[5]})   
        
        # Convertendo para dicionario e corrigindo os tipos
        dt = pd.DataFrame(return_list)
        dt['Data'] = pd.to_datetime(dt['Data'], format = '%d/%m/%Y')
        '''
        
        # Capturando dados da API
        request = requests.get('http://{IP_API}:{PORTA_API}/previsao'.format(IP_API = constant.IP_API, PORTA_API = constant.PORTA_API))
        temp_list = request.json()
        
        # Convertendo para dicionario e corrigindo os tipos
        dt = pd.DataFrame(temp_list)
        dt['Hour'] = dt['Hour'].astype(int) 
        dt[['Press_mm_hg', 'Temperatura_Interna', 'Umidade_Interna', 'Previsao_Energia']] = dt[['Press_mm_hg', 'Temperatura_Interna', 'Umidade_Interna', 'Previsao_Energia']].astype(float)
        dt['Data'] = dt['Data'] + ' ' + dt['Hour'].astype(str).str.zfill(2) + ':00:00'     
        dt['Data'] = pd.to_datetime(dt['Data'], format='%d/%m/%Y %H:%M:%S')
        dt = round(dt, 2)
        

        # Agrupa os dados
        dtPrevisoes = dt.groupby(['Data'])['Previsao_Energia'].sum().reset_index()
        
        dtPrevisoesPeriodo = dt.groupby(['Hour'])['Previsao_Energia'].sum().reset_index()
        dtPrevisoesPeriodo['Hour'] = dtPrevisoesPeriodo['Hour'].replace([0, 1, 2, 3, 4, 5], 'Madrugada')
        dtPrevisoesPeriodo['Hour'] = dtPrevisoesPeriodo['Hour'].replace([6, 7, 8, 9, 10, 11], 'Manhã')
        dtPrevisoesPeriodo['Hour'] = dtPrevisoesPeriodo['Hour'].replace([12, 13, 14, 15, 16, 17], 'Tarde')
        dtPrevisoesPeriodo['Hour'] = dtPrevisoesPeriodo['Hour'].replace([18, 19, 20, 21, 22, 23], 'Noite')
        dtPrevisoesPeriodo = dtPrevisoesPeriodo.groupby(['Hour'])['Previsao_Energia'].sum().reset_index()
        
        dtDescribe = dt.describe().reset_index()
        dtDescribe = round(dtDescribe, 2)
        dtDescribe = dtDescribe.rename({'index': 'Indice', 'Hour': 'Hora', 'Press_mm_hg': 'Pressão', 'Temperatura_Interna': 'Temperatura Interna', 'Umidade_Interna' : 'Umidade Interna', 'Previsao_Energia': 'Previsão Energia'}, axis = 1)

        # Gera o container
        layout = dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id = 'my-line', 
                                      figure = px.line(dtPrevisoes, 
                                                       x = "Data",
                                                       y = 'Previsao_Energia', 
                                                       title = 'Previsão do Consumo de Energia em Wh Total por Dia',
                                                       labels = {'Previsao_Energia': 'Previsão de Energia em Wh'}))
                        ],
                        width=12)
                    ],
                    style = {'padding-bottom': '10px'},
                    no_gutters = True),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id = 'my-pie2', 
                                      figure = px.pie(dtPrevisoesPeriodo, 
                                                      values = 'Previsao_Energia', 
                                                      names = ['Madrugada', 'Manhã', 'Tarde', 'Noite'], 
                                                      title = 'Consumo de Energia por Período do Dia em Wh',
                                                      hole = 0.3))
                        ],
                        width = 5),
                        dbc.Col(dbc.Card([
								dbc.CardHeader("Descrição Estatística dos dados utilizados para previsão"),
								app_element.generate_dashtable(identifier = "table2", dataframe = dtDescribe, height_cell = '40px', height = '375px')],
								className = "shadow p-3 bg-light rounded", style={'height':'45vh'}), width = 7)
                        ],
                    style = {'padding-bottom': '10px'},
                    no_gutters = True)
                ],
                fluid = True)
        return layout
    except:
        layout = dbc.Jumbotron(
                    [
                        html.Div([
                            html.H1("500: Internal Server Error", className = "text-danger"),
                            html.Hr(),
                            html.P(f"Following Exception Occured: "),
                            html.Code(traceback.format_exc())
                        ],
                        style = constant.NAVITEM_STYLE)
                    ]
                )
        return layout
