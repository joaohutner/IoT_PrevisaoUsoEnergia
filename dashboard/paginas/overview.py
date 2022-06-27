# Página de overview

# Imports
import traceback
import pandas as pd
import dash_html_components as html
import dash_bootstrap_components as dbc
import requests

# Módulos customizados
from modulos import app_element, constant

# Função para obter o layout
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
        dt = pd.DataFrame(return_list).reset_index()
        dt = round(dt, 2)
        '''
        
        # Capturando dados da API
        request = requests.get('http://{IP_API}:{PORTA_API}/previsao'.format(IP_API = constant.IP_API, PORTA_API = constant.PORTA_API))
        
        temp_list = request.json()
        
        # Convertendo para dicionario e corrigindo os tipos
        dt = pd.DataFrame(temp_list).reset_index()
        dt = round(dt, 2)     
        
        observacoes = dt.shape[0]     

        # Layout
        layout = dbc.Container([
                 dbc.Row([
                        dbc.Col([
                        dbc.Card([dbc.CardHeader("Objetivo do Dashboard"),
                                  dbc.CardBody([html.H2("Data App", className = "card-text")]),], className = "shadow p-3 bg-light rounded")], width = 3),
                        dbc.Col([
                        dbc.Card([dbc.CardHeader("Número de Registros Analisados"),
                                  dbc.CardBody([html.H2(str(observacoes), className = "card-text")]),], className = "shadow p-3 bg-light rounded")], width = 3),
                        dbc.Col([
                        dbc.Card([dbc.CardHeader("Período de Coleta dos Dados"),
                                  dbc.CardBody([html.H2("2021", className = "card-text")]),], className = "shadow p-3 bg-light rounded")], width = 3),
                        dbc.Col([
                        dbc.Card([dbc.CardHeader("Em Caso de Dúvidas Envie E-mail Para"),
                                  dbc.CardBody([html.H2("Suporte BIGF", className = "card-text")]),], className = "shadow p-3 bg-light rounded")], width = 3)],
                        className= "pb-3"),
                 dbc.Row([
                        dbc.Col(dbc.Card([
                                dbc.CardHeader("Registros Analisados"),
                                app_element.generate_dashtable(identifier = "table1", dataframe = dt, height = '800px')],
                                className = "shadow p-3 bg-light rounded"), width = 12)
                ])
        ],
        fluid = True)

        return layout
    except:
        layout = dbc.Jumbotron(
                    [
                        html.Div([
                            html.H1("500: Internal Server Error", className="text-danger"),
                            html.Hr(),
                            html.P("O seguinte erro ocorreu:"),
                            html.Code(traceback.format_exc())
                        ],
                        style=constant.NAVITEM_STYLE)
                    ]
                )
        return layout


