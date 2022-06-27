# Página de overview

# Imports
import traceback
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import requests
import plotly.graph_objs as go
from datetime import date, datetime

# Módulos customizados
from modulos import constant

# Função para obter o layout
def get_layout():
    try:
        # Capturando dados da API
        request = requests.get('http://{IP_API}:{PORTA_API}/previsao'.format(IP_API = constant.IP_API, PORTA_API = constant.PORTA_API))
        
        temp_list = request.json()
        
        # Convertendo para dicionario e corrigindo os tipos
        dt = pd.DataFrame(temp_list).reset_index()
        dt['Hour'] = dt['Hour'].astype(int) 
        dt[['Press_mm_hg', 'Temperatura_Interna', 'Umidade_Interna', 'Previsao_Energia']] = dt[['Press_mm_hg', 'Temperatura_Interna', 'Umidade_Interna', 'Previsao_Energia']].astype(float)
        dt2 = dt.copy()
        dt3 = dt.copy()
        dt['Data'] = dt['Data'] + ' ' + dt['Hour'].astype(str).str.zfill(2) + ':00:00'     
        dt['Data'] = pd.to_datetime(dt['Data'], format='%d/%m/%Y %H:%M:%S')
        dt2['Data'] = pd.to_datetime(dt2['Data'], format='%d/%m/%Y')
        dt3['Data'] = pd.to_datetime(dt3['Data'], format='%d/%m/%Y')
        dt = round(dt, 2)     
        
        # Calculando Indicadores
        d = date.today().strftime('%Y-%m-%d') + ' 00:00:00'
        dt2 = dt2[dt2['Data'] == d]
        gastoHoje =str(round(dt2['Previsao_Energia'].sum(), 2)) + ' Wh'
        gasto30Dias = str(round(dt[dt['Data'] > datetime.now() - pd.to_timedelta("30day")]['Previsao_Energia'].sum(), 2)) + ' Wh'
        gasto7Dias = str(round(dt[dt['Data'] > datetime.now() - pd.to_timedelta("7day")]['Previsao_Energia'].sum(), 2)) + ' Wh'
        
        dtMesAtual = dt3[dt3['Data'].dt.month == datetime.now().month]
        dtMesAtual = dtMesAtual[dtMesAtual['Data'].dt.day != 31]
        
        subtractMonth = datetime.now().month       
        if subtractMonth == 1:
            subtractMonth = 12
            dtMesAnterior = dt3[dt3['Data'].dt.month == subtractMonth]
            dtMesAnterior = dtMesAnterior[dtMesAnterior['Data'].dt.year == datetime.now().year - 1]
            dtMesAnterior = dtMesAnterior[dtMesAnterior['Data'].dt.day != 31]
        else:
            subtractMonth -= 1
            dtMesAnterior = dt3[dt3['Data'].dt.month == subtractMonth]
        
        # Agrupando por data
        dtMesAtual = dtMesAtual.groupby('Data')['Previsao_Energia'].sum().reset_index()
        dtMesAnterior = dtMesAnterior.groupby('Data')['Previsao_Energia'].sum().reset_index()
        
        dtMesAtual['Data'] = dtMesAtual['Data'].dt.day.astype(str) + '/' + dtMesAtual['Data'].dt.year.astype(str)
        dtMesAnterior['Data'] = dtMesAnterior['Data'].dt.day.astype(str) + '/' + dtMesAnterior['Data'].dt.year.astype(str)
        
        strMesAnterior = datetime.strptime(str(subtractMonth), "%m").strftime("%B")
        strMesAtual = datetime.strptime(str(datetime.now().month), "%m").strftime("%B")
        
        # Plot Mes Atual x Mes Passado

        # Defini��o dos dados no plot
        plot_data = [go.Scatter(x = dtMesAtual['Data'],
                                y = dtMesAtual['Previsao_Energia'],
                                name = strMesAtual),
                     go.Scatter(x = dtMesAnterior['Data'],
                                y = dtMesAnterior['Previsao_Energia'],
                                name = strMesAnterior)]
        
        # Layout
        plot_layout = go.Layout(xaxis = {"type": "category", 'title': 'Periodo'},
                                yaxis = {'title': 'Previsao de Energia'}, 
                                title = 'Diferenca de Previsao de Energia entre o mes vigente e o anterior',
                                height = 550)
        
        # Plot da figura
        dif_fig = go.Figure(data = plot_data, layout = plot_layout)
        
        # Layout
        layout = dbc.Container([
                 dbc.Row([
                        dbc.Col([
                        dbc.Card([dbc.CardHeader("Gasto Energetico Ultimos 30 Dias"),
                                  dbc.CardBody([html.H2(gasto30Dias, className = "card-text")]),], className = "shadow p-3 bg-light rounded")], width = 3),
                        dbc.Col([
                        dbc.Card([dbc.CardHeader("Gasto Energetico Ultimos 7 Dias"),
                                  dbc.CardBody([html.H2(gasto7Dias, className = "card-text")]),], className = "shadow p-3 bg-light rounded")], width = 3),
                        dbc.Col([
                        dbc.Card([dbc.CardHeader("Gasto Energetico Hoje"),
                                  dbc.CardBody([html.H2(gastoHoje, className = "card-text")]),], className = "shadow p-3 bg-light rounded")], width = 3),
                        dbc.Col([
                        dbc.Card([dbc.CardHeader("Em Caso de Dúvidas Envie E-mail Para"),
                                  dbc.CardBody([html.H2("Suporte BIGF", className = "card-text")]),], className = "shadow p-3 bg-light rounded")], width = 3)],
                        className= "pb-3"),
                 dbc.Row([
                        dbc.Col(dcc.Graph(id = 'dif_graph', figure = dif_fig), width = 12)
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


