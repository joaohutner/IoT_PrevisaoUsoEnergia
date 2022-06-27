# Módulo para carga dos dados

# Imports
import json
import pandas as pd
from modulos import constant

# Função para gerar o dataframe
def generate_dataframe():
    
    # Carrega o arquivo
    DATAFRAME_MAIN = pd.read_csv(constant.DATAFILE)

    # Faz o mapeamento das colunas
    DATAFRAME_MAIN.rename(constant.FIELD_MAP, axis = 1, inplace = True)

    # Formata as colunas de data
    DATAFRAME_MAIN[constant.DATA_CRIACAO] = pd.to_datetime(DATAFRAME_MAIN[constant.DATA_CRIACAO], format = constant.DATE_FORMAT)
    DATAFRAME_MAIN[constant.DATA_FECHAMENTO] = pd.to_datetime(DATAFRAME_MAIN[constant.DATA_FECHAMENTO], format = constant.DATE_FORMAT)

    # Adiciona as colunas formatadas ao dataframe
    DATAFRAME_MAIN[constant.DATA_CRIACAO] =  DATAFRAME_MAIN[constant.DATA_CRIACAO].dt.date
    DATAFRAME_MAIN[constant.DATA_FECHAMENTO] =  DATAFRAME_MAIN[constant.DATA_FECHAMENTO].dt.date

    # Filtra pela coluna de status
    DATAFRAME_MAIN.loc[DATAFRAME_MAIN.eval('Status != @constant.CLOSED_ISSUE_STATUS'), constant.STATUS_TYPE] = "Aberto"
    DATAFRAME_MAIN.loc[DATAFRAME_MAIN.eval('Status == @constant.CLOSED_ISSUE_STATUS'), constant.STATUS_TYPE] = "Fechado"

    return(DATAFRAME_MAIN)

# Função para os tickets abertos
def get_open_issues(dataframe):
    return (dataframe.query('Status != @constant.CLOSED_ISSUE_STATUS'))

# Função para os tickets fechados
def get_closed_issues(dataframe):
    return (dataframe.query('Status == @constant.CLOSED_ISSUE_STATUS'))

# Função para ajustar as colunas
def read_config_in_df():

    # Carrega as constantes
    constant.read_config()  

    # Prepara os dados
    data = [
                ['id', constant.ID, constant.CSV_ID],
                ['status', constant.STATUS, constant.CSV_STATUS],
                ['criado_por', constant.CRIADO_POR, constant.CSV_CRIADO_POR],
                ['atribuido_a', constant.ATRIBUIDO_A, constant.CSV_ATRIBUIDO_A],
                ['atendido_por', constant.ATENDIDO_POR, constant.CSV_ATENDIDO_POR],
                ['severidade', constant.SEVERIDADE, constant.CSV_SEVERIDADE],
                ['prioridade', constant.PRIORIDADE, constant.CSV_PRIORIDADE],
                ['cliente', constant.CLIENTE, constant.CSV_CLIENTE],
                ['data_criacao', constant.DATA_CRIACAO, constant.CSV_DATA_CRIACAO],
                ['data_fechamento', constant.DATA_FECHAMENTO, constant.CSV_DATA_FECHAMENTO],
                ['tipo_chamado', constant.TIPO_CHAMADO, constant.CSV_TIPO_CHAMADO]
            ]
    
    # Cria o dataframe
    df = pd.DataFrame(data, columns = ['Id Coluna', 'Nome Coluna', 'Mapeado Para'])

    return (df)

# Grava o arquivo de mapeamento
def write_field_mapping_file(data, dt_format):
    JSON_FILE = {}
    JSON_FILE['KeyMapping'] = {}
    JSON_FILE['KeyMapping']['VarMapping'] = {}
    JSON_FILE['KeyMapping']['FieldMapping'] = {}

    for item in data:
        JSON_FILE['KeyMapping']['VarMapping'].update({item['Id Coluna'].rstrip(): item['Nome Coluna'].rstrip()})
        JSON_FILE['KeyMapping']['FieldMapping'].update({item['Mapeado Para'].rstrip(): item['Nome Coluna'].rstrip()})
    
    JSON_FILE["DateFormat"] = dt_format
    with open(constant.MAPPING_FILE, "w") as f:
        json.dump(JSON_FILE, f, indent = 3)
    
    return 0