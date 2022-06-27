# Módulo de valores constantes

# Imports
import json

# Constantes
APP_LOGO = "imagens/logo.png"
MAPPING_FILE = "config/mapeamento_campos_dataset.json"
DATAFILE = "datasets/dataset.csv"
IP_API = 'localhost'
PORTA_API = '5000'
N_OBSERVACOES = "-1"

# Formatação dos argumentos da barra lateral
SIDEBAR_STYLE = {"position": "fixed", 
                 "top": 0, 
                 "left": 0, 
                 "bottom": 0, 
                 "width": "15rem", 
                 "padding": "0rem 0rem",
                 "background-color": "dark"}

# Formatação dos itens de navegação
NAVITEM_STYLE = {"padding": "0rem 1rem"}

# Os estilos do conteúdo principal posicionam-no à direita da barra lateral e adicionamos um preenchimento.
CONTENT_STYLE = {"margin-left": "15rem", "padding": "0rem 0rem"}

# Inicializa os nomes das colunas do dataset
ID              = "ID"
STATUS          = "Status"
CRIADO_POR      = "Criado Por"
ATRIBUIDO_A     = "Atribuido A"
ATENDIDO_POR    = "Atendido Por"
SEVERIDADE      = "Severidade"
PRIORIDADE      = "Prioridade"
CLIENTE         = "Cliente"
DATA_CRIACAO    = "Data Criação"
DATA_FECHAMENTO = "Data Fechamento"
TIPO_CHAMADO    = "Tipo Chamado"

# Nomes no arquivo csv
CSV_ID              = "ID"
CSV_STATUS          = "Status"
CSV_CRIADO_POR      = "Criado Por"
CSV_ATRIBUIDO_A     = "Atribuido A"
CSV_ATENDIDO_POR    = "Atendido Por"
CSV_SEVERIDADE      = "Severidade"
CSV_PRIORIDADE      = "Prioridade"
CSV_CLIENTE         = "Cliente"
CSV_DATA_CRIACAO    = "Data Criação"
CSV_DATA_FECHAMENTO = "Data Fechamento"
CSV_TIPO_CHAMADO    = "Tipo Chamado"

# Dicionário de mapeamento dos campos
FIELD_MAP = {"ID": "ID",
             "Status": "Status",
             "Criado Por": "Criado Por",
             "Atribuido A": "Atribuido A",
             "Atendido Por": "Atendido Por",
             "Severidade": "Severidade",
             "Prioridade": "Prioridade",
             "Cliente": "Cliente",
             "Data Criação": "Data Criação",
             "Data Fechamento": "Data Fechamento",
             "Tipo Chamado": "Tipo Chamado"}

# Formato de data             
DATE_FORMAT = "%d-%m-%Y %H:%M"

# Variáveis customizadas
CREATED_TIME = 'CreatedTime'
CREATED_DT = 'CreatedDT'
STATUS_TYPE = 'StatusType'
CLOSED_ISSUE_STATUS = ["Fechado", "Resolvido", "Solução Proposta", "Pesquisa Realizada", "Solução Aplicada", "Solução Documentada"]

# Função para leitura das configurações
def read_config():

    # Variáveis
    global ID
    global STATUS
    global CRIADO_POR
    global ATRIBUIDO_A
    global ATENDIDO_POR
    global SEVERIDADE
    global PRIORIDADE
    global CLIENTE
    global DATA_CRIACAO
    global DATA_FECHAMENTO
    global TIPO_CHAMADO

    global CSV_ID
    global CSV_STATUS
    global CSV_CRIADO_POR
    global CSV_ATRIBUIDO_A
    global CSV_ATENDIDO_POR
    global CSV_SEVERIDADE
    global CSV_PRIORIDADE
    global CSV_CUSTOMER
    global CSV_DATA_CRIACAO
    global CSV_DATA_FECHAMENTO
    global CSV_TIPO_CHAMADO

    global FIELD_MAP
    global DATE_FORMAT

    # Carrega o arquivo json
    with open(MAPPING_FILE) as f:
        CONFIG_OBJECT = json.load(f)

    # Mapeamento de campos
    FIELD_MAP       = CONFIG_OBJECT["KeyMapping"]["FieldMapping"]
    ID              = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["id"]
    STATUS          = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["status"]
    CRIADO_POR      = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["criado_por"]
    ATRIBUIDO_A     = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["atribuido_a"]
    ATENDIDO_POR    = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["atendido_por"]
    SEVERIDADE      = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["severidade"]
    PRIORIDADE      = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["prioridade"]
    CLIENTE         = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["cliente"]
    DATA_CRIACAO    = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["data_criacao"]
    DATA_FECHAMENTO = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["data_fechamento"]
    TIPO_CHAMADO    = CONFIG_OBJECT["KeyMapping"]["VarMapping"]["tipo_chamado"]
    DATE_FORMAT     = CONFIG_OBJECT["DateFormat"]

    key_list = list(FIELD_MAP.keys())
    val_list = list(FIELD_MAP.values())   
    
    CSV_ID              = key_list[val_list.index(ID)]
    CSV_STATUS          = key_list[val_list.index(STATUS)]
    CSV_CRIADO_POR      = key_list[val_list.index(CRIADO_POR)]
    CSV_ATRIBUIDO_A     = key_list[val_list.index(ATRIBUIDO_A)]
    CSV_ATENDIDO_POR    = key_list[val_list.index(ATENDIDO_POR)]
    CSV_SEVERIDADE      = key_list[val_list.index(SEVERIDADE)]
    CSV_PRIORIDADE      = key_list[val_list.index(PRIORIDADE)]
    CSV_CLIENTE         = key_list[val_list.index(CLIENTE)]
    CSV_DATA_CRIACAO    = key_list[val_list.index(DATA_CRIACAO)]
    CSV_DATA_FECHAMENTO = key_list[val_list.index(DATA_FECHAMENTO)]
    CSV_TIPO_CHAMADO    = key_list[val_list.index(TIPO_CHAMADO)]
 