# Importar bibliotecas
import sqlite3
import pandas as pd
from flask_restplus import Resource
from pickle import load
from src.server.instance import server

app, api = server.app, server.api

# Conectando ao banco de dados sqlite
banco = sqlite3.connect('../database/banco.db', check_same_thread = False)
cursor = banco.cursor()

# Carregando modelo
pickle_model = open('modelo_iot_energia.pkl', 'rb')
modelo = load(pickle_model)
pickle_model.close()

# Carregando Scale
pickle_scale = open('scaler.pkl', 'rb')
scaler = load(pickle_scale)
pickle_scale.close()

# Prevendo Appliances
def prediction(NSM, Hour, Press_mm_hg, T3, T8, RH_3):

    dt = 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3'  

    quantitativas = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4',\
       'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9',\
       'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility',\
       'Tdewpoint', 'NSM']
   

    dt = {'lights': [1], 'T1': [1], 'RH_1': [1], 'T2': [1], 'RH_2': [1], 'T3': [T3], 'RH_3': [RH_3], 'T4': [1], 'RH_4': [1], 'T5': [1],\
          'RH_5': [1], 'T6': [1], 'RH_6': [1], 'T7': [1], 'RH_7': [1], 'T8': [T8], 'RH_8': [1], 'T9': [1], 'RH_9': [1], 'T_out': [1],\
              'Press_mm_hg': [Press_mm_hg], 'RH_out': [1], 'Windspeed': [1], 'Visibility': [1], 'Tdewpoint': [1], 'NSM': [NSM],\
                  'Weekend': [1], 'Day_of_week': [1], 'Month': [1], 'Day': [1], 'Hour': [Hour]} 

    dt = pd.DataFrame.from_dict(dt)
    
    # Padronizando dados
    dt_padronizado = dt.copy()
    dt_padronizado[quantitativas] = scaler.transform(dt[quantitativas])

    # Convertendo para dataframe
    dt_padronizado = pd.DataFrame(dt_padronizado.copy(), columns = dt.columns) 

    # Prevendo Appliances
    pred = modelo.predict(dt_padronizado[['T3', 'RH_3', 'T8', 'Press_mm_hg', 'NSM', 'Hour']])

    return pred

@api.route('/previsao')
class Previsao(Resource):
    def get(self,):
        try:
            cursor.execute("SELECT * FROM previsao_energia LIMIT 100")
            
            temp_list = cursor.fetchall() 
            return_list = []
            
            for temp in temp_list:
                return_list.append({
                    "Data": temp[0],
                    "Hour": temp[1],
                    "Press_mm_hg": temp[2],
                    "Temperatura_Interna": temp[3],
                    "Umidade_Interna": temp[4],
                    "Previsao_Energia": temp[5]})
        except:
            return "Erro na captura dos dados do banco de dados.", 500
        
        return return_list, 200
    
    def post(self, ): 
        # Convertendo entrada dos dados para dicionario
        response = dict(api.payload)
        
        try:
            # Capturando os dados
            data = response["data"]
            Hour = int(response["Hour"])
            Press_mm_hg = float(response["Press_mm_hg"])
            T3 = float(response["T3"])
            RH_3 = float(response["RH_3"])  
            
            # Definindo valores pre-default
            NSM = (24 - Hour) * 60 * 60
            T8 = T3 + 0.25
        except:
            return "Formato dos dados invalido.", 400                    
        
        try:
            # Chamando função de previsão
            result = prediction(NSM, Hour, Press_mm_hg, T3, T8, RH_3)
        except:
            return "Erro na previsão dos dados", 400

        try:
            # Inserindo dados no banco de dados interno
            cursor.execute("INSERT INTO previsao_energia VALUES ('" + str(data) + "', '" + str(Hour) + "', '" + str(Press_mm_hg) + "', '" + str(T3) + "', '" + str(RH_3) + "', '" + str(round(result[0], 2)) + "')")
            banco.commit()
        except:
            return "Erro ao inserir dados no banco de dados", 500
        
        return str(result), 200
    
@api.route('/verifica')
class VerificaDados(Resource):
    def get(self):
        try:
            # Consulta banco de dados
            cursor.execute("SELECT COUNT('Data') FROM previsao_energia")
            
            count = cursor.fetchall()[0][0]
            
            return count, 200
        except:
            return "Erro na captura dos dados do banco de dados.", 500
        