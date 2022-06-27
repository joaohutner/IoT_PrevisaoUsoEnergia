import sqlite3
import random
import pandas as pd
from pickle import load
from datetime import datetime, timedelta

from random import randrange, uniform

# Instanciando Previsão de Energia

banco = sqlite3.connect('banco.db')

cursor = banco.cursor()

cursor.execute("SELECT * FROM previsao_energia LIMIT 3")
banco.commit()   

a = cursor.fetchall()

min_year = 2021
max_year = datetime.now().year

start = datetime(min_year, 1, 1, 00, 00, 00)
years = max_year - min_year + 1
end = start + timedelta(days = 365 * years)

n = 1000

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

for i in range(0, n):
    random_date = start + (end - start) * random.random()
    data = str(random_date.strftime('%d/%m/%Y'))
    
    random_hour = randrange(0, 11)
    '''
    if random_hour < 1: 
        Hour = randrange(0, 6)
    elif random_hour >= 1 and random_hour <= 3: 
        Hour = randrange(6, 12)
    elif random_hour > 3 and random_hour <= 5: 
        Hour = randrange(12, 18)
    else:
        Hour = randrange(18, 24)
    '''
    Hour = randrange(0, 24) 
    Press_mm_hg = uniform(720, 781)
    T3 = uniform(17, 31)
    RH_3 = uniform(28, 52)
    
    # Definindo valores pre-default
    NSM = (24 - Hour) * 60 * 60
    T8 = T3 + 0.25
    
    result = prediction(NSM, Hour, Press_mm_hg, T3, T8, RH_3)

    cursor.execute("INSERT INTO previsao_energia VALUES ('" + str(data) + "', '" + str(Hour) + "', '" + str(Press_mm_hg) + "', '" + str(T3) + "', '" + str(RH_3) + "', '" + str(round(result[0], 2)) + "')")
    banco.commit()   


'''
cursor.execute("SELECT * FROM previsao_energia")
               
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

dt2 = pd.DataFrame(return_list)
dt2['Data'] = pd.to_datetime(dt2['Data'], format = '%d/%m/%Y')
'''
'''
min_year = 2021
max_year = datetime.now().year

start = datetime(min_year, 1, 1, 00, 00, 00)
years = max_year - min_year + 1
end = start + timedelta(days = 365 * years)

for i in range(10):
    random_date = start + (end - start) * random.random()
    print(random_date.strftime('%d/%m/%Y'))
'''