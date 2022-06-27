import sqlite3

# Instanciando Previsão de Energia

banco = sqlite3.connect('banco.db')

cursor = banco.cursor()

cursor.execute("CREATE TABLE previsao_energia (data text, Hour integer, Press_mm_hg real, Temperatura_Interna real, Umidade_Interna real, Previsao_Energia real)")
               
banco.commit()