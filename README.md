# Modelagem Preditiva em IoT - Previsão de Uso de Energia

Este projeto de IoT tem como objetivo a criação de modelos preditivos para 
a previsão de consumo de energia de eletrodomésticos. Os dados utilizados 
incluem medições de sensores de temperatura e umidade de uma rede sem fio, 
previsão do tempo de uma estação de um aeroporto e uso de energia utilizada por
luminárias. 

O conjunto de dados foi coletado por um período de 10 minutos por cerca de
5 meses. As condições de temperatura e umidade da casa foram monitoradas com 
uma rede de sensores sem fio ZigBee. Cada nó sem fio transmitia as condições
de temperatura e umidade em torno de 3 min. Em seguida, a média dos dados foi
calculada para períodos de 10 minutos. Os dados de energia foram registrados
a cada 10 minutos com medidores de energia de barramento. O tempo da estação
meteorológica mais próxima do aeroporto (Aeroporto de Chievres, Bélgica) foi
baixado de um conjunto de dados públicos do Reliable Prognosis (rp5.ru) e mesclado
com os conjuntos de dados experimentais usando a coluna de data e hora. Duas variáveis
aleatórias foram incluídas no conjunto de dados para testar os modelos de
regressão e filtrar os atributos não preditivos (parâmetros).

Instruções para executar:

1 - Baixar modelo em:
https://drive.google.com/file/d/1kg8nQO1-Zuo3OHiZf7FmJVCI6MPWLudp/view?usp=sharing

2 - Extrair .zip

3 - Renomear se necessário para "modelo_final.pkl"

4 - Copiar o modelo para a pasta /api com o seguinte nome "modelo_iot_energia.pkl"

5 - Copiar o modelo para a pasta /app com o seguinte nome "modelo_iot_energia.pkl"

6 -  Instalar todas a bibliotecas na raiz da pasta, arquivo de instalação "requirements.txt"

Orientações Gerais:

- Dentro da pasta docs esta localizado o notebook utilizado durante o projeto, juntamente com a sua versão convertido para .html .
- Na pasta modelos possui outro readme.md com instruções de como baixar o modelo já treinado, esse não foi incluso no repositorio devido a sua alta volumetria (mesmo compactado).
- O script .py na raiz do projeto é uma conversão direta do notebook utilizado, sendo assim é sugerido a utilização do notebook na pasta docs.

Ordem de execução:

- API em api/
    - Abrir terminal de comando na pasta api/
    - Executar o comando "python main.py"
- APP em app/
    - Abrir terminal de comando na pasta app/
    - Executar o comando "streamlit run app.py"
- Dashboard em dashboard/
    - Abrir terminal de comando na pasta dashboard/
    - Executar o comando "python dataapp.py"

Instruções em mais detalhes em cada uma das pastas.
