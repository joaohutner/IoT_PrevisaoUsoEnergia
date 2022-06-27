# Importar bibliotecas
from flask import Flask
from flask_restplus import Api

class Server():
    def __init__(self, ):
        self.app = Flask(__name__)
        self.api = Api(self.app,
                       version = '1.0',
                       title = 'Energy Prediction API',
                       description = 'Energy Prediction API',
                       doc = '/docs'
                       )
        
    def run(self, ):
        self.app.run(
            debug = False,
            host = "localhost"
            )
        
server = Server()