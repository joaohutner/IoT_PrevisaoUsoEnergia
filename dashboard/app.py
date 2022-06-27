# Módulo de criação do servidor Dash

# Import
import dash

# meta_tags são necessárias para que o layout do aplicativo seja responsivo em dispositivos móveis
app = dash.Dash(__name__, suppress_callback_exceptions = True, meta_tags = [{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

# Cria o servidor
server = app.server
