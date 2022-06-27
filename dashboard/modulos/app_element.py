# Módulo para gerar a tabela de dados no Dash

# Imports
import dash_table

# Função para gerar a tabela dash
def generate_dashtable(identifier, dataframe, height = '300px', width = 'auto', height_cell = 'auto', textAlign_cell = 'left'):
    return dash_table.DataTable(id = identifier,
                                columns = [{"name": i, "id": i} for i in dataframe.columns],
                                data = dataframe.to_dict('records'),
                                filter_action = "native",
                                style_header = {'fontWeight': 'bold', 'textAlign': 'center'},
                                style_cell = {'whiteSpace': 'normal', 'height': height_cell, 'textAlign': textAlign_cell},
                                fixed_rows = {'headers': True},
                                page_action = 'none',
                                style_table = {'height': height,'width': width})