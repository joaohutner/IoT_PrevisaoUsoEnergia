# Módulo para a barra de navegação

# Imports
import dash_html_components as html
import dash_bootstrap_components as dbc

# Módulos customizados
from app import app
from modulos import constant

layout = dbc.Container(
        [
            dbc.Row([
                dbc.Col([
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src=app.get_asset_url(constant.APP_LOGO), height="30px"), className="col-md-2"),
                                dbc.Col(dbc.NavbarBrand("Smart Looker", className="ml-1"), className="col-md-2")
                            ],
                            align = "start",
                            no_gutters = True,
                            className = "p-3 pt-4 pb-3"
                        )
                    ),
                    html.Hr(),
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Home", href = "/paginas/dashboard"), style = constant.NAVITEM_STYLE),
                            dbc.NavItem(dbc.NavLink("Visão Geral", href = "/paginas/overview"), style = constant.NAVITEM_STYLE),
                            dbc.NavItem(dbc.NavLink("Monitoramento", href = "/paginas/monitoring"), style=constant.NAVITEM_STYLE),
                        ], 
                        className = "h-100", 
                        navbar = True,
                        pills = True,
                        vertical = True
                    ),
                ], 
                className = "h-100")
            ], 
            className = "h-100")
        ],
        style = constant.SIDEBAR_STYLE,
        fluid = True,
        className = "bg-dark text-white"
    )
