import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import dash_table


df = pd.read_csv('Sende.csv')
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Menu", className="display-4"),
        html.Hr(),
        html.P(
            "Navigate", className="lead"
        ),
        dbc.Nav(
            [
                
                dbc.NavLink("Upload", href="/page-1", active="exact"),
                dbc.NavLink("View", href="/", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)

def render_page_content(pathname):
    if pathname == "/":
        @app.callback(
                Output("download-component", "data"),
                Input("btn", "n_clicks"),
                prevent_initial_call=True,
                )
        def func(n_clicks):
            global df2
            df2=pd.DataFrame()
            header_list=['expected','Predicted','F1Score']
            df2 = df2.reindex(columns = header_list)
    
            n=df.index[df['F1Score']>float(0.8)]
        
            df2=(df.iloc[n])
            df2['Predicted']=df2['expected']
            return dcc.send_data_frame(df2.to_csv, "mydf_csv.csv")
        return[dash_table.DataTable(
        persistence=True,
        persistence_type='memory',
        id='table-dropdown',
        data=df.to_dict('records'), 
        
            #the contents of the table
        fixed_rows={'headers': True},
        
        columns=[
            {'id': 'expected', 'name': 'expected','editable':True, 'presentation': 'dropdown'},
            {'id': 'Predicted', 'name': 'Predicted','editable':True,'presentation': 'dropdown'},
            {'id': 'F1Score', 'name': 'F1Score'},
            {'id': 'Accept','name': 'Accept','editable':True,'presentation': 'dropdown'},
            {'id': 'Change','name':'Change','editable':True,'presentation': 'dropdown'}
        ],
        
        style_table={'height': 900},
        style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
        style_cell={'minWidth':95,'width':95,'maxWidth':95,'text-align':'center'},
        editable=True,
        
        dropdown={                      #dictionary of keys that represent column IDs,
            'expected': {                #its values are 'options' and 'clearable'
                'options': [            #'options' represents all rows' data under that column
                    {'label': i, 'value': i}
                    for i in df['expected'].unique()
                ],
                
                'clearable':True,
            },
            'Predicted': {
                'options':[
                    {'label': i, 'value': i}
                    for i in df['Predicted'].unique()
                ],
                     
                'clearable':True,
            },
            'Accept':{
                'options':[
                    {"label":"☑", "value": "checked"},
                    {"label": "☐", "value": "unchecked"}
                    #for i in df['Accept']
                ],
                "clearable": True,
            },
            'Change':{
                'options':[
                    {"label":"☑ ", "value": "checked"},
                    {"label": "☐", "value": "unchecked"},
                ],
                "clearable": False,
    
        }
        

        }),
        dbc.Checklist(id='Tick',
        options=[
            {'label': 'Tick on the box to save the changes', 'value':'sav'},
        
            ]
    
            )  ,
        dbc.Container(
            [
                dbc.Button(id='btn',
                children=[html.I(className="fa fa-download mr-1"), "Download"],
                color="info",
                className="mt-1"
                ),

            dcc.ConfirmDialog(id="confirm",message="Changes saved,Click on download"),
            dcc.Download(id="download-component")],className='m-4')]

   

        
    elif pathname == "/page-1":
        
               pass
            
    elif pathname == "/page-2":
        pass
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__=='__main__':
    app.run_server(debug=True, port=3000)