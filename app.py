import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output
from utils import Header, make_dash_table

import pandas as pd
import numpy as np
import pathlib
import plotly.graph_objs as go
from datetime import datetime as dt
from sklearn.ensemble import RandomForestRegressor

# app.config['suppress_callback_exceptions']=True
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# Describe the layout/ UI of the app
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df = pd.read_excel("data/units.xlsx")
df.loc[df.Day == '-', 'Day'] = 7
df.MonthNumber = np.nan
df.loc[df.Month == 'Jan', 'MonthNumber'] = 1
df.loc[df.Month == 'Feb', 'MonthNumber'] = 2
df.loc[df.Month == 'Mar', 'MonthNumber'] = 3
df.loc[df.Month == 'Apr', 'MonthNumber'] = 4
df.loc[df.Month == 'May', 'MonthNumber'] = 5
df.loc[df.Month == 'Jun', 'MonthNumber'] = 6
df.loc[df.Month == 'Jul', 'MonthNumber'] = 7
df = df.dropna()
raw_data = df
print(raw_data.columns)
df_pivot = df.pivot_table(
    values='Units', index='Week',
    columns='Factory')

'''
-------------------------------------------------
Q1 : This following section is the code for building the prediction for number of units Q1
-------------------------------------------------
'''
df = df.drop(columns=['MainRootCause', 'DetailsOfTheRootCause', 'SupplierName',
                      'ParmaCode', 'PartReference', 'Total Loss Cost'])
target = 'Units'
X = df.drop(columns=[target])
y = df[target]

categorical_features = [
    column for column in X.columns if X[column].dtype.name == 'object']
X = pd.get_dummies(data=X, columns=categorical_features)
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
# Train the model on training data
rf.fit(X, y)
'''
-------------------------------------------------
End of section
-------------------------------------------------
'''


'''
-------------------------------------------------
Q2 : Predict reason
-------------------------------------------------
'''
q2df = raw_data.drop(columns=['DetailsOfTheRootCause', 'SupplierName',
                              'ParmaCode', 'PartReference', 'Total Loss Cost'])
q2df.loc[q2df.MainRootCause == 'Supplier delay',
         'MainRootCause'] = 'SupplierDelay'
q2df.loc[q2df.MainRootCause == 'Supplier delayed',
         'MainRootCause'] = 'SupplierDelay'
q2df = q2df.dropna()

'''
-------------------------------------------------
End of section Q2
-------------------------------------------------
'''


'''
-------------------------------------------------
Q3 : Lines that are impacted by Units lost
-------------------------------------------------
'''
q3df = raw_data.drop(columns=['DetailsOfTheRootCause', 'SupplierName', 'MainRootCause',
                              'ParmaCode', 'PartReference', 'Total Loss Cost'])
q3df = q3df.dropna()


'''
-------------------------------------------------
End of section Q3
-------------------------------------------------
'''

available_indicators = df['Factory'].unique()


# def create_layout(app):
#     # Page layouts
#     return html.Div(
#         [
app.layout = html.Div(children=[
    html.Div([Header(app)]),
    dcc.Tabs(id='tabs', children=[
        dcc.Tab(label='Units lost in 2019', children=[
            # page 1
            html.Div(
                [
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        "Units Lost in 2019"),
                                    html.Br([]),
                                    html.P(
                                        "\
                                    Parts shortages have been a significant threat to production \
                                    and overall profitability of the Hagerstown plant. \
                                    This tool summarizes the Material shortages at Point of use (Ma@PU)\
                                    Below chart is the average number of units lost each week in 2019 due to part shortages. \
                                    The main idea behind this tool is to forecast number of units that we could lose in the future, their reasons and \
                                    the lines that could be impacted. This can help eliminate holding costs, \
                                    lower operational expenses and improve efficiencies during peak periods. ",
                                        style={
                                            "color": "#ffffff", 'text-align': 'left', 'font-size': '12px'},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    # Row 4
                    html.Div([
                        html.Div([
                            dcc.Dropdown(
                                id='xaxis-column',
                                style={'width': '150px'},
                                options=[{'label': i, 'value': i}
                                         for i in available_indicators],
                                value=''
                            ),
                            # dcc.RadioItems(
                            #     id='xaxis-type',
                            #     options=[{'label': i, 'value': i}
                            #              for i in ['Week']],
                            #     value='Week',
                            #     labelStyle={
                            #         'display': 'inline-block'}
                            # )
                        ]),
                        html.Div([
                            dcc.Graph(
                                animate=True,
                                id='indicator-graphic',
                                figure={
                                    'data': [
                                        go.Scatter(
                                            x=df_pivot.index,
                                            y=df_pivot[i],
                                            text=df['Factory'],
                                            mode='lines+markers',
                                            opacity=0.7,
                                            marker={
                                                'size': 7,
                                                'line': {'width': 0.5, 'color': 'white'}
                                            },
                                            name=i
                                        ) for i in df.Factory.unique()
                                    ],
                                    'layout': go.Layout(
                                        xaxis={
                                            'title': 'Week of the year'},
                                        yaxis={
                                            'title': 'Average units lost'},
                                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                        legend={
                                            'x': 0, 'y': 1},
                                        # plot_bgcolor=app_color["graph_bg"],
                                        # paper_bgcolor=app_color["graph_bg"],
                                        hovermode='closest'
                                    )
                                }
                            ),
                            dcc.Slider(
                                id='year--slider',
                                min=df['Week'].min(),
                                max=df['Week'].max(),
                                value=df['Week'].max(),
                                marks={str(i): str(i)
                                       for i in df['Week'].unique()},
                                step=None
                            )
                        ], style={'marginBottom': '3em'}),
                        html.Div([
                            html.Div(
                                [
                                    html.H5(
                                        "Predict Units that could be Lost"),
                                    html.Br([]),
                                    html.P(
                                        "\
                                        This section predicts the number of units that could be lost for a given week in the future due to \
                                        material shortages in supply chain ? ",
                                        style={
                                            "color": "#ffffff", 'text-align': 'left', 'font-size': '14px'},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            ),
                            html.Div(
                                'Powertrain site of your interest'),
                            dcc.RadioItems(
                                id='choose-site-for-units-lost',
                                options=[{'label': i, 'value': i}
                                         for i in available_indicators],
                                value='HAG',
                                labelStyle={
                                    'display': 'inline-block'}
                            ),
                            html.Div(
                                'Line that you are interested in :'),
                            dcc.Dropdown(
                                id='choose-line-for-units-lost',
                                options=[{'label': i, 'value': i}
                                         for i in df['UnitImpacted'].unique()],
                                value='HDE 9'
                            ),
                            html.Div('Week number'),
                            dcc.Input(
                                id='chosen-week-number',
                                placeholder='Enter a week number...',
                                type='number',
                                value='35',
                                max='45',
                                min='1',
                                style={
                                    'display': 'inline-block'}
                            ),
                            html.Div('Day number'),
                            dcc.Input(
                                id='chosen-day-number',
                                placeholder='Enter a day number...',
                                type='number',
                                value='2',
                                max='7',
                                min='1',
                                style={
                                    'display': 'inline-block'}
                            ),
                            html.Div('Month number'),
                            dcc.Input(
                                id='monthNumber',
                                placeholder='Enter a month number...',
                                type='number',
                                value='8',
                                min='1',
                                max='10',
                                style={
                                    'display': 'inline-block'}
                            ),
                            html.Div(
                                'Choose month (Ensure it matches with week and month number)'),
                            dcc.Dropdown(
                                id='choose-month-for-units-lost',
                                options=[
                                    {'label': 'January',
                                     'value': 'Jan'},
                                    {'label': 'February',
                                     'value': 'Feb'},
                                    {'label': 'March',
                                     'value': 'Mar'},
                                    {'label': 'April',
                                     'value': 'Apr'},
                                    {'label': 'May',
                                     'value': 'May'},
                                    {'label': 'June',
                                     'value': 'Jun'},
                                    {'label': 'July',
                                     'value': 'Jul'},
                                    {'label': 'August',
                                     'value': 'Aug'},
                                    {'label': 'September',
                                     'value': 'Sept'}
                                ],
                                value='Aug'
                            ),
                            # html.Div('Loss Category'),
                            # dcc.Dropdown(
                            #     id = 'choose-losstype-for-units-lost',
                            #     options=[{'label': i, 'value': i} for i in df['LossCategory'].unique()],
                            #     value='Units_lost',
                            # ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H5(id='output-container-button',
                                                             children='Predicted units that could be lost for the given parameters is :'
                                                             ),
                                                ],
                                                className="product",
                                            )
                                        ],
                                        className="row",
                                    ),
                                ]
                            )
                        ])
                    ])
                ],
                className="sub_page",
            ),
        ]),

        # Tab 2
        dcc.Tab(label='Reasons for units lost', children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H5(
                                                "Possible reason for units lost"),
                                            html.Br([]),
                                            html.P(
                                                "\
                                                This section shows a count of different reasons which caused part shortages at point of use.",
                                                style={
                                                    "color": "#ffffff", 'text-align': 'left', 'font-size': '12px'},
                                                className="row",
                                            ),
                                        ],
                                        className="product",
                                    )
                                ],
                                className="row",
                            ),
                        ]),

                    html.Div([
                        html.Div('Site Name'),
                        dcc.Dropdown(
                            id='site-list',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value='HAG'
                        ),
                        dcc.Graph(
                            id='mainrootcause',
                            figure={
                                'data': [
                                    {'x': q2df['MainRootCause'], 'y': q2df['MainRootCause'].value_counts(),
                                     'type': 'bar', 'name': 'MainRootCAuse'},
                                ],
                                'layout': [go.Layout(
                                        xaxis={
                                            'title': 'Main Root Cause for Part Shortages'},
                                        yaxis={
                                            'title': 'Count of Main Root Cause'},
                                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                        legend={
                                            'x': 0, 'y': 1},
                                        # plot_bgcolor=app_color["graph_bg"],
                                        # paper_bgcolor=app_color["graph_bg"],
                                        hovermode='closest'
                                    )
                                ]
                            }
                            
                        ),
                        # dcc.DatePickerSingle(
                        # 	id='date-picker-mainrootcause',
                        # 	date=dt(2019, 8, 15)
                        # ),
                        html.Div([
                            html.Div(
                                [
                                    html.H5(
                                        "Predict reason for the lost units"),
                                    html.Br([]),
                                    html.P(
                                        "\
                                        This section predicts the most probable reason for units lost on the chosen day.",
                                        style={
                                            "color": "#ffffff", 'text-align': 'left', 'font-size': '12px'},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            ),
                        ]),
                        html.Div('Unit Impacted'),
                        dcc.Dropdown(
                            id='unitImpacted-mainrootcause',
                            options=[{'label': i, 'value': i}
                                     for i in q2df['UnitImpacted'].unique()],
                            value='HDE 13'
                        ),
                        html.Div('Loss Category'),
                        dcc.Dropdown(
                            id='LossCategory-mainrootcause',
                            options=[{'label': i, 'value': i}
                                     for i in q2df['LossCategory'].unique()],
                            value='Units_lost'
                        ),
                        html.Div('Month'),
                        dcc.Dropdown(
                            id='month-mainrootcause',
                            options=[
                                {'label': 'January',
                                 'value': 'Jan'},
                                {'label': 'February',
                                 'value': 'Feb'},
                                {'label': 'March',
                                 'value': 'Mar'},
                                {'label': 'April',
                                 'value': 'Apr'},
                                {'label': 'May',
                                 'value': 'May'},
                                {'label': 'June',
                                 'value': 'Jun'},
                                {'label': 'July',
                                 'value': 'Jul'},
                                {'label': 'August',
                                 'value': 'Aug'},
                                {'label': 'September',
                                 'value': 'Sept'}
                            ],
                            value='Aug'
                        ),
                        html.Div('Week Number'),
                        dcc.Input(
                            id='week-mainrootcause',
                            placeholder='Enter a week number...',
                            type='number',
                            value='34'
                        ),
                        html.Div('Day Number'),
                        dcc.Input(
                            id='day-mainrootcause',
                            placeholder='Enter a day number...',
                            type='number',
                            value='2'
                        ),
                        html.Div('Month number'),
                        dcc.Input(
                            id='monthnumber-mainrootcause',
                            placeholder='Enter a monthnumber number...',
                            type='number',
                            value='8'
                        ),
                        html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H5(id='output-container-mainrootcause',
                                                                children='Enter the values to see predicted root cause'
                                                             #  [
                                                             #      style={
                                                             #          "color": "#ffffff", 'text-align': 'left', 'font-size': '12px'},
                                                             #      className="row",
                                                             #  ]
                                                             ),
                                                ],
                                                className="product",
                                            )
                                        ],
                                        className="row",
                                    ),
                                ]
                            )
                    ]),
                ],
                className="sub_page",
            ),
        ]),
        # Tab 3
        dcc.Tab(id='tab3', label='Line impacted', children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H5(
                                                "Number of units lost at each Line"),
                                            html.Br([]),
                                            html.P(
                                                "\
                                                This section shows the number of times units are lost in each line across different sites.",
                                                style={
                                                    "color": "#ffffff", 'text-align': 'left', 'font-size': '12px'},
                                                className="row",
                                            ),
                                        ],
                                        className="product",
                                    )
                                ],
                                className="row",
                            ),
                        ]),
                    html.Div([
                        html.Div('Site Name for Line'),
                        dcc.Dropdown(
                            id='site-list-line',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value='HAG'
                        ),
                        dcc.Graph(
                            id='graph-line',
                            figure={
                                'data': [
                                    {'x': q3df['UnitImpacted'], 'y': q3df['UnitImpacted'].value_counts(),
                                     'type': 'bar', 'name': 'UnitImpacted'},
                                ],
                                'layout': go.Layout(
                                        xaxis={
                                            'title': 'Line impacted due Part Shortages'},
                                        yaxis={
                                            'title': 'Count of line impacted'},
                                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                        legend={
                                            'x': 0, 'y': 1},
                                        # plot_bgcolor=app_color["graph_bg"],
                                        # paper_bgcolor=app_color["graph_bg"],
                                        hovermode='closest'
                                    )
                            }
                        ),
                        # dcc.DatePickerSingle(
                        # 	id='date-picker-mainrootcause',
                        # 	date=dt(2019, 8, 15)
                        # ),
                        html.Div([
                            html.Div(
                                [
                                    html.H5(
                                        "Predict line for the lost units"),
                                    html.Br([]),
                                    html.P(
                                        "\
                                            This section predicts the most probable line that is going to be affected on the chosen day.",
                                        style={
                                            "color": "#ffffff", 'text-align': 'left', 'font-size': '12px'},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            ),
                        ]),
                        html.Div('Loss Category'),
                        dcc.Dropdown(
                            id='LossCategory-line',
                            options=[{'label': i, 'value': i}
                                     for i in q2df['LossCategory'].unique()],
                            value='Units_lost'
                        ),
                        html.Div('Month'),
                        dcc.Dropdown(
                            id='month-line',
                            options=[
                                {'label': 'January',
                                 'value': 'Jan'},
                                {'label': 'February',
                                 'value': 'Feb'},
                                {'label': 'March',
                                 'value': 'Mar'},
                                {'label': 'April',
                                 'value': 'Apr'},
                                {'label': 'May',
                                 'value': 'May'},
                                {'label': 'June',
                                 'value': 'Jun'},
                                {'label': 'July',
                                 'value': 'Jul'},
                                {'label': 'August',
                                 'value': 'Aug'},
                                {'label': 'September',
                                 'value': 'Sept'}
                            ],
                            value='Aug'
                        ),
                        html.Div('Week Number'),
                        dcc.Input(
                            id='week-line',
                            placeholder='Enter a week number...',
                            type='number',
                            value='34'
                        ),
                        html.Div('Day Number'),
                        dcc.Input(
                            id='day-line',
                            placeholder='Enter a day number...',
                            type='number',
                            value='2'
                        ),
                        html.Div('Month number'),
                        dcc.Input(
                            id='monthnumber-line',
                            placeholder='Enter a monthnumber number...',
                            type='number',
                            value='8'
                        ),
                        html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H5(id='output-container-line',
                                                                children='Enter the values to see the line that could be afftected due to part shortages'
                                                             #  [
                                                             #      style={
                                                             #          "color": "#ffffff", 'text-align': 'left', 'font-size': '12px'},
                                                             #      className="row",
                                                             #  ]
                                                             ),
                                                ],
                                                className="product",
                                            )
                                        ],
                                        className="row",
                                    ),
                                ]
                            )
                    ]),
                ],
                className="sub_page",
            ),
        ]),
    ]),
],
    className="page",
)


@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('year--slider', 'value')])
def update_graph(xaxis_column_name, year_value):
    #xaxis-type = 'Week'
    dff = df[df['Week'] <= year_value]
    dff = dff[dff['Factory'] == xaxis_column_name]
    df_pivot = dff.pivot_table(
        values='Units', index='Week',
        columns='Factory')
    return {
        'data': [go.Scatter(
            x=df_pivot.index,
            y=df_pivot[xaxis_column_name],
            text=dff['Factory'],
            mode='lines+markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
        )],
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name,
            },
            yaxis={'title': 'Average units lost'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


@app.callback(
    Output('output-container-button', 'children'),
    [Input('chosen-week-number', 'value'),
     Input('choose-site-for-units-lost', 'value'),
     Input('choose-line-for-units-lost', 'value'),
     Input('chosen-day-number', 'value'),
     Input('choose-month-for-units-lost', 'value'),
     # Input('choose-losstype-for-units-lost', 'value'),
     Input('monthNumber', 'value')]
)
def update_output(week, site, line, day, month, monthNumber):
    CFactory = site
    CMonth = month
    CDay = day
    CUnitImpacted = line
    CLossCategory = 'Units_lost'
    CMonthNumber = monthNumber
    CWeek = week

    new = pd.DataFrame(np.nan, index=[0], columns=['Week', 'MonthNumber', 'Factory_CUR', 'Factory_HAG', 'Factory_KOP',
                                                   'Factory_SDE', 'Factory_VNX', 'Month_Apr', 'Month_Feb', 'Month_Jan',
                                                   'Month_Jul', 'Month_Jun', 'Month_Mar', 'Month_May', 'Day_1', 'Day_2',
                                                   'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7', 'UnitImpacted_AMT',
                                                   'UnitImpacted_Axles', 'UnitImpacted_DUO', 'UnitImpacted_Duo',
                                                   'UnitImpacted_HDE 11', 'UnitImpacted_HDE 12', 'UnitImpacted_HDE 13',
                                                   'UnitImpacted_HDE 13V', 'UnitImpacted_HDE 16', 'UnitImpacted_HDE 9',
                                                   'UnitImpacted_HDE Penta', 'UnitImpacted_IPS', 'UnitImpacted_MDE 5',
                                                   'UnitImpacted_MDE 8', 'UnitImpacted_PENTA', 'UnitImpacted_Powertronic',
                                                   'UnitImpacted_SMT', 'UnitImpacted_T300', 'LossCategory_Blocked_units',
                                                   'LossCategory_Drumbeats_E', 'LossCategory_Drumbeats_T',
                                                   'LossCategory_Incomplete_units', 'LossCategory_Missing_at_customer',
                                                   'LossCategory_Units_lost'])

    Fac_cols = [col for col in new.columns if CFactory in col]
    new[Fac_cols] = 1

    Mon_cols = [col for col in new.columns if CMonth in col]
    new[Mon_cols] = 1

    UnitImpacted_cols = [col for col in new.columns if CUnitImpacted in col]
    new[UnitImpacted_cols] = 1

    LossCategory_cols = [col for col in new.columns if CLossCategory in col]
    new[LossCategory_cols] = 1

    Day_cols = [col for col in new.columns if str(CDay) in col]
    new[Day_cols] = 1

    new['MonthNumber'] = CMonthNumber
    # new['Day'] = CDay
    new['Week'] = CWeek
    new = new.replace(np.nan, 0)

    predictions_rf = rf.predict(new)
    print(type(predictions_rf))
    return 'The predicted loss units value is "{}" '.format(round(predictions_rf[0]))


@app.callback(
    Output('mainrootcause', 'figure'),
    [Input('site-list', 'value')])
def update_mainrootcause(sitelist):
    newq2df = q2df.loc[q2df['Factory'] == sitelist]
    return {
        'data': [
            {'x': newq2df['MainRootCause'], 'y': newq2df['MainRootCause'].value_counts(),
             'type': 'bar', 'name': 'MainRootCAuse'},
        ]
    }


@app.callback(
    Output('output-container-mainrootcause', 'children'),
    [Input('site-list', 'value'),
        Input('month-mainrootcause', 'value'),
        Input('unitImpacted-mainrootcause', 'value'),
        Input('LossCategory-mainrootcause', 'value'),
        Input('week-mainrootcause', 'value'),
        Input('day-mainrootcause', 'value'),
        Input('monthnumber-mainrootcause', 'value')])
def update_predicted_mainrootcause(site, month, unitimpacted, losscategory, week, day, monthnumber):
    MFactory = site
    MMonth = month
    MDay = day
    MUnitImpacted = unitimpacted
    MLossCategory = losscategory
    MWeek = week

    target_mainrootcause = 'MainRootCause'
    X_mainrootcause = q2df.drop(columns=[target_mainrootcause])
    y_mainrootcause = q2df[target_mainrootcause]

    from sklearn.preprocessing import LabelEncoder
    X_mainrootcause = pd.get_dummies(X_mainrootcause, sparse=True)
    # encode string class values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y_mainrootcause)
    label_encoded_y = label_encoder.transform(y_mainrootcause)

    from sklearn.ensemble import RandomForestClassifier
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_mainrootcause, label_encoded_y)

    new = pd.DataFrame(np.nan, index=[0], columns=['Week', 'Units', 'MonthNumber', 'Factory_CUR', 'Factory_HAG',
                                                   'Factory_KOP', 'Factory_SDE', 'Factory_VNX', 'Month_Apr', 'Month_Feb',
                                                   'Month_Jan', 'Month_Jul', 'Month_Jun', 'Month_Mar', 'Month_May',
                                                   'Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7',
                                                   'UnitImpacted_AMT', 'UnitImpacted_Axles', 'UnitImpacted_DUO',
                                                   'UnitImpacted_Duo', 'UnitImpacted_HDE 11', 'UnitImpacted_HDE 12',
                                                   'UnitImpacted_HDE 13', 'UnitImpacted_HDE 13V', 'UnitImpacted_HDE 16',
                                                   'UnitImpacted_HDE 9', 'UnitImpacted_HDE Penta', 'UnitImpacted_IPS',
                                                   'UnitImpacted_MDE 5', 'UnitImpacted_MDE 8', 'UnitImpacted_PENTA',
                                                   'UnitImpacted_Powertronic', 'UnitImpacted_SMT', 'UnitImpacted_T300',
                                                   'LossCategory_Blocked_units', 'LossCategory_Drumbeats_E',
                                                   'LossCategory_Drumbeats_T', 'LossCategory_Incomplete_units',
                                                   'LossCategory_Missing_at_customer', 'LossCategory_Units_lost'])

    Fac_cols = [col for col in new.columns if MFactory in col]
    new[Fac_cols] = 1
    Mon_cols = [col for col in new.columns if MMonth in col]
    new[Mon_cols] = 1
    UnitImpacted_cols = [col for col in new.columns if MUnitImpacted in col]
    new[UnitImpacted_cols] = 1
    LossCategory_cols = [col for col in new.columns if MLossCategory in col]
    new[LossCategory_cols] = 1
    Day_cols = [col for col in new.columns if str(MDay) in col]
    new[Day_cols] = 1

    # new['Day'] = MDay
    new['Week'] = MWeek
    new['MonthNumber'] = monthnumber
    new = new.replace(np.nan, 0)

    pred = (clf.predict(new))
    return 'The predicted loss units is probably going to be caused by is "{}" '.format((label_encoder.inverse_transform(pred))[0])


# Q3 Callbacks
@app.callback(
    Output('graph-line', 'figure'),
    [Input('site-list-line', 'value')])
def update_line(sitelist):
    newq3df = q3df.loc[q3df['Factory'] == sitelist]
    return {
        'data': [
            {'x': newq3df['UnitImpacted'], 'y': newq3df['UnitImpacted'].value_counts(),
             'type': 'bar', 'name': 'UnitImpacted'},
        ]
    }


@app.callback(
    Output('output-container-line', 'children'),
    [Input('site-list-line', 'value'),
        Input('day-line', 'value'),
        Input('month-line', 'value'),
        Input('LossCategory-line', 'value'),
        Input('week-line', 'value'),
        Input('monthnumber-line', 'value')]
)
def update_predicted_line(site, day, month, losscategory, week, monthnumber):
    LFactory = site
    LMonth = month
    LDay = day
    # LUnitImpacted = 'HDE 9'
    LLossCategory = losscategory
    LWeek = week
    monthnumber = monthnumber

    target_line = 'UnitImpacted'
    X_line = q3df.drop(columns=[target_line])
    y_line = q3df[target_line]

    from sklearn.preprocessing import LabelEncoder
    X_line = pd.get_dummies(X_line, sparse=True)
    # encode string class values as integers
    label_encoder_l = LabelEncoder()
    label_encoder_l = label_encoder_l.fit(y_line)
    label_encoded_l_y = label_encoder_l.transform(y_line)
    from sklearn.ensemble import RandomForestClassifier
    # Create a Gaussian Classifier
    clfl = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clfl.fit(X_line, label_encoded_l_y)

    newl = pd.DataFrame(np.nan, index=[0], columns=['Week', 'Units', 'MonthNumber', 'Factory_CUR', 'Factory_HAG',
                                                    'Factory_KOP', 'Factory_SDE', 'Factory_VNX', 'Month_Apr', 'Month_Feb',
                                                    'Month_Jan', 'Month_Jul', 'Month_Jun', 'Month_Mar', 'Month_May',
                                                    'Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7',
                                                    'LossCategory_Blocked_units', 'LossCategory_Drumbeats_E',
                                                    'LossCategory_Drumbeats_T', 'LossCategory_Incomplete_units',
                                                    'LossCategory_Missing_at_customer', 'LossCategory_Units_lost'])

    Fac_cols = [col for col in newl.columns if LFactory in col]
    newl[Fac_cols] = 1
    Mon_cols = [col for col in newl.columns if LMonth in col]
    newl[Mon_cols] = 1
    LossCategory_cols = [col for col in newl.columns if LLossCategory in col]
    newl[LossCategory_cols] = 1
    Day_cols = [col for col in newl.columns if str(LDay) in col]
    newl[Day_cols] = 1

    # new['Day'] = MDay
    newl['Week'] = LWeek
    newl['MonthNumber'] = monthnumber
    newl = newl.replace(np.nan, 0)

    predl = (clfl.predict(newl))
    return 'The predicted line that could be impacted is : "{}" '.format((label_encoder_l.inverse_transform(predl))[0])


if __name__ == "__main__":
    app.run_server(debug=False)
