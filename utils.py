import dash_html_components as html
import dash_core_components as dcc


def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu()])

def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("instock-logo.png"),
                        className="logo",
                        style={
                            'height': '30%',
                            'width': '30%',
                            'float': 'middle',
                            'position': 'relative',
                            'padding-top': 0,
                            'padding-right': 0
                        },
                    ),
                    html.Img(
                        src=app.get_asset_url("volvo-logo-iron-mark.png"),
                        className="logo",
                        style={
                            'height': '12%',
                            'width': '12%',
                            'float': 'right',
                            'position': 'right',
                            'padding-top': 0,
                            'padding-right': 0,
                            'padding-left' : 0,
                            'margin-left': 0
                        },
                    ),
                    # html.Img(
                    #     src=app.get_asset_url("mack-logo.png"),
                    #     className="logo",
                    #     style={
                    #         'height' : '12%',
                    #         'width' : '12%',
                    #         'display': 'inline-block',
                    #         'float': 'right',
                    #         'position': 'right',
                    #         'padding-top': 0,
                    #         'padding-right': 0,
                    #         'margin-right':0
                    #     },
                    # ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H1("InStock")],
                        className="main-title1",
                        style={
                            'float': 'middle',
                            'text-align': 'center'
                        }
                    ),
                    html.Div(
                        [html.H5(
                            "Understanding Part Shortages Using Machine Learning")],
                        className="main-title2",
                        style={
                            'float': 'middle',
                            'text-align': 'center'
                        }
                    ),
                ],
                className="thirteen columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row1",
    )
    return header


def get_menu():
    menu = html.Div(
        # [
        #     dcc.Link(
        #         #"Units lost in 2019",
        #         href="/dash-report/q1",
        #         className="tab first",
        #     ),
        #     dcc.Link(
        #         #"Reasons for units lost in 2019",
        #         href="/dash-report/q2",
        #         className="tab",
        #     ),
        #     dcc.Link(
        #         #"Lines where the units are lost",
        #         href="/dash-report/q3",
        #         className="tab",
        #     ),
        # ],
        className="row all-tabs",
    )
    return menu


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table
