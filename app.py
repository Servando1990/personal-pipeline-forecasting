import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import statsmodels.api as sm

import plotly.express as px

data_folder = Path("data").resolve()

# Read in the data
df = pd.read_csv(data_folder / "marketing_data.csv", sep=";")

df.date = pd.to_datetime(df.date, format="%Y-%m-%d")
df['cohort'] = df['site'] + '_' + df['country']

df = df.groupby(['cohort', 'date']).sum().reset_index()

df['conversion_per_session'] = df['conversions'] / df['sessions']
df['price_per_impression'] = df['total_spend'] / df['impressions']

df = df.groupby(['cohort', 
                 pd.Grouper(key='date', 
                                      freq='D')])['total_spend', 
                                                  'impressions', 
                                                              'sessions',
                                                                'conversions', 
                                                                'price_per_impression', 
                                                              'conversion_per_session'].sum()
# Define a dash app
app = dash.Dash(__name__)

# Define the dropdown menus
cohort_dropdown = dcc.Dropdown(
    id='cohort-dropdown',
    options=[{'label': cohort, 'value': cohort} for cohort in df.index.levels[0]],
    value=df.index.levels[0][0]
)

feature_dropdown = dcc.Dropdown(
    id='feature-dropdown',
    options=[{'label': feature, 'value': feature} for feature in df.columns],
    value=df.columns[0]
)


# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Seasonal Decomposition Dashboard'),
    html.Div(children='Select a cohort and a feature to see the seasonal decomposition.'),
    html.Div(children=[
        html.Label('Cohort'),
        cohort_dropdown
 
    ]),
    html.Div(children=[
        html.Label('Feature'),
        feature_dropdown
    ]),

    html.Div(children=[
        dcc.Graph(id='graph1'),
        dcc.Graph(id='graph2')
    ])
])

# Define the callbacks for updating the graphs

@app.callback(
    dash.dependencies.Output('graph1', 'figure'),
    [dash.dependencies.Input('cohort-dropdown', 'value'),
     dash.dependencies.Input('feature-dropdown', 'value')])
def update_graph1(cohort, feature):
    df_cohort = df.loc[df.cohort == cohort]
    df_monthly = df_cohort.groupby(pd.Grouper(level='date', freq='M')).sum()
    fig = px.bar(df_monthly,
                  x=df_monthly.index,
                  y=feature,
                  barmode='group',
                  title=f'{feature} {cohort} Monthly'
                  )
    return fig

@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    [dash.dependencies.Input('cohort-dropdown', 'value'),
     dash.dependencies.Input('feature-dropdown', 'value')])

def update_graph2(cohort, feature):
    df_cohort = df.loc[cohort]
    series = df_cohort[feature]
    result = sm.tsa.seasonal_decompose(series, model='additive', period=30)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.observed.index,
        y=result.observed.values,
        mode='lines',
        name='Observed'
    ))

    fig.add_trace(go.Scatter(
        x=result.trend.index,
        y=result.trend.values,
        mode='lines',
        name='Trend'
    ))

    fig.add_trace(go.Scatter(
        x=result.seasonal.index,
        y=result.seasonal.values,
        mode='lines',
        name='Seasonal'
    ))

    fig.update_layout(
        title=f'{feature} {cohort} Seasonal Decomposition',
        xaxis_title='Date',
        yaxis_title='Value'
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(port=8051)