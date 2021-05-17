# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:40:25 2021

@author: Rein Arnold
"""

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px


external_stylesheets = ['http://psychiatrieschoorl.nl/css/stylesheet.css']


import Project1

#Make dataframes

#Raw Data
df_raw = Project1.all_data[['Power_kW','temp_C','HR','windSpeed_m/s','windGust_m/s','pres_mbar','solarRad_W/m2','rain_mm/h','rain_day']]

#Feature selection
df_rf = pd.DataFrame(Project1.all_data.columns[1:17],columns=['Name of variable'])
df_rf['Correlation score'] = Project1.model.feature_importances_

df_k = pd.DataFrame(Project1.all_data.columns[1:17],columns=['Name of variable'])
df_k['Correlation score'] = Project1.fit.scores_


#Regression
##Summary
regressors = ['Linear Regression','Random Forest','Decision Tree','Gradient Boosting','XGradient Boosting','Bootstrapping','Neural Networks']
mean_absolute = [Project1.MAE_LR,Project1.MAE_RF,Project1.MAE_DT,Project1.MAE_GB,Project1.MAE_XGB,Project1.MAE_BT,Project1.MAE_NN]
mean_squared = [Project1.MSE_LR,Project1.MSE_RF,Project1.MSE_DT,Project1.MSE_GB,Project1.MSE_XGB,Project1.MSE_BT,Project1.MSE_NN]
root_mean_squared = [Project1.RMSE_LR,Project1.RMSE_RF,Project1.RMSE_DT,Project1.RMSE_GB,Project1.RMSE_XGB,Project1.RMSE_BT,Project1.RMSE_NN]
cv_root_mean_squared = [Project1.cvRMSE_LR,Project1.cvRMSE_RF,Project1.cvRMSE_DT,Project1.cvRMSE_GB,Project1.cvRMSE_XGB,Project1.cvRMSE_BT,Project1.cvRMSE_NN]

df_errors = pd.DataFrame(regressors,columns=['Model'])
df_errors['Mean absolute error'] = mean_absolute
df_errors['Mean squared error'] = mean_squared
df_errors['Root mean squared error'] = root_mean_squared
df_errors['CV Root mean squared error'] = cv_root_mean_squared 

##Linear Regression
df_LR = pd.DataFrame(Project1.y_pred_NN[1:200], columns=['Model'])
df_LR['Real'] = Project1.y_test[1:200]

##Random Forest 
df_RF = pd.DataFrame(Project1.y_pred_RF[1:200], columns=['Model'])
df_RF['Real'] = Project1.y_test[1:200]

##Decision Tree
df_DT = pd.DataFrame(Project1.y_pred_DT[1:200], columns=['Model'])
df_DT['Real'] = Project1.y_test[1:200]

##Gradient Boosting
df_GB = pd.DataFrame(Project1.y_pred_GB[1:200], columns=['Model'])
df_GB['Real'] = Project1.y_test[1:200]

##XGradient Boosting
df_XGB = pd.DataFrame(Project1.y_pred_XGB[1:200], columns=['Model'])
df_XGB['Real'] = Project1.y_test[1:200]

##Bootstrapping
df_BT = pd.DataFrame(Project1.y_pred_BT[1:200], columns=['Model'])
df_BT['Real'] = Project1.y_test[1:200]

##Neural Networks
df_NN = pd.DataFrame(Project1.y_pred_NN[1:200], columns=['Model'])
df_NN['Real'] = Project1.y_test[1:200]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H2(['Project 2 - Results of Energy Forecasting for the Central Building of the IST Campus'],
        )
        ], style={'color': '#3D9970'})
    ,
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Data Analysis', value='tab-1'),
        dcc.Tab(label='Clustering', value='tab-2'),
        dcc.Tab(label='Feature Selection', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-4')  ,
    ]),
    html.Div(id='tabs-content'),
    html.Div(['by Rein Arnold (98023)'], style={'float': 'right','display': 'inline-block','position': 'absolute','top': '10px','right': '10px','font-style': 'italic'})
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('Exploratory Data Analysis'),
            html.P('The raw data consists of the power consumption of the IST Central Building for both 2017 and 2018, together with weather data acquired by the weather station at IST. The interactive boxplot below gives a comprehensive and insightful overview of the used raw data in the forecasting model.'),
            dcc.RadioItems(
        id='y-axis', 
        options=[{'value': y, 'label': y} 
                 for y in df_raw],
        value=['Power_kW'], style={'padding-top': '15px'},
        labelStyle={'display': 'inline-block'}),
            dcc.Graph(id="box-plot",className="eight columns")
                    ],className="nine columns")
            
    
    elif tab == 'tab-2':
        return html.Div([
     html.Div([
            html.Div([
                html.Div(['On the basis of the elbow curve displayed below the number of clusters is decided at 3, as the relative gain in score is low for n>3.'],
                         style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '15px', 'padding-bottom': '10px'}
                     ),
                html.Img(src='assets/elbow.png',className="twelve columns"),
                html.Div(['The four figures on the right side of this page show the three clusters for various selected variables. This gives a good insight in the daily and weekly variation of the power consumption. The three clusters which can be distinguished are the base load (green), the mid-peak (blue) and the peak load (red). The two figures below give further insight in the various clusters. The left figure shows three typical daily patterns: a weekend or holiday day with a stable power consumption, a day with a medium power consumption and a day with a high power consumption. The right figure gives a 3D overview of the weekly power consumption. '],
                         style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '15px', 'padding-bottom': '10px'}
                    )
                
        ]       ,className="four columns"),
                html.Div([
                    html.Div([  
                        html.Img(src='assets/cluster_a.png',className="six columns"),
                        html.Img(src='assets/cluster_b.png',className="six columns"),                        
                        ],className="row"),
                    html.Div([
                        html.Img(src='assets/cluster_c.png',className="six columns"),
                        html.Img(src='assets/cluster_d.png',className="six columns"  ),                                          
                        ],className="row")
        ],className="eight columns")
                ],className="row"),
     html.Div([
         html.Img(src='assets/cluster_set.png',className="eight columns"),
         html.Img(src='assets/cluster_3d.png',className="four columns")
         ],className="row")
     
     ])    
            
    elif tab == 'tab-3':
            return html.Div([
            html.Div(['The features for the forecasting model are selected according to the Random Forest feature importances scale, as shown below. The wind speed, wind gust, rain, rain per day and pressure variables are disregarded in the model because of their insignificance.'],style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '15px'}),
                dcc.Graph(id='features',    
                      figure= px.histogram(df_k, x="Name of variable", y="Correlation score", log_y=True, title="Random Forest feature importances (log scale)"))
        ])
            
    
    elif tab == 'tab-4':
        return html.Div([
            dcc.Dropdown( 
        id='dropdown',
        options=[
            {'label': 'Summary', 'value': 'sum'},
            {'label': 'Linear Regression', 'value': 'LR'},
            {'label': 'Random Forest', 'value': 'RF'},
            {'label': 'Decision Tree', 'value': 'DT'},
            {'label': 'Gradient Boosting', 'value': 'GB'},
            {'label': 'XGradient Boosting', 'value': 'XGB'},
            {'label': 'Bootstrapping', 'value': 'BT'},
            {'label': 'Neural Networks', 'value': 'NN'}
        ], 
        value='sum'
        ),
            html.Div(id='regression_model')
        ])
    
    
@app.callback(Output('regression_model', 'children'), 
              Input('dropdown', 'value'))

def render_model(model_name):
    
    if model_name == 'sum':
        return html.Div([
            html.H6('Overview of errors obtained for each forecasting method'),
            html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df_errors.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df_errors.iloc[i][col]) for col in df_errors.columns
            ]) for i in range(len(df_errors))
        ])
    ])       
            ])
            

    elif model_name == 'LR':
        return html.Div([
            dcc.Graph(id='graph1', 
              figure= px.scatter(x=Project1.y_pred_LR, y=Project1.y_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ),
            dcc.Graph(id='graph2', 
              figure= px.line(df_LR,
                              title="First 200 entries of tested data")    
              ),    
            ])

    elif model_name == 'RF':
        return html.Div([
            dcc.Graph(id='graph3', 
              figure= px.scatter(x=Project1.y_pred_RF, y=Project1.y_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ), 
            dcc.Graph(id='graph4', 
              figure= px.line(df_RF,
                              title="First 200 entries of tested data")
              ),
            ])


    elif model_name == 'DT':
        return html.Div([
           dcc.Graph(id='graph5', 
              figure= px.scatter(x=Project1.y_pred_DT, y=Project1.y_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ),  
            dcc.Graph(id='graph6', 
              figure= px.line(df_DT,
                              title="First 200 entries of tested data")
              ),    
            ])

    elif model_name == 'GB':
        return html.Div([
            dcc.Graph(id='graph7', 
              figure= px.scatter(x=Project1.y_pred_GB, y=Project1.y_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ),
            dcc.Graph(id='graph8', 
              figure= px.line(df_GB,
                              title="First 200 entries of tested data")
              ),
            ])

    elif model_name == 'XGB':
        return html.Div([
            dcc.Graph(id='graph9', 
              figure= px.scatter(x=Project1.y_pred_XGB, y=Project1.y_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ),  
            dcc.Graph(id='graph10', 
              figure= px.line(df_XGB,
                              title="First 200 entries of tested data")
              ),    
            ])

    elif model_name == 'BT':
        return html.Div([
            dcc.Graph(id='graph11', 
              figure= px.scatter(x=Project1.y_pred_BT, y=Project1.y_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ),
            dcc.Graph(id='graph12', 
              figure= px.line(df_BT,
                              title="First 200 entries of tested data")
              ),            
            ])
    
    elif model_name == 'NN':
        return html.Div([
           dcc.Graph(id='graph13', 
              figure= px.scatter(x=Project1.y_pred_NN, y=Project1.y_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ),
            dcc.Graph(id='graph14', 
              figure= px.line(df_NN,
                              title="First 200 entries of tested data")
              ),    
            ])

@app.callback(
    Output("box-plot", "figure"), 
    [Input("y-axis", "value")])
def generate_chart(y):
    fig = px.box(df_raw, y=y)
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)

