import streamlit as st
from streamlit_shap import st_shap
import shap
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from zipfile import ZipFile

# Chargement des données
@st.cache_data
def load_data():
    data=pd.read_csv("data_sample.csv", index_col=[0])
    description = pd.read_csv("HomeCredit_columns_description.csv", 
                            usecols=['Row', 'Description'], encoding= 'unicode_escape')
    id_client = data.SK_ID_CURR.values
    return data, description, id_client

data, description, id_client=load_data()
    
@st.cache_data
def load_shap():
    z = ZipFile("exp.zip")
    z.extractall()
    pickle_in = open("exp.pkl","rb")
    exp=pickle.load(pickle_in)
    return exp
exp=load_shap()

@st.cache_data
def load_threshold():
    pickle_in = open("threshold.pkl", "rb")
    threshold=pickle.load(pickle_in)
    return threshold
threshold=load_threshold()

@st.cache_data
def predict_credit():
    df_score=data[["SK_ID_CURR","score"]]
    return df_score
df_score=predict_credit()

@st.cache_data
def decision():
    data['decision'] = data['score'].apply(lambda x: 'Accepté' if x > threshold else 'Refusé')
    decision_counts = data['decision'].value_counts()
    decision_percentages = (decision_counts / len(data)) * 100
    plot_data = pd.DataFrame({
    'Decision': decision_counts.index,
    'Count': decision_counts.values,
    'Percentage': decision_percentages.values
    })
    return plot_data
plot_data=decision()

#Chargement de la selectbox
id = st.sidebar.selectbox("Client ID", id_client)

# Envoi des données client à l'API
clt=data[data["SK_ID_CURR"]==int(id)].drop(['TARGET', 'score'], axis=1)
data_clt={"client_id":int(id) , "features":clt.to_dict(orient="records")[0]}

#récupération du score de prédiction    
resp = requests.post(url="http://DefaultResourceGroup-WEU7.westeurope.azurecontainer.io/predict", data= json.dumps(data_clt))
if resp.status_code == 200:
    result = resp.json()
else:
    print(resp.json())

# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open("logo.png")
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")

st.title('Scoring crédit')

#Création de la jauge
gauge = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = result['score']*100,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': result['decision'], 'font': {'size': 48}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
        'bar': {'color': "black"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, threshold*100], 'color': 'red'},
            {'range': [threshold*100, 100], 'color': 'green'}],
        }))    
# préparation des données
X=data[data["SK_ID_CURR"]==int(id)]
idx=data.index.get_loc(data[data['SK_ID_CURR'] == id].index[0])

# Creation des tabs
tab_titles = ['Informations générales', 'Résultats client', 'Positionnement']
tabs = st.tabs(tab_titles)
 
# Ajout du contenu
with tabs[0]:
    st.header('Informations générales')

    st.write(f'Nombre de clients: {len(data)}')

    st.write(f'Principales caractéristiques') 
    # Centrage de l'image de la feature importance
    image = Image.open("sum_plot.png")
    st.image(image, use_column_width="always")
    
    #plot un camembert pour la décision crédit et pourcentage
    colors = {'Accepté': 'green', 'Refusé': 'red'}
    fig=px.pie(plot_data, names='Decision', values='Count', title='décision crédit',
             hover_data=['Percentage'], labels={'Percentage': 'Percentage (%)'}, color= 'Decision', color_discrete_map=colors)
    st.plotly_chart(fig)
    
with tabs[1]:
    st.header('Résultats client')
    # Plot gauge
    st.plotly_chart(gauge, use_container_width=True)
    resp_score="{:.2%}".format(result['score'])
    # Plot local feature importance
    st_shap(shap.plots.waterfall(exp[idx]))
 
with tabs[2]:
    st.header('Positionnement')
    st.write('Histogramme : la valeur pour le client apparait en rouge')
    feat_1 = st.selectbox("Sélectionnez une caractéristique", data.columns)
    default_color = "blue"
    x_value = X[feat_1].values[0]
    colors = {x_value: "red"}

    color_discrete_map = {
        c: colors.get(c, default_color) 
        for c in data[feat_1].unique()}


    fig_histogram = px.histogram(data, x=feat_1, color=feat_1, color_discrete_map=color_discrete_map, 
                                 text_auto=True, title=f"Distribution de {feat_1}")
    fig_histogram.update_traces(showlegend=False)
    st.plotly_chart(fig_histogram)

    st.write('Positionnement du client sur 2 axes')
    feat_2 = st.selectbox("Sélectionnez l'axe des abscisses", data.columns)
    feat_3 = st.selectbox("Sélectionnez l'axe des ordonnées", data.columns)
    
    x_value = X[feat_2].values[0]
    y_value = X[feat_3].values[0]
    fig = px.scatter(data, x=feat_2, y=feat_3, color="score", color_continuous_scale= px.colors.sequential.deep)
    fig.add_scatter(
        x=[x_value],
        y=[y_value],
        mode='markers',
        marker=dict(size=10, color='red'),  
        name='Client value'
    )
    st.plotly_chart(fig)

 



