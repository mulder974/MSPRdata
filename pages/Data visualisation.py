import db
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit_keplergl as keplergl
from streamlit_keplergl import keplergl_static
import time





warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")





df = db.query_data()



df['Age moyen'] = df['Age moyen'].str.replace(',' ,'.')
df['Age moyen'] = df['Age moyen'].astype('float')

# df["Nombre d'habitants de la commune"] = df["Nombre d'habitants de la commune"].astype('int')


# df.drop(columns=['Unnamed: 0.1', 'index', 'Geo Point', 'Unnamed: 0','NOM_COM'],inplace=True)
st.title("La Donnée")
st.dataframe(df)



colonnes = ["Policiers pour 100 habs", "Evolution", "Revenu fiscal de référence des foyers fiscaux 2021"]




# Sélectionner la colonne pour le graphique boxplot
colonne_selectionnee1 = st.selectbox('Sélectionnez une colonne', colonnes)






fig, ax = plt.subplots()
sns.boxplot(data=df, x ='partie', y=colonne_selectionnee1)
st.pyplot(fig)



st.title("Distribution")




fig, ax = plt.subplots()
sns.histplot(data=df, x ='partie')
st.pyplot(fig)




st.title("Application avec KeplerGL")



st_keplergl = keplergl.KeplerGl(height=600)



st_keplergl.add_data(data=df, name='GeoData')
# Affichage de la carte KeplerGL dans Streamlit
keplergl_static(st_keplergl)