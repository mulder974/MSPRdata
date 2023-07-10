import pandas as pd
import psycopg2 as psy

from sqlalchemy import create_engine
import pandas as pd


import pickle

import db


def load_model_from_disk(model_path):
    loaded_model = pickle.load(open(f'{model_path}', 'rb'))
    loaded_model = pickle.load(open(f'{model_path}', 'rb'))
    return loaded_model




def insert_model(model_name):
    model_to_save = pickle.load(open(f'{model_name}.SAV', 'rb'))
    connection = psy.connect("postgresql://pierresimplon974:hIf3kgq7rJiQ@ep-square-salad-072972.eu-central-1.aws.neon.tech/neondb?options=endpoint%3Dep-square-salad-072972")
    connection.set_session(autocommit=True)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO models(model_name,model_pickle) VALUES (%s, %s)", (model_name,psy.Binary(model_to_save)))


def insert_data():
    df = pd.read_csv("coord.csv")
    df = df.drop("Unnamed: 0.1" , axis = 1)

    # Create an engine instance
    alchemyEngine = create_engine('postgresql://pierresimplon974:hIf3kgq7rJiQ@ep-square-salad-072972.eu-central-1.aws.neon.tech/neondb?options=endpoint%3Dep-square-salad-072972', pool_recycle=3600)

    # Connect to PostgreSQL server
    dbConnection = alchemyEngine.connect()

    try:
        # Insert whole DataFrame into PostgreSQL
        df.to_sql('elections_data', dbConnection, if_exists='replace')
    except ValueError as vx:
        print(vx)
    except Exception as ex:
        print(ex)
    else:
        print("PostgreSQL Table has been created successfully.")
    finally:
        dbConnection.close()

0


def query_data():


    conn = psy.connect("postgresql://pierresimplon974:hIf3kgq7rJiQ@ep-square-salad-072972.eu-central-1.aws.neon.tech/neondb?options=endpoint%3Dep-square-salad-072972")
    conn.set_session(autocommit=True)
    cursor = conn.cursor()
    sql_query = "SELECT * FROM elections_data"
    df = pd.read_sql(sql_query, conn)

    conn.close()
    return df



def query_model_id(model_name):
    connection = psy.connect("postgresql://pierresimplon974:hIf3kgq7rJiQ@ep-square-salad-072972.eu-central-1.aws.neon.tech/neondb?options=endpoint%3Dep-square-salad-072972")
    connection.set_session(autocommit=True)
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM models WHERE model_name = %s", (model_name,))
    id = cursor.fetchone()
    return id



