import pandas as pd
import psycopg2 as psy


def insert_model(model_name,model_pickle):
    connection = psy.connect("postgresql://pierresimplon974:hIf3kgq7rJiQ@ep-square-salad-072972.eu-central-1.aws.neon.tech/neondb?options=endpoint%3Dep-square-salad-072972")
    connection.set_session(autocommit=True)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO models(model_name,model_pickle) VALUES (%s, %s)", (model_name,psy.Binary(model_pickle)))



def query_data():
    connection = sqlalchemy.create_engine('postgresql://pierresimplon974:hIf3kgq7rJiQ@ep-square-salad-072972.eu-central-1.aws.neon.tech/neondb?options=endpoint%3Dep-square-salad-072972').connect()
    cursor = connection.cursor()

    # Use pandas to execute the query and assign the result to a DataFrame
    df = pd.read_sql_query("SELECT * FROM your_table_name", connection)

    # Close the cursor and the connection
    cursor.close()
    connection.close()

    # Now you can work with the DataFrame df
    print(df.head())
    df = pd.read_sql_table('elections_data', connection)
    return df



def query_model_id(model_name):
    connection = psy.connect("postgresql://pierresimplon974:hIf3kgq7rJiQ@ep-square-salad-072972.eu-central-1.aws.neon.tech/neondb?options=endpoint%3Dep-square-salad-072972")
    connection.set_session(autocommit=True)
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM models WHERE model_name = %s", (model_name,))
    id = cursor.fetchone()
    return id


