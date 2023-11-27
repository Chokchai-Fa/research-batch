import pandas as pd
import json
from datetime import datetime

def query_last_published_date(engine):
    query = "SELECT MAX(publishes) FROM research"
    last_published_date = pd.read_sql_query(query, con=engine).iloc[0, 0]

    # Convert last_published_date to datetime object (assuming it's stored as a string)
    last_published_date = pd.to_datetime(last_published_date)

    if last_published_date == None:
        return "2023-11-1"
    

    formatted_string = last_published_date.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_string[:10]


def insert_research(engine, df):
    # convert authors into json format for dumps into database
    df['authors'] = df['authors'].apply(lambda authors: json.dumps([str(author) for author in authors]))

    df.to_sql('research', con=engine, if_exists='append', index=False)
    print("insert data successfully for :", len(df), "records")