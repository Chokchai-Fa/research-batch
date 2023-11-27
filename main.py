import pandas as pd
from pathlib import Path
from datetime import datetime
import arxiv
from arxiv_util import *
import schedule
import time
from sqlalchemy import create_engine

from repository import *


PATH_DATA_BASE = Path.cwd() / "data"



# arxiv_data.to_csv(PATH_DATA_BASE / 'data.csv', index=False)



def test_insert():
    username = 'postgres'
    password = 'postgres'
    host = 'localhost:5432'
    database = 'postgres'

    engine = create_engine(f'postgresql://{username}:{password}@{host}/{database}')

    client = arxiv.Client(num_retries=20, page_size=500)

    

    # quert data to check last published date
    last_published_date = query_last_published_date(engine)

    current_date = datetime.now().strftime("%Y-%m-%d")


    fetch_data = fetch_arxiv(client, last_published_date, current_date)
    if len(fetch_data) >0:
        print(len(fetch_data))

        # insert data into database
        insert_research(engine, fetch_data)

# test_insert()

def run_batch_process():
   

    username = 'postgres'
    password = 'postgres'
    host = 'localhost:5432'
    database = 'postgres'

    engine = create_engine(f'postgresql://{username}:{password}@{host}/{database}')

    print("start batch process")

    client = arxiv.Client(num_retries=20, page_size=500)

    

    # quert data to check last published date
    last_published_date = query_last_published_date(engine)

    current_date = datetime.now().strftime("%Y-%m-%d")


    fetch_data = fetch_arxiv(client, last_published_date, current_date)
    if len(fetch_data) >0:
        print(len(fetch_data))

        # insert data into database
        insert_research(engine, fetch_data)



# Schedule the batch process to run every week on Sunday at a specific time
schedule.every().tuesday.at("00:48").do(run_batch_process)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)



