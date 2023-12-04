import pandas as pd
from pathlib import Path
from datetime import datetime
import arxiv
from arxiv_util import *
import schedule
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import time
from repository import *


query_keywords = [
    # "\"image segmentation\"",
    # "\"self-supervised learning\"",
    # "\"representation learning\"",
    # "\"image generation\"",
    # "\"object detection\"",
    # "\"transfer learning\"",
    # "\"transformers\"",
    # "\"adversarial training\"",
    # "\"generative adversarial networks\"",
    # "\"model compression\"",
    # "\"few-shot learning\"",
    # "\"natural language processing\"",
    # "\"graph neural networks\"",
    # "\"colorization\"",
    # "\"depth estimation\"",
    # "\"point cloud\"",
    # "\"structured data\"",
    # "\"optical flow\"",
    # "\"reinforcement learning\"",
    # "\"super resolution\"",
    # "\"attention mechanisms\"",
    # "\"tabular data\"",
    # "\"unsupervised learning\"",
    # "\"semi-supervised learning\"",
    # "\"explainable AI\"",
    # "\"radiance field\"",
    "\"decision tree\"",
    "\"time series analysis\"",
    "\"molecule generation\"",
    "\"large language models\"",
    "\"LLMs\"",
    "\"language models\"",
    "\"image classification\"",
    "\"document image classification\"",
    "\"encoder-decoder\"",
    "\"multimodal learning\"",
    "\"multimodal deep learning\"",
    "\"speech recognition\"",
    "\"generative models\"",
    "\"anomaly detection\"",
    "\"recommender systems\"",
    "\"robotics\"",
    "\"knowledge graphs\"",
    "\"cross-modal learning\"",
    "\"attention mechanisms\"",
    "\"unsupervised translation\"",
    "\"machine translation\"",
    "\"dialogue systems\"",
    "\"sentiment analysis\"",
    "\"question answering\"",
    "\"text summarization\"",
    "\"sequential modeling\"",
    "\"neurosymbolic AI\"",
    "\"fairness in AI\"",
    "\"transferable skills\"",
    "\"data augmentation\"",
    "\"neural architecture search\"",
    "\"active learning\"",
    "\"automated machine learning\"",
    "\"meta-learning\"",
    "\"domain adaptation\"",
    "\"time series forecasting\"",
    "\"weakly supervised learning\"",
    "\"self-supervised vision\"",
    "\"visual reasoning\"",
    "\"knowledge distillation\"",
    "\"hyperparameter optimization\"",
    "\"cross-validation\"",
    "\"explainable reinforcement learning\"",
    "\"meta-reinforcement learning\"",
    "\"generative models in NLP\"",
    "\"knowledge representation and reasoning\"",
    "\"zero-shot learning\"",
    "\"self-attention mechanisms\"",
    "\"ensemble learning\"",
    "\"online learning\"",
    "\"cognitive computing\"",
    "\"self-driving cars\"",
    "\"emerging AI trends\"",
    "\"Attention is all you need\"",
    "\"GPT\"",
    "\"BERT\"",
    "\"Transformers\"",
    "\"yolo\"",
    "\"speech recognisation\"",
    "\"LSTM\"",
    "\"GRU\"",
    "\"BERT - Bidirectinal Encoder Representation of Transformers\"",
    "\"Large Language Model\" ",
    "\"Stabel diffusion\"",
    "\"Attention is all you need\"",
    "\"Encoder-Decoder\"",
     "\"Paper Recommendatin systems\"",
     "\" Latent Dirichlet Allocation (LDA)\"",
     "\"Transformers\"",
     "\"Generative Pre-trained Transformer\"",
]

PATH_DATA_BASE = Path.cwd() / "data"

# arxiv_data.to_csv(PATH_DATA_BASE / 'data.csv', index=False)


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

    for keyword in query_keywords:
        fetch_data = fetch_arxiv(client, keyword, last_published_date, current_date)
        if len(fetch_data) >0:

            # insert data into database
            insert_research(engine, fetch_data)

# Schedule the batch process to run every week on Sunday at a specific time
schedule.every().sunday.at("12:00").do(run_batch_process)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)

