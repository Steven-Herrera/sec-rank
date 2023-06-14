from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import json

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.airbyte.operators.airbyte import AirbyteTriggerSyncOperator

with DAG(dag_id='trigger_sec_scrape_airbyte_job',
        default_args={'owner':'admin'},
        schedule_intervals='@daily',
        start_date=days_ago(1)
        ) as dag:

df = pd.read_csv('./Labeled_Data/ExcelParser_Outputs/sec_pages.csv')
response = requests.get('https://www.sec.gov/comments/s7-09-22/s70922.htm')
soup = BeautifulSoup(response, 'html.parser')

def test_scrape(df):
    response = requests.get(df['URL'][0])
    soup = BeautifulSoup(response, 'html.parser')
    assert isinstance(soup.get_text(), str)