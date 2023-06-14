from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import json

from webscraper import sec_webscraper

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.airbyte.operators.airbyte import AirbyteTriggerSyncOperator

with DAG(dag_id='trigger_sec_scrape_airbyte_job',
        default_args={'owner':'admin'},
        schedule_intervals='@daily',
        start_date=days_ago(1)
        ) as dag:

# sec_pages.csv contains proposed rules we wish to scrape
df = pd.read_csv('./Labeled_Data/ExcelParser_Outputs/sec_pages.csv')

# For each rule we want:
#   1. The name of the commenter
#   2. The date they commented
#   3. The URL that leads to their comment
#   4. The comment itself
json_data = []

for i in range(len(df)):
    URL = df['URL'][i]
    scraper = sec_webscraper(URL)
    commenter_names, commenter_links, commenter_dates = scraper.rule_comment_info_retriever()

    if len(commenter_names) == len(commenter_links) == len(commenter_dates):
        for j in range(len(commenter_names)):
            comment_data = dict()
            comment_data['Name'] = commenter_names[j]
            comment_data['Date'] = commenter_dates[j]
            comment_data['URL'] = commenter_links[j]
            json_data.append(comment_data)

