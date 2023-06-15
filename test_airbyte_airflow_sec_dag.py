"""This is the test DAG for using BeautifulSoup to scrape commenter information
   from various regulatory rules proposed by the SEC. The URL for rules are stored 
   in a CSV file which are fed to the scraper. After scraping every rule, the scraper
   stores the data in a JSON file which triggers an Airbyte connection. Airbyte migrates
   the data from the JSON file into a PostgreSQL database."""

# standard imports
import json

# Airflow imports
#from airflow.utils.dates import days_ago # deprecated
from pendulum import today
from airflow import DAG
from airflow.providers.airbyte.operators.airbyte import AirbyteTriggerSyncOperator
from airflow.models import DagBag

# third party imports
import pandas as pd
# my custom library import
from webscraper import sec_webscraper

def test_dags_load_with_no_errors():
    """Testing DAG with pytest"""
    dag_bag = DagBag(include_examples=False)
    dag_bag.process_file('test_airbyte_airflow_sec_dag.py')
    assert len(dag_bag.import_errors) == 0

with DAG(dag_id='test_airbyte_airflow_sec_dag',
        default_args={'owner':'admin'},
        schedule='@daily',
        start_date=today('UTC').add(days=-1) #days_ago(1)
        ) as dag:

    # sec_pages.csv contains proposed rules we wish to scrape
    rules_df = pd.read_csv('./Labeled_Data/ExcelParser_Outputs/sec_pages.csv')

    # For each rule we want:
    #   1. The name of the commenter
    #   2. The date they commented
    #   3. The URL that leads to their comment
    #   4. The comment itself

    json_data = {}

    for i in range(len(rules_df)):
        # scrape proposed rule
        rule_url = rules_df['URL'][i]

        # lst to store individual rules
        json_rule_data = []
        rule_name = rules_df['Name'][i]

        scraper = sec_webscraper(rule_url)
        commenter_names, commenter_links, commenter_dates = scraper.rule_comment_info_retriever() # pylint: disable=line-too-long

        # ensure same number of names, urls, and dates
        if len(commenter_names) == len(commenter_links) == len(commenter_dates):
            for j in range(len(commenter_names)): # pylint: disable=consider-using-enumerate
                commenter_data = {"Name": commenter_names[j],
                                "Date": commenter_dates[j],
                                "URL": commenter_links[j]}
                json_rule_data.append(commenter_data)

            json_data[rule_name] = json_rule_data
        else:
            raise TypeError(f"Data scraped from {rule_name} is not rectangular.")

    # test file location
    # //wsl.localhost/Ubuntu/tmp/airbyte_local/sec_webscraper/sec_webscraper.json
    json_filepath = "sec_webscraper.json" # pylint: disable=invalid-name
    with open(json_filepath, 'w', encoding='utf-8') as json_fp: # pylint: disable=line-too-long
        # scraped data to json
        json.dump(json_data, json_fp, indent=4)

        # json to postgres
        airbyte_airflow_sec = AirbyteTriggerSyncOperator(
        task_id='airbyte_airflow_sec',
        airbyte_conn_id='airbyte_sec_scraper_connection',
        # this can be found in the URL for the connection page in Airbyte
        connection_id='http://localhost:8000/workspaces/1c143870-88cf-43cf-a90a-7f60e2c003d4/connections/d84af434-8e5f-4cfe-bc76-578a06d3821e', # pylint: disable=line-too-long
        asynchronous=False,
        timeout=3600,
        wait_seconds=3
        )
