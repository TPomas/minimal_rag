from airflow import DAG

from airflow.operators.bash import BashOperator

import os

default_args = {
    "owner": "tp",
    "retries": 0
}

os.environ["WORKING_DIRECTORY"] = # Path to the directory you intend to call the scripts from

# Absolute or relative to WORKING_DIRECTORY
os.environ["INPUT_DIR"] = "docs" 
os.environ["PARSED_DOCS_DIR"] = "output/parsed_docs"
os.environ["VECTORDB_DIR"] = "output/vectordb" 

os.environ["EMBEDDING_MODEL_PATH"] = # Path/URL to the embedding model

with DAG(
  dag_id="test_RAG_steps",
  description="Attempting to run the RAG steps as Airflow DAG",
  schedule_interval=None) as dag:
 
  t0 = BashOperator(
  task_id="parsing",
  bash_command="cd ${WORKING_DIRECTORY} && python3 code/parsing.py --input ${INPUT_DIR} --output ${PARSED_DOCS_DIR} --chunking_strategy by_title --folder_tags")
  
  t1 = BashOperator(
  task_id="embedding",
  bash_command="cd ${WORKING_DIRECTORY} && python3 code/embed.py --data-path ${PARSED_DOCS_DIR} --emb-model-path ${EMBEDDING_MODEL_PATH} --path-to-db ${VECTORDB_DIR}")
  
  t0 >> t1
