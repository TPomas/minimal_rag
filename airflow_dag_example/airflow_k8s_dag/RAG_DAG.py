from airflow import DAG

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

with DAG(
  dag_id="RAG_DAG",
  description="Description of RAG DAG",
  schedule_interval=None) as dag:

  t1 = KubernetesJobOperator(
    name="parse-docs",
    task_id="parse-docs",
    job_template_file="<PATH_TO>/parsing-job.yaml" # /opt/airflow/dags if K8s job files are next to DAG files
  )

  t2 = KubernetesJobOperator(
    name="embed-chunks",
    task_id="embed-chunks",
    job_template_file="<PATH_TO>/embedding-job.yaml" # /opt/airflow/dags if K8s job files are next to DAG files
  )
 
  t1 >> t2
