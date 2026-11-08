FROM apache/airflow:2.10.5-python3.12

USER airflow 

COPY  requirements.txt /requirements.txt

RUN pip install --no-cache-dir "apache-airflow==2.10.5" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt" -r /requirements.txt 

