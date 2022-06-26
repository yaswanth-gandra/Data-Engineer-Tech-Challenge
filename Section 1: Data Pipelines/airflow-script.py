from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np

def dataProcess_task1(**kwargs):
    #Reading for dataset1
    df = pd.read_csv(kwargs['path']+'/input/dataset1.csv', sep=',')
    #removing the values which doesnot have name
    df = df[(df['name'].notnull()) & (df['name'].str.len() != 0)]

    df['list_full_name'] = df['name'].str.split(' ')
    df['len_list_full_name'] = df['list_full_name'].str.len()
    #extracting first name
    df['first_name'] = np.where(df['len_list_full_name'] == 2, df['list_full_name'].str[0],
                      np.where(df['len_list_full_name']  == 3,
                              np.where(df['list_full_name'].str[0].isin(['Mrs.', 'Mr.','Dr.','Mrs','Ms.']),df['list_full_name'].str[1] ,df['list_full_name'].str[0]),df['list_full_name'].str[1]))
    #extracting last name
    df['last_name'] = np.where(df['len_list_full_name'] == 2, df['list_full_name'].str[1],
                      np.where(df['len_list_full_name']  == 3,
                              np.where(df['list_full_name'].str[0].isin(['Mrs.', 'Mr.','Dr.','Mrs','Ms.']),df['list_full_name'].str[2] ,df['list_full_name'].str[1]),df['list_full_name'].str[2]))
    #removing leading zeros
    df['price'] = df['price'].astype(str).str.lstrip('0').astype(float)
    #calculating above 100 field
    df['above_100'] = np.where(df['price'] > 100, True,False)
    df = df[['first_name','last_name','price','above_100']]
    #writing to the local file
    df.to_csv(kwargs['path']+'/output/dataset1_output.csv', sep=',', encoding='utf-8',index=False)
    
    
def dataProcess_task2(**kwargs):
    #Reading for dataset2
    df = pd.read_csv(kwargs['path']+'/input/dataset2.csv', sep=',')
    #removing the values which doesnot have name
    df = df[(df['name'].notnull()) & (df['name'].str.len() != 0)]

    df['list_full_name'] = df['name'].str.split(' ')
    df['len_list_full_name'] = df['list_full_name'].str.len()
    #extracting first name
    df['first_name'] = np.where(df['len_list_full_name'] == 2, df['list_full_name'].str[0],
                      np.where(df['len_list_full_name']  == 3,
                              np.where(df['list_full_name'].str[0].isin(['Mrs.', 'Mr.','Dr.','Mrs','Ms.']),df['list_full_name'].str[1] ,df['list_full_name'].str[0]),df['list_full_name'].str[1]))
    #extracting last name
    df['last_name'] = np.where(df['len_list_full_name'] == 2, df['list_full_name'].str[1],
                      np.where(df['len_list_full_name']  == 3,
                              np.where(df['list_full_name'].str[0].isin(['Mrs.', 'Mr.','Dr.','Mrs','Ms.']),df['list_full_name'].str[2] ,df['list_full_name'].str[1]),df['list_full_name'].str[2]))
    #removing leading zeros
    df['price'] = df['price'].astype(str).str.lstrip('0').astype(float)
    #calculating above 100 field
    df['above_100'] = np.where(df['price'] > 100, True,False)
    df = df[['first_name','last_name','price','above_100']]
    #writing to the local file
    df.to_csv(kwargs['path']+'/output/dataset2_output.csv', sep=',', encoding='utf-8',index=False)

with DAG('datapipeline_dag', description='Data Pipeline DAG', schedule_interval='30 1 * * *', start_date=datetime(2022, 06, 26), catchup=False) as dag:
#dummy operation for start 
dummy_start_task 	= DummyOperator(task_id='start')
#python operator for procession dataset1
dataProcess_task1	= PythonOperator(task_id='python_task',provide_context=True, python_callable=dataProcess_task1,op_kwargs={'path': '/bin/assignment'})
#python operator for procession dataset2
dataProcess_task2	= PythonOperator(task_id='python_task2',provide_context=True, python_callable=dataProcess_task2,op_kwargs={'path': '/bin/assignment'})
#dummy operation for stop 
dummy_stop_task 	= DummyOperator(task_id='stop')

#operation sequence
dummy_start_task >> dataProcess_task1 >> dataProcess_task2 >> dummy_stop_task