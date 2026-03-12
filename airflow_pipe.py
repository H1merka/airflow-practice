import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
from pathlib import Path
import os

# Импорт функции train (из train_model.py)
from train_model_new import train

RAW_URL = "https://raw.githubusercontent.com/sumit0072/Car-Price-Prediction-Project/main/car%20data.csv"
LOCAL_RAW = "cars.csv"
CLEAR_CSV = "df_clear.csv"

def download_data():
    """Скачиваем CSV (копия CarDekho) и сохраняем как cars.csv"""
    print("Downloading dataset from:", RAW_URL)
    resp = requests.get(RAW_URL, timeout=30)
    resp.raise_for_status()
    with open(LOCAL_RAW, "wb") as f:
        f.write(resp.content)
    df = pd.read_csv(LOCAL_RAW)
    print("Downloaded:", df.shape, "columns:", list(df.columns))
    return True

def clear_data():
    """
    Чистим и подготавливаем CarDekho-датасет.
    Преобразуем категориальные признаки в числовые (one-hot),
    переименуем целевую колонку в 'Price' и сохраним df_clear.csv
    """
    df = pd.read_csv(LOCAL_RAW)

    # Быстрая предобработка / выравнивание имён колонок в разных копиях
    # возможные имена: 'Selling_Price' / 'selling_price', 'Kms_Driven' / 'kms_driven' и т.д.
    col_map = {}
    if 'Selling_Price' in df.columns:
        col_map['Selling_Price'] = 'Price'
    elif 'selling_price' in df.columns:
        col_map['selling_price'] = 'Price'
    if 'Kms_Driven' in df.columns:
        col_map['Kms_Driven'] = 'Distance'
    if 'kms_driven' in df.columns:
        col_map['kms_driven'] = 'Distance'
    if 'Present_Price' in df.columns:
        col_map['Present_Price'] = 'Present_Price'
    if 'Car_Name' in df.columns:
        col_map['Car_Name'] = 'Car_Name'
    df = df.rename(columns=col_map)

    # Убедимся, что целевая есть
    if 'Price' not in df.columns:
        raise ValueError("Target column 'Price' not found after renaming. Available cols: " + ", ".join(df.columns))

    # Оставим только полезные колонки (какую структуру ожидаем)
    keep_cols = []
    # numeric candidates
    for c in ['Year', 'Distance', 'Present_Price', 'Owner']:
        if c in df.columns:
            keep_cols.append(c)
    # categorical candidates
    cat_cols = []
    for c in ['Fuel_Type', 'Seller_Type', 'Transmission']:
        if c in df.columns:
            cat_cols.append(c)

    # Пример: если есть Car_Name, удалим (слишком большая кардинальность) — можно изменить при необходимости
    if 'Car_Name' in df.columns:
        df = df.drop(columns=['Car_Name'])

    # Оставляем target
    keep_cols = keep_cols  # numeric
    print("Numeric cols detected:", keep_cols)
    print("Categorical cols detected:", cat_cols)

    # Очистка: удаляем строки с NaN в важных колонках
    required = keep_cols + cat_cols + ['Price']
    df = df.dropna(subset=required).reset_index(drop=True)

    # Приведение типов: Distance / Kms_Driven -> numeric
    if 'Distance' in df.columns:
        df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    if 'Present_Price' in df.columns:
        df['Present_Price'] = pd.to_numeric(df['Present_Price'], errors='coerce')
    if 'Owner' in df.columns:
        df['Owner'] = pd.to_numeric(df['Owner'], errors='coerce')
    df = df.dropna(subset=required).reset_index(drop=True)

    # Простейшие фильтры выбросов (настрой на свой датасет при необходимости)
    if 'Distance' in df.columns:
        df = df[(df.Distance >= 0) & (df.Distance < 1e7)]  # отфильтруем очевидные мусорные значения
    if 'Year' in df.columns:
        df = df[(df.Year >= 1950) & (df.Year <= datetime.now().year + 1)]

    df = df.reset_index(drop=True)
    print("After cleaning:", df.shape)

    # One-Hot encode категориальные колонки
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe_arr = ohe.fit_transform(df[cat_cols])
        ohe_cols = list(ohe.get_feature_names_out(cat_cols))
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)
        df_final = pd.concat([df[keep_cols], df_ohe, df['Price']], axis=1)
    else:
        df_final = pd.concat([df[keep_cols], df['Price']], axis=1)

    # Сохраняем подготовленный csv (все числовые колонки + target 'Price')
    df_final.to_csv(CLEAR_CSV, index=False)
    print("Saved cleaned file:", CLEAR_CSV, "shape:", df_final.shape)
    return True

# DAG
dag_cars = DAG(
    dag_id="train_pipe_cardekho",
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(python_callable=download_data, task_id="download_cardekho", dag=dag_cars)
clear_task = PythonOperator(python_callable=clear_data, task_id="clear_cardekho", dag=dag_cars)
train_task = PythonOperator(python_callable=train, task_id="train_cardekho", dag=dag_cars)

download_task >> clear_task >> train_task
