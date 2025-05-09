from typing import Optional
import pandas as pd
from loguru import logger
from datetime import datetime
import numpy as np
"""
selected_columns = ['medical_record_number', 'hospitalization_count', 'payment_category', 
                    'gender', 'age',
                    'main_operation_name', 'current_address',  
                    'admission_time', 'admission_department(front_page)',
                    'discharge_department(front_page)', 'discharge_time',
                    'outpatient_diagnosis','discharge_main_diagnosis_name1', 
                    'discharge_main_diagnosis_admission_condition1', 
                    'pathological_diagnosis','main_operation_date']

"""


def align_diagnosis_next_day(df):
    '''
    Sort the dataframe by medical record number and admission time.
    This function takes the previous diagnosis information, admission department, discharge department,
    and places them in the current diagnosis information, admission department, discharge department fields.
    If it's the first admission, then the history diagnosis, admission department, discharge department will be empty.
    
    Args:
        df (pd.DataFrame): Input dataframe containing hospital records
        
    Returns:
        pd.DataFrame: Processed dataframe with aligned historical information
    '''
    # Ensure admission time and discharge time are in datetime format
    df['入院时间'] = pd.to_datetime(df['入院时间'])  # 'admission_time'
    df['出院时间'] = pd.to_datetime(df['出院时间'])  # 'discharge_time'
    
    # Sort by medical record number and admission time
    df = df.sort_values(by=['病案号', '入院时间'])  # 'medical_record_number', 'admission_time'
    
    # Create history diagnosis columns
    df['上次主要手术操作名称'] = df.groupby('病案号')['主要手术操作名称'].shift()  # 'previous_main_operation_name', 'main_operation_name'
    df['上次出院主要诊断名称1'] = df.groupby('病案号')['出院主要诊断名称1'].shift()  # 'previous_discharge_main_diagnosis_name1', 'discharge_main_diagnosis_name1'
    df['上次病理诊断'] = df.groupby('病案号')['病理诊断'].shift()  # 'previous_pathological_diagnosis', 'pathological_diagnosis'


    df['上次出院科别'] = df.groupby('病案号')['出院科别(首页)'].shift()  # 'previous_discharge_department', 'discharge_department(front_page)'
    df['上次入院时间'] = df.groupby('病案号')['入院时间'].shift()  # 'previous_admission_time', 'admission_time'
    df['上次出院时间'] = df.groupby('病案号')['出院时间'].shift()  # 'previous_discharge_time', 'discharge_time'

    # If admission time is first admission, then history diagnosis will be empty
    df['上次主要手术操作名称'] = df['上次主要手术操作名称'].fillna('')
    df['上次出院主要诊断名称1'] = df['上次出院主要诊断名称1'].fillna('')
    df['上次病理诊断'] = df['上次病理诊断'].fillna('')

    df['上次出院科别'] = df['上次出院科别'].fillna('')
    df['上次入院时间'] = df['上次入院时间'].fillna(pd.NaT)
    df['上次出院时间'] = df['上次出院时间'].fillna(pd.NaT)

    # Calculate previous hospitalization duration (in days)
    df['上次住院时长'] = (df['上次出院时间'] - df['上次入院时间']).dt.total_seconds() / (24 * 3600)  # 'previous_hospitalization_duration'
    
    return df

def align_target_day(df):
    '''
    Sort by medical record number and admission time, find next admission time.
    This function puts the next admission time in the current admission time column.
    If it's the last admission, the next admission time will be empty.
    
    Args:
        df (pd.DataFrame): Input dataframe containing hospital records
        
    Returns:
        pd.DataFrame: Processed dataframe with aligned historical information
    '''
    # Ensure admission time is in datetime format
    df['入院时间'] = pd.to_datetime(df['入院时间'])  # 'admission_time'
    
    # Sort by medical record number and admission time
    df = df.sort_values(by=['病案号', '入院时间'])  # 'medical_record_number', 'admission_time'

 
    # Remove duplicates, keep the row with the least missing values
    df = df.drop_duplicates(subset=['病案号', '入院时间'], keep='last')  # 'medical_record_number', 'admission_time'
 

    # Create next admission time column
    df['下一次入院时间'] = df.groupby('病案号')['入院时间'].shift(-1)  # 'next_admission_time', 'admission_time'
    
    # If admission time is the last admission, then next admission time will be empty
    df['下一次入院时间'] = df['下一次入院时间'].fillna(pd.NaT)  # 'next_admission_time'

    # Remove data where next admission time is empty
    df = df[df['下一次入院时间'] != pd.NaT]  # 'next_admission_time'

    df['时间差'] = (df['下一次入院时间'] - df['出院时间']).dt.days  # 'time_difference', 'next_admission_time', 'discharge_time'
    
    return df
    

def align_data(medical_record_data: pd.DataFrame):
    """
    Align data
    """

    # Align diagnosis information
    medical_record_data = align_diagnosis_next_day(medical_record_data)

    # Align target admission time
    medical_record_data = align_target_day(medical_record_data)

    return medical_record_data


