import pandas as pd
import numpy as np

def normalize_value(x, min_val, max_val):
    """
    Normalize test result x based on normal range:
    - x < min_val => <0
    - min_val <= x <= max_val => [0,1]
    - x > max_val => >1
    
    Returns normalized value to measure how much the test result deviates from the normal range.
    """
    if pd.isnull(x) or pd.isnull(min_val) or pd.isnull(max_val):
        return np.nan
    range_span = max_val - min_val
    if range_span == 0:
        return 0  # Prevent division by zero
    if x > max_val:
        return (x - max_val) / range_span + 1
    elif x < min_val:
        return (x - min_val) / range_span
    else:
        return (x - min_val) / range_span


def normalize_bmi(df):
    # Convert to numeric
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

    # Define normal range
    bmi_min, bmi_max = 18.5, 24.9

    # Use normalize_value function
    df['bmi'] = df['bmi'].apply(lambda x: normalize_value(x, bmi_min, bmi_max))

    return df

def normalize_heart_rate(df):
    # Convert to numeric
    df['心率'] = pd.to_numeric(df['心率'], errors='coerce')  # 心率: Heart rate

    # Define normal range
    heart_rate_min, heart_rate_max = 60, 100

    # Use normalize_value function
    df['心率'] = df['心率'].apply(lambda x: normalize_value(x, heart_rate_min, heart_rate_max))  # 心率: Heart rate

    return df

def normalize_temperature(df):
    # Convert to numeric
    df['体温'] = pd.to_numeric(df['体温'], errors='coerce')  # 体温: Body temperature

    # Define normal range
    temperature_min, temperature_max = 35, 37.5

    # Use normalize_value function 
    df['体温'] = df['体温'].apply(lambda x: normalize_value(x, temperature_min, temperature_max))  # 体温: Body temperature

    return df

def normalize_high_pressure(df):
    # Convert to numeric
    df['高压'] = pd.to_numeric(df['高压'], errors='coerce')  # 高压: Systolic blood pressure

    # Define normal range
    high_pressure_min, high_pressure_max = 90, 140

    # Use normalize_value function 
    df['高压'] = df['高压'].apply(lambda x: normalize_value(x, high_pressure_min, high_pressure_max))  # 高压: Systolic blood pressure

    return df

def normalize_low_pressure(df):
    # Convert to numeric
    df['低压'] = pd.to_numeric(df['低压'], errors='coerce')  # 低压: Diastolic blood pressure

    # Define normal range
    low_pressure_min, low_pressure_max = 60, 90

    # Use normalize_value function 
    df['低压'] = df['低压'].apply(lambda x: normalize_value(x, low_pressure_min, low_pressure_max))  # 低压: Diastolic blood pressure

    return df
    

def normalize_respiratory_rate(df):
    '''
    Respiratory rate normalization
    '''
    # Convert to numeric
    df['呼吸'] = pd.to_numeric(df['呼吸'], errors='coerce')  # 呼吸: Respiratory rate

    # Define normal range
    respiratory_rate_min, respiratory_rate_max = 12, 20

    # Use normalize_value function 
    df['呼吸'] = df['呼吸'].apply(lambda x: normalize_value(x, respiratory_rate_min, respiratory_rate_max))  # 呼吸: Respiratory rate

    return df

def normalize_base_bodyfeature(df):
    '''
    Basic vital signs normalization
    '''
    df=normalize_bmi(df)
    df=normalize_high_pressure(df)
    df=normalize_low_pressure(df)
    df=normalize_heart_rate(df)
    df=normalize_temperature(df)
    df=normalize_respiratory_rate(df)
    return df

def normalize_data_mark(data_mark):
    """
    Normalize examination results in data_mark using upper and lower limits.
    Specific method:
    1. Check if the examination result is within the normal range
    """

    data_mark['归一化检查结果'] = data_mark.apply(  # 归一化检查结果: Normalized examination result
        lambda row: normalize_value(row['检查结果'], row['下限'], row['上限']),  # 检查结果: Examination result, 下限: Lower limit, 上限: Upper limit
        axis=1
    )
    data_mark=normalize_base_bodyfeature(data_mark)

    return data_mark



