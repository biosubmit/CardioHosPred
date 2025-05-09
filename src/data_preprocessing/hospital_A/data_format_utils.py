"""Format columns in merged data
1. Ensure time-related columns are datetime type
2. Ensure other columns are str type
3. Format and process range values
4. Format and process diagnosis, ensure diagnosis information fields are space-separated
"""
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm library for progress bar display
from data_helper import check_dtype_row
from data_normalcheck_utils import normalize_data_mark

def format_score_columns(data_mark):
    """Format columns in merged data
    1. Ensure examination result columns are float type
    2. Ensure examination result columns are space-separated
    3. Split range column into upper and lower limits
    """
    # Convert examination results to float type
    data_mark['检查结果'] = pd.to_numeric(data_mark['检查结果'], errors='coerce')  # 'examination result'
    
    # Create new columns to store upper and lower limits, default values are negative and positive infinity
    data_mark['下限'] = float('-inf')  # 'lower limit'
    data_mark['上限'] = float('inf')  # 'upper limit'
    
    # First clean the range column - copy to a new column to avoid modifying original data
    data_mark['范围清理'] = data_mark['范围'].astype(str).str.replace('[', '').str.replace(']', '').str.strip()  # 'range cleaned'
    
    # Process ranges containing commas (e.g., "3.5, 7.2")
    comma_mask = data_mark['范围清理'].str.contains(',', na=False)
    if comma_mask.any():
        # Separate lower limit
        data_mark.loc[comma_mask, '下限_temp'] = data_mark.loc[comma_mask, '范围清理'].str.split(',').str[0].str.strip()  # 'lower limit temp'
        # Separate upper limit
        data_mark.loc[comma_mask, '上限_temp'] = data_mark.loc[comma_mask, '范围清理'].str.split(',').str[1].str.strip()  # 'upper limit temp'
        
        # Convert to float
        data_mark.loc[comma_mask, '下限'] = pd.to_numeric(data_mark.loc[comma_mask, '下限_temp'], errors='coerce')
        data_mark.loc[comma_mask, '上限'] = pd.to_numeric(data_mark.loc[comma_mask, '上限_temp'], errors='coerce')
    
    # Process ranges with only upper limit (e.g., "<5.0")
    less_mask = data_mark['范围清理'].str.contains('<', na=False) & ~comma_mask
    if less_mask.any():
        data_mark.loc[less_mask, '上限'] = pd.to_numeric(
            data_mark.loc[less_mask, '范围清理'].str.replace('<', '').str.strip(), 
            errors='coerce'
        )
    
    # Process ranges with only lower limit (e.g., ">3.0")
    greater_mask = data_mark['范围清理'].str.contains('>', na=False) & ~comma_mask
    if greater_mask.any():
        data_mark.loc[greater_mask, '下限'] = pd.to_numeric(
            data_mark.loc[greater_mask, '范围清理'].str.replace('>', '').str.strip(),
            errors='coerce'
        )
    
    # Process single values (no <, >, or ,)
    single_value_mask = ~comma_mask & ~less_mask & ~greater_mask
    # Try to convert to numeric
    data_mark.loc[single_value_mask, '值_temp'] = pd.to_numeric(data_mark.loc[single_value_mask, '范围清理'], errors='coerce')  # 'value temp'
    # Set upper and lower limits for successfully converted rows
    valid_value_mask = single_value_mask & data_mark['值_temp'].notna()
    if valid_value_mask.any():
        data_mark.loc[valid_value_mask, '下限'] = data_mark.loc[valid_value_mask, '值_temp']
        data_mark.loc[valid_value_mask, '上限'] = data_mark.loc[valid_value_mask, '值_temp']
    
    # Delete temporary columns
    data_mark = data_mark.drop(columns=['范围清理', '下限_temp', '上限_temp', '值_temp'], errors='ignore')
    
    return data_mark

def remove_invalid_rows(data_mark):
    """Remove invalid rows
    1. If both upper and lower limits are inf (positive or negative infinity), remove
    2. If both upper and lower limits are NaN, remove
    3. If one is inf/NaN and the other is not, keep
    4. If neither is inf/NaN, keep
    """
    # Create masks for upper and lower limits that are inf
    upper_is_inf = data_mark['上限'].isin([np.inf, -np.inf])
    lower_is_inf = data_mark['下限'].isin([np.inf, -np.inf])
    
    # Create masks for upper and lower limits that are NaN
    upper_is_nan = data_mark['上限'].isna()
    lower_is_nan = data_mark['下限'].isna()
    
    # Remove rows where both limits are inf or both are NaN
    valid_rows = ~((upper_is_inf & lower_is_inf) | (upper_is_nan & lower_is_nan))
    
    # Return valid rows
    return data_mark[valid_rows]

def format_bmi(data_mark):
    '''
    Format BMI column
    1. If BMI column is numeric, keep it
    2. If BMI column is string, remove decimal point
    
    '''
    # Calculate BMI from height and weight
    data_mark['体重'] = pd.to_numeric(data_mark['体重'], errors='coerce')  # 'weight'
    data_mark['身高'] = pd.to_numeric(data_mark['身高'], errors='coerce')  # 'height'
    data_mark['bmi'] = data_mark['体重'] / (data_mark['身高'] * data_mark['身高'])

    # Delete original columns
    data_mark = data_mark.drop(columns=['体重', '身高'], errors='ignore')

    return data_mark

def format_blood_pressure(data_mark):
    '''
    Format blood pressure column e.g. 130/80mmHg
    Normal upper limit for high pressure is 140, normal upper limit for low pressure is 90
    Normal lower limit for high pressure is 90, normal lower limit for low pressure is 60
    
    '''
    # Remove units
    data_mark['血压'] = data_mark['血压'].str.replace('mmHg', '')  # 'blood pressure'

    # Split into intervals, replace with original values
    upper_value = data_mark['血压'].str.split('/').str[0]
    lower_value = data_mark['血压'].str.split('/').str[1]

    data_mark['高压'] = upper_value  # 'systolic pressure'
    data_mark['低压'] = lower_value  # 'diastolic pressure'

    # Delete original column
    data_mark = data_mark.drop(columns=['血压'], errors='ignore')

    return data_mark

def format_heart_rate(data_mark):
    '''
    Format heart rate column
    1. If heart rate column is numeric, keep it
    2. If heart rate column is string, remove decimal point
    
    '''
    # Ensure it's in float format
    data_mark['心率'] = pd.to_numeric(data_mark['心率'], errors='coerce')  # 'heart rate'
    # Fill non-float values with nan
    data_mark['心率'] = data_mark['心率'].fillna(np.nan)

    return data_mark

def format_temperature(data_mark):
    '''
    Format body temperature column
    1. If temperature column is numeric, keep it
    2. If temperature column is string, remove decimal point
    '''
    # Ensure it's in float format
    data_mark['体温'] = pd.to_numeric(data_mark['体温'], errors='coerce')  # 'body temperature'
    # Fill non-float values with nan
    data_mark['体温'] = data_mark['体温'].fillna(np.nan)

    return data_mark

def format_columns(data_mark):
    """Format columns in merged data
    1. Ensure time-related columns are datetime type
    2. Ensure other columns are str type
    3. Format and process range values
    4. Format and process diagnosis, ensure diagnosis information fields are space-separated
    """
    # 1. Ensure time-related columns are datetime type

    # Format examination results and range columns
    formatted_data_mark = format_score_columns(data_mark)

    # Format BMI
    formatted_data_mark = format_bmi(formatted_data_mark)

    # Format blood pressure
    formatted_data_mark = format_blood_pressure(formatted_data_mark)    

    # Format heart rate
    formatted_data_mark = format_heart_rate(formatted_data_mark)

    # Format body temperature
    formatted_data_mark = format_temperature(formatted_data_mark)   
    
    # Remove invalid rows
    formatted_data_mark = remove_invalid_rows(formatted_data_mark)

    # Ensure examination results and range limits are the correct type
    check_dtype_row(formatted_data_mark['检查结果'], [float])  # 'examination result'
    check_dtype_row(formatted_data_mark['上限'], [float])  # 'upper limit'
    check_dtype_row(formatted_data_mark['下限'], [float])  # 'lower limit'

    return formatted_data_mark

if __name__ == "__main__":
    # Read merged data
    # #combined_patient_info = pd.read_csv('./data/hospital_A/tmp_preprocessed_data/combined_patient_info.csv')
    data_mark = pd.read_csv('./data/hospital_A/tmp_preprocessed_data/old_data_mark.csv')

    # Format data
    formatted_data_mark = format_columns(data_mark)
    normalized_data_mark = normalize_data_mark(formatted_data_mark)

    print(normalized_data_mark['归一化检查结果'])  # 'normalized examination result'


