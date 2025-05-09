import pandas as pd
import numpy as np
import os

def load_patient_info_old(file_path):
    """
    Load medical record front page file and select specified feature columns
    
    Selected feature columns:
    - 病案号 (Medical record number)
    - 性别 (Gender)
    - 出生日期 (Birth date)
    - 地址.1 (Address)
    - 入院科室 (Admission department)
    - 出院科室 (Discharge department)
    - 主要诊断 (Main diagnosis)
    - 入院时间 (Admission time)
    - 出院时间 (Discharge time)
    """
    try:
        df = pd.read_csv(file_path)
        selected_columns = [
            '病案号', '性别', '出生日期', '年龄',
            '入院科室', '出院科室', '主要诊断', '入院时间', '出院时间'
        ]
        return df[selected_columns]
    except Exception as e:
        print(f"Error loading medical record front page file: {str(e)}")
        return None

def load_data_mark_old(file_path):
    """
    Load data mark file and select specified feature columns
    
    Selected feature columns:
    - 住院号 (Inpatient number)
    - 日期 (Date)
    - 身高 (Height)
    - 体重 (Weight)
    - 项目名称 (Item name)
    - 范围 (Range)
    - 检查结果 (Examination result)
    """
    try:
        df = pd.read_csv(file_path)
        selected_columns = [
            '住院号', '日期', '身高', '体重', '血压', '心率', '体温', '呼吸',
            '项目名称', '范围', '检查结果'
        ]
        return df[selected_columns]
    except Exception as e:
        print(f"Error loading data mark file: {str(e)}")
        return None

def change_column_name(df, old_name, new_name):
    """
    Change column names in data mark
    """
    df.rename(columns={old_name: new_name}, inplace=True)
    return df


def preprocess_old_data(medical_record_path, data_mark_path):
    """
    Preprocess old data
    
    Args:
        medical_record_path: Path to medical record front page file
        data_mark_path: Path to data mark file
    
    Returns:
        tuple: (Processed patient information data, processed data mark data)
    """
    # Load data
    patient_info = load_patient_info_old(medical_record_path)
    data_mark = load_data_mark_old(data_mark_path)

    # Change column names
    data_mark = change_column_name(data_mark, '住院号', '病案号')  # 住院号: Inpatient number, 病案号: Medical record number

    data_mark = change_column_name(data_mark, '日期', '检查时间')  # 日期: Date, 检查时间: Examination time
    
    return patient_info, data_mark

if __name__ == "__main__":
    # Set input output paths
    medical_record_path = "./data/hospital_A/病案首页信息提取.csv"  # 病案首页信息提取: Medical record front page information extraction
    data_mark_path = "./data/hospital_A/data_mark.csv"
    output_dir = "./data/hospital_A/tmp_preprocessed_data"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process data
    patient_info, data_mark = preprocess_old_data(medical_record_path, data_mark_path)
    
    # Save processed data
    if patient_info is not None and data_mark is not None:
        patient_info.to_csv(f"{output_dir}/processed_medical_records.csv", index=False)
        data_mark.to_csv(f"{output_dir}/processed_data_mark.csv", index=False)
        print("First stage data preprocessing completed!")
        print(f"Processed files saved to: {output_dir}")
    else:
        print("Error occurred during data preprocessing")