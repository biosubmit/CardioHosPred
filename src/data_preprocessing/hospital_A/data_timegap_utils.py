import pandas as pd
from tqdm import tqdm
def calculate_time_gap(data_mark: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the time interval in the data mark
    """
    # Calculate time interval
    data_mark['time_gap'] = data_mark['入院日期'] - data_mark['日期']  # 入院日期: Admission date, 日期: Date
    return data_mark



    
def merge_datamark_and_patient_info(
    datamark_df: pd.DataFrame,
    patient_info_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Based on checkup_df (main input), find matching future (or recent) admission information in patient_info_df,
    and merge corresponding information (gender, address, admission date, discharge date, length of stay, department, age, etc.) into checkup_df.
    
    Processing logic:
      1. Group patient_info_df by medical record number
      2. For each record in checkup_df (inpatient number, date):
         - Find multiple records in patient_info with corresponding inpatient number
         - Filter out admission information where [admission date >= current examination date]
         - If exists, select the one with minimum distance (closest future admission)
         - Merge relevant information into the result
      3. Unmatched records remain unchanged or can be optionally deleted
    Return checkup_df with newly added information columns.
    """
    # Add new columns
    new_columns = ['性别', '年龄', '上次入院时间', '上次出院时间',
                   '上次住院天数', '住院次数标记', '上次入院科室', '上次出院科室','time_gap']
    # 性别: Gender, 年龄: Age, 上次入院时间: Last admission time, 上次出院时间: Last discharge time
    # 上次住院天数: Last length of stay, 住院次数标记: Admission count marker, 上次入院科室: Last admission department
    # 上次出院科室: Last discharge department
    
    for col in new_columns:
        datamark_df[col] = None

    # Group by medical record number for easy lookup
    patient_info_grouped = patient_info_df.groupby('病案号')  # 病案号: Medical record number

    # Temporary storage for merge results, to update checkup_df all at once
    results = []

    # Iterate through the examination table
    for row in tqdm(datamark_df.itertuples(), total=datamark_df.shape[0]):
        patient_id = row.病案号  # 病案号: Medical record number
        checkup_date = row.检查时间  # 检查时间: Examination time
        
        # If the patient_id exists in patient_info, get all admission records
        if patient_id in patient_info_grouped.groups:
            patient_admissions = patient_info_grouped.get_group(patient_id).copy()
            # Sort + mark admission count
            patient_admissions = patient_admissions.sort_values('入院时间').reset_index(drop=True)  # 入院时间: Admission time
            patient_admissions['入院次数'] = range(1, len(patient_admissions) + 1)  # 入院次数: Admission count
            
            patient_admissions['上次住院天数'] = patient_admissions['上次出院时间'] - patient_admissions['上次入院时间']  # 上次住院天数: Last length of stay, 上次出院时间: Last discharge time, 上次入院时间: Last admission time
            patient_admissions['上次住院天数'] = patient_admissions['上次住院天数'].fillna(0)  # 上次住院天数: Last length of stay

            # Only take "future or current" admission records
            future_admissions = patient_admissions[patient_admissions['入院时间'] >= checkup_date]  # 入院时间: Admission time
            if future_admissions.empty:
                # No future admission records found, can choose to continue / or take the most recent past
                continue

            # Find the one with minimum distance (closest admission)
            future_admissions['日期差'] = (future_admissions['入院时间'] - checkup_date).abs()  # 日期差: Date difference, 入院时间: Admission time
            closest_admission = future_admissions.loc[future_admissions['日期差'].idxmin()]  # 日期差: Date difference

            # Extract information for merging
            gender = closest_admission['性别']  # 性别: Gender
            address = closest_admission.get('省市级地址', None)  # 省市级地址: Province-city level address
            checkout_date = closest_admission['上次出院时间']  # 上次出院时间: Last discharge time
            days_in_hospital = closest_admission['上次住院天数']  # 上次住院天数: Last length of stay
            admission_dep = closest_admission['上次入院科室']  # 上次入院科室: Last admission department
            out_dep = closest_admission['上次出院科室']  # 上次出院科室: Last discharge department
            age_at_admission = closest_admission['年龄']  # 年龄: Age
            admission_date = closest_admission['上次入院时间']  # 上次入院时间: Last admission time
            time_label = closest_admission['入院次数']  # 入院次数: Admission count

            # Calculate time difference/age, etc.
            year_diff = checkup_date.year - admission_date.year
            age_at_checkup = age_at_admission + year_diff if pd.notnull(age_at_admission) else None
            gap = (admission_date - checkup_date).days

            # Construct merge information dict
            merge_dict = {
                '性别': gender,  # 性别: Gender
                '年龄': age_at_checkup,  # 年龄: Age
                '上次入院时间': admission_date,  # 上次入院时间: Last admission time
                '上次出院时间': checkout_date,  # 上次出院时间: Last discharge time
                '上次住院天数': days_in_hospital,  # 上次住院天数: Last length of stay
                '住院次数标记': time_label,  # 住院次数标记: Admission count marker
                '上次入院科室': admission_dep,  # 上次入院科室: Last admission department
                '上次出院科室': out_dep,  # 上次出院科室: Last discharge department
                'time_gap': gap
            }

            # Store (checkup_df row index, merge information)
            results.append((row.Index, merge_dict))
    
    # Construct temporary df for one-time update
    results_df = pd.DataFrame([res[1] for res in results], index=[res[0] for res in results])
    # Update row by row
    datamark_df.update(results_df)

    # Delete rows with empty gender (if this is your business requirement)
    datamark_df = datamark_df.dropna(subset=['性别'])  # 性别: Gender

    return datamark_df

def merge_datamark_and_patient_info_v2(
    datamark_df: pd.DataFrame,
    patient_info_df: pd.DataFrame
) -> pd.DataFrame:
    """
    An alternative approach to merge checkup data with patient information
    """
    

    merged = pd.merge(
        datamark_df,
        patient_info_df,
        on='病案号',  # 病案号: Medical record number
        how='left'
    )

    # 2. Only keep combinations where admission time >= examination time
    merged = merged[merged['入院时间'] >= merged['检查时间']]  # 入院时间: Admission time, 检查时间: Examination time

    # 3. Calculate days difference between "admission time - examination time" (smaller is closer)
    merged['时间差'] = (merged['入院时间'] - merged['检查时间']).dt.days  # 时间差: Time difference, 入院时间: Admission time, 检查时间: Examination time

    # 4. For each examination record (medical record number + examination time + item name), select the closest admission
    # ⚠️ Note: if in your data "same medical record number + time + item name" can be duplicated, you can also use only "medical record number + examination time"
    merged_closest = (
        merged.sort_values(by=['病案号', '检查时间', '时间差'])  # 病案号: Medical record number, 检查时间: Examination time, 时间差: Time difference
        .drop_duplicates(subset=['病案号', '检查时间'])  # Keep the closest admission for each examination item  # 病案号: Medical record number, 检查时间: Examination time
    )
    

    return merged_closest



# 5. Now merged_closest is the final result