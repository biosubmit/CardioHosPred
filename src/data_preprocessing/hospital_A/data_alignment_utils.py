import pandas as pd

def align_id_format(df, label:str='old'):
    # Convert medical record number column to int first, then to str
    # Check if there are suffixes like _XQq in the medical record number, if yes, remove them using regex to remove everything after _
    df['病案号'] = df['病案号'].astype(str).str.replace(r'_.*', '', regex=True)  # 病案号: Medical record number

    df['病案号'] = df['病案号'].astype(str)  # 病案号: Medical record number
    
    return df

def align_diagnosis_next_day(df):
    '''
    Sort the dataframe by medical record number and admission time.
    Move the previous diagnostic information, admission department, discharge department to the current diagnostic information, admission department, discharge department.
    If it's the first admission, then the historical diagnosis, admission department, discharge department will be empty.
    '''
    # Sort by medical record number and admission time
    df = df.sort_values(by=['病案号', '入院时间'])  # 病案号: Medical record number, 入院时间: Admission time
    
    # Create historical diagnosis columns
    df['上次诊断'] = df.groupby('病案号')['主要诊断'].shift()  # 上次诊断: Previous diagnosis, 主要诊断: Main diagnosis
    df['上次入院科室'] = df.groupby('病案号')['入院科室'].shift()  # 上次入院科室: Previous admission department, 入院科室: Admission department
    df['上次出院科室'] = df.groupby('病案号')['出院科室'].shift()  # 上次出院科室: Previous discharge department, 出院科室: Discharge department
    df['上次入院时间'] = df.groupby('病案号')['入院时间'].shift()  # 上次入院时间: Previous admission time, 入院时间: Admission time
    df['上次出院时间'] = df.groupby('病案号')['出院时间'].shift()  # 上次出院时间: Previous discharge time, 出院时间: Discharge time

    # If admission time is the first admission, then historical diagnosis will be empty
    df['上次诊断'] = df['上次诊断'].fillna('无历史记录')  # 无历史记录: No historical record
    df['上次入院科室'] = df['上次入院科室'].fillna('')  # 上次入院科室: Previous admission department
    df['上次出院科室'] = df['上次出院科室'].fillna('')  # 上次出院科室: Previous discharge department
    df['上次入院时间'] = df['上次入院时间'].fillna('')  # 上次入院时间: Previous admission time
    df['上次出院时间'] = df['上次出院时间'].fillna('')  # 上次出院时间: Previous discharge time
    
    return df


def align_gender(df, label:str='old'):
    """
    Convert '男性' and '女性' (male and female) in the gender column of new data to '1' and '2'
    Convert 1.0 and 2.0 in the gender column of old data to '1' and '2'
    """
    df['性别'] = df['性别'].astype(str)  # 性别: Gender
    if label == 'new':
        df['性别'] = df['性别'].map({'男性': '1', '女性': '2'})  # 男性: Male, 女性: Female
    if label =='old':
        df['性别'] = df['性别'].map({'1.0': '1', '2.0': '2'})
    
    
    return df


def align_date(df, col:str):
    """Convert date-related columns in the medical record front page, including Chinese date format, to standard date format"""

    # Handle Chinese date format (e.g., "2014年12月26日 21:56:48")
    if df[col].dtype == 'object':  # Only preprocess columns of string type
        # Replace Chinese year, month, day with standard separators
        df[col] = df[col].astype(str).str.replace('年', '-').str.replace('月', '-').str.replace('日', '')  # 年: Year, 月: Month, 日: Day
        # Remove any time part that may exist
        df[col] = df[col].str.split(' ').str[0]

    # Convert to datetime format, set errors='coerce' to set values that cannot be converted to NaT
    df[col] = pd.to_datetime(df[col], errors='coerce')

    
    return df


# def align_age(patient_info):
#     """Subtract birth date from admission time to get age"""
#     patient_info['年龄'] = (patient_info['入院时间'] - patient_info['出生日期']).dt.days // 365
#     return patient_info


def align_age(patient_info):
    """Remove the '岁' (year) character from age format like '65岁'"""
    # First convert to string and remove the '岁' character
    patient_info['年龄'] = patient_info['年龄'].astype(str).str.replace('岁', '')  # 年龄: Age, 岁: Year
    # Handle 'nan' values, replace 'nan' or 'NaN', etc. with actual NaN values
    patient_info['年龄'] = patient_info['年龄'].replace(['nan', 'NaN', 'None', 'null'], pd.NA)
    # Use pd.to_numeric to convert, errors='coerce' can set values that cannot be converted to NaN
    patient_info['年龄'] = pd.to_numeric(patient_info['年龄'], errors='coerce')
    return patient_info

def remove_nan_rows(aligned_patient_info):
    """Remove rows where age is nan"""
    # Remove rows where age is nan
    aligned_patient_info = aligned_patient_info[aligned_patient_info['年龄'].notna()]  # 年龄: Age
     
    # Remove rows where gender is nan
    aligned_patient_info = aligned_patient_info[aligned_patient_info['性别'].notna()]  # 性别: Gender
    
    # Remove rows where admission time is nan
    aligned_patient_info = aligned_patient_info[aligned_patient_info['入院时间'].notna()]  # 入院时间: Admission time

    # Remove rows where discharge time is nan
    aligned_patient_info = aligned_patient_info[aligned_patient_info['出院时间'].notna()]  # 出院时间: Discharge time

    return aligned_patient_info


def remove_speicial_fig1(aligned_patient_info):

    """
    According to the requirements of fig1 in the paper, remove some special data
    """
    # Remove rows where age is less than or equal to 18 or greater than or equal to 100
    aligned_patient_info = aligned_patient_info[aligned_patient_info['年龄'] > 18]  # 年龄: Age
    aligned_patient_info = aligned_patient_info[aligned_patient_info['年龄'] < 100]  # 年龄: Age


    return aligned_patient_info
    
def align_data_formats(patient_info, data_mark, label):
    """
    Align data formats
    Parameters:
        patient_info: Medical record front page data
        data_mark: data mark data
        label: label str to mark whether it's old or new
    """
    aligned_patient_info = patient_info.copy()
    aligned_data_mark = data_mark.copy()


    # Align medical record number format
    aligned_patient_info = align_id_format(aligned_patient_info, label)
    aligned_data_mark = align_id_format(aligned_data_mark, label)

    # Align gender format
    aligned_patient_info = align_gender(aligned_patient_info, label)



    # Align date data format
    aligned_patient_info = align_date(aligned_patient_info, '入院时间')  # 入院时间: Admission time
    aligned_patient_info = align_date(aligned_patient_info, '出院时间')  # 出院时间: Discharge time
    aligned_patient_info = align_date(aligned_patient_info, '出生日期')  # 出生日期: Birth date
    aligned_data_mark = align_date(aligned_data_mark, '检查时间')  # 检查时间: Examination time


    # Align age format
    aligned_patient_info = align_age(aligned_patient_info)

    # Align historical information
    aligned_patient_info = align_diagnosis_next_day(aligned_patient_info)

    # Align date format in historical information
    aligned_patient_info = align_date(aligned_patient_info, '上次入院时间')  # 上次入院时间: Previous admission time
    aligned_patient_info = align_date(aligned_patient_info, '上次出院时间')  # 上次出院时间: Previous discharge time

    # Remove rows where age is nan
    aligned_patient_info = remove_nan_rows(aligned_patient_info)

    # According to the requirements of fig1 in the paper, remove some special data
    aligned_patient_info = remove_speicial_fig1(aligned_patient_info)

    

    return aligned_patient_info, aligned_data_mark


