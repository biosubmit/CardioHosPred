"""
This document contains miscellaneous functions used in Peking Union Medical College Hospital data processing

"""
import pandas as pd
from data_alignment_utils import align_id_format
from loguru import logger


def check_dtype_row(row, allowed_dtype):
    """Check the data type of a row
    row: a row in dataframe
    allowed_dtype: allowed data types [str, int, float, datetime]
    return:
        True: data type meets requirements
        False: data type does not meet requirements
    """
    if row.dtype not in allowed_dtype:
        # Output log to notify that a column's data type does not meet requirements
        logger.error(f"{row.name} column type does not meet requirements: {row.dtype}")
        raise ValueError
    else:
        logger.info(f"{row.name} column type meets requirements: {row.dtype}")
    
    

            

def diff_old_new_patient_id(new_patient_id: pd.DataFrame, old_patient_id: pd.DataFrame) -> tuple[set, set]:
    """
    Check the difference between old_patient_id and new_patient_id
    args:
        new_patient_id: dataframe, new_patient_id
        old_patient_id: dataframe, old_patient_id
    """

    new_patient_id_list = new_patient_id['病案号'].unique().tolist()  # 'medical record number'
    old_patient_id_list = old_patient_id['病案号'].unique().tolist()
    print(len(new_patient_id_list))
    print(len(old_patient_id_list))
    print(type(new_patient_id_list[0]))
    print(type(old_patient_id_list[0]))
    return set(new_patient_id_list) - set(old_patient_id_list), set(old_patient_id_list)

def build_new_patient_id_candidates():
    """
    Read the medical record number column from patient.csv files, build candidate medical record numbers for new_patient_id.
    For cases where a row has multiple numbers, all numbers need to be listed.
    return:
        patient_df['病案号']: dataframe, candidate medical record numbers for new_patient_id
    """

    patient_df_1 = pd.read_csv('./data/hospital_A/new/group1/患者_utf8.csv', usecols=['病案号'])  # 'patients_utf8.csv'
    patient_df_2 = pd.read_csv('./data/hospital_A/new/group2/患者_utf8.csv', usecols=['病案号'])
    # Merge

    patient_df = pd.concat([patient_df_1, patient_df_2])
    # Remove rows with only one medical record number
    patient_df = patient_df[patient_df['病案号'].apply(lambda x: len(str(x).split('；')) > 1)]
    # Step 0: Add group_id to each row to record the original group
    patient_df['group_id'] = patient_df.index  # Keep original index as group_id

    # Step 1: Split
    patient_df['病案号'] = patient_df['病案号'].apply(
        lambda x: str(x).split('；') if pd.notnull(x) else []
    )

    # Step 2: Clean
    patient_df['病案号'] = patient_df['病案号'].apply(
        lambda lst: [s.replace('_XQ', '') for s in lst]
    )

    # Step 3: Explode
    patient_df = patient_df.explode('病案号')


    return patient_df

def find_related_ids_by_group(df, target_id):
    group_ids = df[df['病案号'] == target_id]['group_id'].unique()
    related_ids = []
    for group_id in group_ids:
        related_ids.extend(df[df['group_id'] == group_id]['病案号'].unique().tolist())
    return set(related_ids)


def build_new_patient_id_mapping(old_patient_id, new_patient_id):
    """
    For new_patient_id in diff_old_new_patient_id, build a candidate dictionary
    """
    
    diff_id, old_patient_id_set = diff_old_new_patient_id(new_patient_id, old_patient_id)
    candidate_id = build_new_patient_id_candidates()
    print(len(diff_id))
    print(len(candidate_id))
    # save candidate_id
    candidate_id.to_csv('./data/hospital_A/tmp_preprocessed_data/candidate_id.csv', index=False)
    i=0
    id = '1337494'
    related_ids = find_related_ids_by_group(candidate_id, id)
    print(related_ids)
    # Check if there are elements from old_patient_id_set in related_ids, if yes, output the count
    # 
    i = 0
    for id in diff_id:
        related_ids = find_related_ids_by_group(candidate_id, id)
        print(id, len(related_ids & old_patient_id_set))
        print(related_ids & old_patient_id_set)
        if len(related_ids & old_patient_id_set) == 0:
            i+=1
            print(id, 'not found')
            print(i)




            # Find other IDs with the same index as id. First find the index of id in candidate_id, then find other IDs with the same index in candidate_id


    return None
    

def map_new_patient_id_to_old_patient_id(old_patient_id, new_patient_id):
    """
    For new_patient_id, if it's not in old_patient_id, then substitute from candidate_patient_id to ensure finding a new_patient_id that exists in old_patient_id
    """
    return dict(zip(old_patient_id, new_patient_id))

def check_final_column_name(df):
    """Check if the column names in df meet requirements
    If it's data_mark, check if the following columns exist:
        Medical record number, examination time, gender, age, last admission time, last discharge time, hospitalization count marker,
        last admission department, last discharge department, 'GDP per capita (yuan)', 'Number of hospitals and health centers (units)',
        'Number of beds in hospitals and health centers (units)', 'Number of doctors (persons)', 'Average employee salary (yuan)',
        BMI, heart rate, body temperature, respiration, systolic pressure, diastolic pressure,
        project name, examination result, range, upper limit, lower limit, normalized examination result, time_gap
    """
    # Check if column names meet requirements 
    required_columns = ['病案号', '检查时间', '性别', '年龄', '上次入院时间', '上次出院时间', '住院次数标记',
        '上次入院科室', '上次出院科室', '人均地区生产总值(元)', '医院、卫生院数(个)',
        '医院、卫生院床位数(张)', '医生数(人)', '职工平均工资(元)','bmi', '心率', '体温', '呼吸', '高压', '低压',
        '项目名称', '检查结果', '范围', '上限', '下限', '归一化检查结果', 'time_gap']
    # Check if the above column names exist
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"data_mark is missing the following columns: {set(required_columns) - set(df.columns)}")
    else:
        logger.info(f"data_mark column names meet requirements")
        # Adjust order
        df = df[required_columns]


    return df.columns



def convert_longtable_to_wide(df):
    """
    Convert examination results from long table format to wide table format
    """
    # Convert examination results from long table to wide table

    selected_columns = ['病案号', '检查时间', '项目名称', '归一化检查结果','bmi', '高压', '低压', '心率', '体温', '呼吸']
    df = df[selected_columns]
    wide_data = df.pivot_table(
            index= ['病案号', '检查时间'],
            columns='项目名称',
            values='归一化检查结果',
            aggfunc='mean',
            observed=True
        ).reset_index()
        

    basic_info = (
        df.sort_values(by=['病案号', '检查时间'])  # Ensure order
        .drop_duplicates(subset=['病案号', '检查时间'])  # Keep the first occurrence by default
        [['病案号', '检查时间', 'bmi', '高压', '低压', '心率', '体温', '呼吸']]
        )
    wide_data_sum = pd.merge(wide_data, basic_info, on=['病案号', '检查时间'], how='left')
    return wide_data_sum

if __name__ == "__main__":

    old_patient_id = pd.read_csv('./data/hospital_A/tmp_preprocessed_data/old_patient_info.csv', usecols=['病案号'])
    new_patient_id = pd.read_csv('./data/hospital_A/tmp_preprocessed_data/new_patient_info.csv', usecols=['病案号'])
    old_patient_id = align_id_format(old_patient_id, label='old')
    new_patient_id = align_id_format(new_patient_id, label='new')
    build_new_patient_id_mapping(old_patient_id, new_patient_id)