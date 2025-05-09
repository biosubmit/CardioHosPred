import os
import pandas as pd
from loguru import logger
import sys
# Set the working directory
# Set the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# logger color
logger.add(sys.stdout, colorize=True)
# Import existing preprocessing scripts
# Import existing preprocessing scripts
from data_feature_select_utils import preprocess_old_data
from data_alignment_utils import align_data_formats
from data_format_utils import format_columns
from data_normalcheck_utils import normalize_data_mark
from data_economic_utils import integrate_economy_data
from data_diagnosis_utils import incorporate_diagnosis_embedding
from data_timegap_utils import merge_datamark_and_patient_info, merge_datamark_and_patient_info_v2
from data_helper import convert_longtable_to_wide

ECONOMIC_INFO_FLAG = True
DIAGNOSIS_EMBEDDING_FLAG = False

def main_preprocessing_pipeline(
    old_medical_record_path,
    old_data_mark_path,
    output_dir="./data/hospital_A/tmp_preprocessed_data"
):
    """
    Main preprocessing pipeline for Hospital A data
    
    Parameters:
        old_medical_record_path: Path to old version of medical record information file
        old_data_mark_path: Path to old version of data_mark file
        new_data_path: Root directory of new data, including subdirectories such as group1, group2, etc.
        economic_info_path: Path to economic information file (optional)
        diagnosis_embedding_path: Path to diagnosis embedding file (optional)
        output_dir: Output directory
    
    Returns:
        The final processed dataset
    """
    
    logger.info("Starting Hospital A data preprocessing pipeline...")
    
    # Create output directory
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Data Alignment
    # Step 1: Data Alignment
    logger.info("Step 1: Data Alignment")
    
    # 1.1 Process old version data - Call pre_feature_select_old.py
    # 1.1 Process old version data - Call pre_feature_select_old.py
    logger.info("1.1 Selecting feature columns...")
    old_patient_info, old_data_mark = preprocess_old_data(
        old_medical_record_path, old_data_mark_path
    )

    # 1.2 Align data formats - Call column format processing function
    # 1.2 Align data formats - Call column format processing function
    logger.info("1.2 Column format processing...")
    aligned_old_patient_info, aligned_old_data_mark = align_data_formats(
        old_patient_info, old_data_mark, label='old'
    )
    aligned_old_patient_info.to_csv(f"{output_dir}/pre_aligned_old_patient_info.csv", index=False)
    aligned_old_data_mark.to_csv(f"{output_dir}/pre_aligned_old_data_mark.csv", index=False)

 
    # Step 2: Process data feature columns sequentially
    # Step 2: Process data feature columns sequentially
    logger.info("Step 2: Process data feature columns sequentially")
    
    # 3.1 Normalize and format columns
    # 3.1 Normalize and format columns
    logger.info("2.1 Format normal range columns in data_mark table")
    formatted_data_mark = format_columns(aligned_old_data_mark)
    # save
    formatted_data_mark.to_csv(f"{output_dir}/pre_formatted_data_mark.csv", index=False)


    logger.info("2.2 Normalize examination results")
    normalized_data_mark = normalize_data_mark(formatted_data_mark)
    # save
    normalized_data_mark.to_csv(f"{output_dir}/pre_normalized_data_mark.csv", index=False)

    
    logger.info("2.3 Convert data_mark table to wide format")
    normalized_data_mark = pd.read_csv(f"{output_dir}/pre_normalized_data_mark.csv",parse_dates=['检查时间']) # 检查时间: Examination time
    wide_normalized_data_mark = convert_longtable_to_wide(normalized_data_mark)
    wide_normalized_data_mark.to_csv(f"{output_dir}/pre_wide_data_sum.csv", index=False)

    wide_normalized_data_mark = pd.read_csv(f"{output_dir}/pre_wide_data_sum.csv",parse_dates=['检查时间']) # 检查时间: Examination time
    
    # read
    aligned_old_patient_info = pd.read_csv(f"{output_dir}/pre_aligned_old_patient_info.csv",parse_dates=['入院时间','出院时间','上次入院时间','上次出院时间']) 
    # 入院时间: Admission time, 出院时间: Discharge time, 上次入院时间: Last admission time, 上次出院时间: Last discharge time
    
    # 2.3 Introduce economic information
    # 2.3 Introduce economic information
    if ECONOMIC_INFO_FLAG:
        logger.info("2.3 Introducing economic information...")
        economy_wide_normalized_data_mark = integrate_economy_data(wide_normalized_data_mark)
    
    # 2.4 Introduce diagnosis information embedding
    # 2.4 Introduce diagnosis information embedding
    if DIAGNOSIS_EMBEDDING_FLAG:
        logger.info("2.4 Introducing diagnosis information embedding...")
        diagnosis_aligned_old_patient_info = incorporate_diagnosis_embedding(aligned_old_patient_info)
    
    # save
    economy_wide_normalized_data_mark.to_csv(f"{output_dir}/pre_economy_wide_normalized_data_mark.csv", index=False)
    # diagnosis_aligned_old_patient_info.to_csv(f"{output_dir}/pre_diagnosis_aligned_old_patient_info.csv", index=False)
    
    # read
    economy_wide_normalized_data_mark = pd.read_csv(f"{output_dir}/pre_economy_wide_normalized_data_mark.csv",parse_dates=['检查时间']) # 检查时间: Examination time
    # diagnosis_aligned_old_patient_info = pd.read_csv(f"{output_dir}/pre_diagnosis_aligned_old_patient_info.csv",parse_dates=['入院时间','出院时间','上次入院时间','上次出院时间'])
    # read
    aligned_old_patient_info = pd.read_csv(f"{output_dir}/pre_aligned_old_patient_info.csv",parse_dates=['入院时间','出院时间','上次入院时间','上次出院时间'])
    # 入院时间: Admission time, 出院时间: Discharge time, 上次入院时间: Last admission time, 上次出院时间: Last discharge time
    
    # 2.5 Sort and calculate time intervals
    # 2.5 Sort and calculate time intervals
    logger.info("2.5 Sort and calculate time intervals...")
    processed_data = merge_datamark_and_patient_info_v2(economy_wide_normalized_data_mark, aligned_old_patient_info)
    logger.info(f"processed_data: {processed_data.shape}")
    logger.info(f"processed_data: {processed_data.columns}")
    logger.info(f"processed_data: {processed_data.head()}")

    # Output statistical distribution of time difference column
    # Output statistical distribution of time difference column
    logger.info(f"processed_data['时间差'].describe()") # 时间差: Time difference
    logger.info(processed_data['时间差'].describe()) # 时间差: Time difference
    # Output number of rows with time difference equal to 0
    # Output number of rows with time difference equal to 0
    logger.info(f"Number of rows where processed_data['时间差']==0: {len(processed_data[processed_data['时间差']==0])}")
    # save
    processed_data.to_csv(f"{output_dir}/pre_processed_data.csv", index=False)


    
    # Step 3: Final check if the output meets requirements, and save
    # Step 3: Final check if the output meets requirements, and save
    logger.info("3.1 Checking if output meets requirements...")
    # Check if column names of data_mark table meet requirements
    # Check if column names of data_mark table meet requirements
    # check_result = check_data_mark_columns(processed_data)
    # if check_result:
    #     logger.info("Check passed, saving final result...")
    #     logger.info("Check passed, saving final result...")
    # else:
    #     logger.error("Check failed, please check the data...")
    #     logger.error("Check failed, please check the data...")
    #     raise ValueError

    # Save final processing result
    # Save final processing result
    final_output_path = f"{output_dir}/final_preprocessed_data.csv"
    processed_data.to_csv(final_output_path, index=False)
    logger.info(f"Preprocessing completed! Final data saved to: {final_output_path}")
    
    return None

if __name__ == "__main__":
    # Set input file paths
    # Set input file paths
    old_medical_record_path = "../../data/hospital_A/old/病案首页信息提取.csv" # 病案首页信息提取: Medical record front page information extraction
    old_data_mark_path = "../../data/hospital_A/old/data_mark.csv"
    new_data_path = "../../data/hospital_A/new"

    # Output directory
    # Output directory
    output_dir = "../../data/hospital_A/tmp_preprocessed_data"
    
    # Run preprocessing pipeline
    # Run preprocessing pipeline
    main_preprocessing_pipeline(
        old_medical_record_path,
        old_data_mark_path,
        output_dir
    )
