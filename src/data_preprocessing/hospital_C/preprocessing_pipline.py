import os
import pandas as pd
from loguru import logger
import sys
# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# logger color
logger.add(sys.stdout, colorize=True)
# Import existing preprocessing scripts
from pre_concact_subfile import concat_subfile
from pre_feature_select import select_columns

from data_format_utils import format_date, format_data
from data_economic_utils import integrate_economy_data
from data_combine_utils import combine_data

from data_alignment_utils import align_data

ECONOMIC_INFO_FLAG = True
DIAGNOSIS_EMBEDDING_FLAG = False

def main_preprocessing_pipeline(
    old_medical_record_path,
    output_dir="./data/hospital_A/tmp_preprocessed_data"
):
    """
    Main preprocessing pipeline for Hospital C data
    
    Parameters:
        old_medical_record_path: Path to old version of medical record information file
        output_dir: Output directory
    
    Returns:
        Final processed dataset
    """
    
    logger.info("Starting Hospital C data preprocessing pipeline...")
  
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Data alignment ==============================================
    logger.info("Step 1: Data merging")
    
    # Read all csv files in the directory and merge them into one dataframe
    combined_data = concat_subfile(old_medical_record_path)

    # 1.1 Process old version data - call pre_feature_select_old.py
    logger.info("1.1 Selecting feature columns...")

    selected_data = select_columns(combined_data)
    print(selected_data.columns)
    selected_data.to_csv(f"{output_dir}/selected_feature_data.csv", index=False, encoding="utf-8")
    
    slected_data = pd.read_csv(f"{output_dir}/selected_feature_data.csv", encoding="utf-8")


    new_data = pd.read_csv(f"{output_dir}/merged_data.csv", encoding="utf-8")
    logger.info("1.2 Merging data...")
    combined_data = combine_data(slected_data, new_data)

    # 1.2 Align data format - call column format processing functions
    logger.info("1.2 Column format processing...")
    formatted_data = format_data(combined_data)


 

    logger.info(f"Number of unique medical record numbers after formatting: {len(formatted_data['病案号'].unique())}")  # 'medical_record_number'

    
    # Step 2: Incorporate economic information ========================================
    logger.info("2.1 Incorporating economic information")

  
    #economy_formatted_data = integrate_economy_data(formatted_data)
    #economy_formatted_data.to_csv(f"{output_dir}/economy_formatted_data.csv", index=False, encoding="utf-8")
    #print(economy_formatted_data.columns)
    
    # Step 3: Align data ========================================
    logger.info("3.1 Aligning data...")
    aligned_data = align_data(formatted_data)
    aligned_data.to_csv(f"{output_dir}/final_data.csv", index=False, encoding="utf-8")
    logger.info(f"Column names after alignment: {aligned_data.head()}")
    logger.info(f"Number of unique medical record numbers after alignment: {len(aligned_data['病案号'].unique())}")  # 'medical_record_number'

    # Missing medical record numbers
    missing_mrn = aligned_data[aligned_data['病案号'].isna()]['病案号']  # 'medical_record_number'
    logger.info(f"Number of missing medical record numbers: {len(missing_mrn)}")



   
    return None

if __name__ == "__main__":
    # Set input file paths
    # 设置输入文件路径
    medical_record_path = "../../data/hospital_C/"
  
    # 输出目录
    output_dir = "../../data/hospital_C/tmp_preprocessed_data"
    
    # 运行预处理流水线
    main_preprocessing_pipeline(
        medical_record_path,
        output_dir
    )
