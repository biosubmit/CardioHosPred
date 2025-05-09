import pandas as pd
import os
import glob
from loguru import logger
from typing import List, Dict

def read_csv_with_fallback_encoding(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file with fallback encoding options if the primary encoding fails.
    
    Args:
        file_path: Path to the CSV file to read
        
    Returns:
        DataFrame containing the CSV data
        
    Raises:
        Exception: If all encoding attempts fail
    """
    encodings = ['utf-8']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode {file_path} with {encoding} encoding")
            continue
    
    # If we get here, all encodings failed
    raise Exception(f"Failed to read {file_path} with any of the encodings: {encodings}")

def concat_subfile(old_medical_record_path: str) -> pd.DataFrame:
    """
    Concatenate all CSV files in the specified directory into a single DataFrame.
    
    Args:
        old_medical_record_path: Path to the directory containing CSV files
        
    Returns:
        DataFrame containing the combined data from all CSV files
    """
    combined_data_path = f"{old_medical_record_path}/combined_data.csv"
    
    # Check if combined data file already exists
    if os.path.exists(combined_data_path):
        logger.info(f"Loading existing combined data from {combined_data_path}")
        return pd.read_csv(combined_data_path, low_memory=False)
    
    # Get all CSV files in the directory
    csv_files = glob.glob(f"{old_medical_record_path}/*.csv")
    
    if not csv_files:
        logger.warning(f"No CSV files found in {old_medical_record_path}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(csv_files)} CSV files to combine")
    
    # Read and combine all CSV files
    dataframes = []
    for file in csv_files:
        try:
            df = read_csv_with_fallback_encoding(file)
            dataframes.append(df)
            logger.info(f"Successfully read {file}")
        except Exception as e:
            logger.error(f"Error reading {file}: {str(e)}")
            # Continue with other files if one fails
            continue
    
    if not dataframes:
        logger.error("No files could be read successfully")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_data = pd.concat(dataframes, ignore_index=True)
    
    # Save combined data
    combined_data.to_csv(combined_data_path, index=False, encoding="utf-8")
    logger.success(f"Combined data saved to {combined_data_path}")
    
    return combined_data

if __name__ == "__main__":
    old_medical_record_path = "../../data/hospital_C/"
    logger.info(f"Starting to combine files from {old_medical_record_path}")
    
    try:
        combined_data = concat_subfile(old_medical_record_path)
        logger.info(f"Combined data shape: {combined_data.shape}")
    except Exception as e:
        logger.error(f"Error in concatenating files: {str(e)}")