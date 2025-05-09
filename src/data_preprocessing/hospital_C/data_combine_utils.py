import pandas as pd
from loguru import logger
def combine_data(df1,df2):
    """
    Combine data from two dataframes
    """

    # Remove specified columns
    df2.drop(columns=['住院天数'],inplace=True)  # 住院天数: Length of stay

    # Check if column names in df1 and df2 are exactly the same
    common_columns = set(df1.columns.tolist()) & set(df2.columns.tolist())
    if len(common_columns) != len(df1.columns.tolist()) or len(common_columns) != len(df2.columns.tolist()):
        raise ValueError("Column names in df1 and df2 are not the same")
    


    
    # Merge df1 and df2
    combined_data = pd.concat([df1, df2], axis=0)

    logger.info(f"Number of unique medical record numbers after merging: {len(combined_data['病案号'].unique())}")  # 病案号: Medical record number

    combined_data['病案号'] = combined_data['病案号'].astype(str)  # 病案号: Medical record number

    # Check if there are medical record numbers starting with 0
    if combined_data['病案号'].str.startswith('0').any():  # 病案号: Medical record number
        print(combined_data[combined_data['病案号'].str.startswith('0')])
        raise ValueError("There are medical record numbers starting with 0")

      

    # Convert medical record number to int (ignore NaN values)
    combined_data['病案号'] = combined_data['病案号'].astype(int, errors='ignore')  # 病案号: Medical record number



    logger.info(f"Column names after merging: {combined_data.columns}")


    return combined_data


