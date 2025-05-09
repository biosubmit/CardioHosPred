import pandas as pd
import numpy as np

def extract_patient_adress_dicy(df: pd.DataFrame):
    """
    Extract a mapping dictionary of patient ID and address from the dataframe, considering ID duplications, using pandas operations
    For example:
    # Remove duplicate rows based on 'id' column, keeping the first occurrence of the address
    #    keep='first' is the default behavior, indicating to keep the first record corresponding to each id
    #    If you want to keep the last occurrence of the address, you can use keep='last'
    unique_id_address = id_address_df.drop_duplicates(subset=['id'], keep='first')


    #    Set the 'id' column as the index, then select the 'address' column, and finally call .to_dict()
    id_address_dict = unique_id_address.set_index('id')['address'].to_dict()

    
    Args:
    df: Dataframe containing patient ID and address
    
    Returns:
    Mapping dictionary of patient ID and address
    """
    df = df[['病案号', '地址.1']]  # 病案号: Medical record number, 地址.1: Address
    df = df.drop_duplicates(subset=['病案号'], keep='first')  # 病案号: Medical record number
    patient_adress_dict = df.set_index('病案号')['地址.1'].to_dict()  # 病案号: Medical record number, 地址.1: Address

    return patient_adress_dict


def extract_normal_range_dict(df: pd.DataFrame):
    """
    Extract normal range values for examination items from the dataframe

    Args:
    df: Dataframe containing examination items and normal ranges
    
    Returns:
    Mapping dictionary of examination items and normal ranges
    """
    df = df[['项目名称', '范围']]  # 项目名称: Item name, 范围: Range
    df = df.drop_duplicates(subset=['项目名称'], keep='first')  # 项目名称: Item name
    normal_range_dict = df.set_index('项目名称')['范围'].to_dict()  # 项目名称: Item name, 范围: Range
    return normal_range_dict


def add_patient_address(df_with_address, df_without_address):
    """
    Add patient address information from the first dataframe to the second dataframe.
    
    Args:
    df_with_address: Dataframe containing patient ID and address information
    df_without_address: Dataframe containing only patient ID but no address information
    
    Returns:
    Merged dataframe containing patient ID and address information
    """

    address_dict = extract_patient_adress_dicy(df_with_address)

    # Add address information to df_without_address based on the dictionary
    df_without_address['地址.1'] = df_without_address['病案号'].map(address_dict)  # 地址.1: Address, 病案号: Medical record number

    return df_without_address



def add_normal_range(df_with_range, df_without_range, item_col='项目名称'):
    """
    Add normal range from the first dataframe to the second dataframe.
    
    Args:
    df_with_range: Dataframe containing examination items and normal ranges
    df_without_range: Dataframe containing examination items but no normal ranges
    item_col: Column name for examination items, default is '项目名称'
    range_col: Column name for normal ranges, default is '范围'
    
    Returns:
    Merged dataframe containing examination items and normal ranges
    """
    range_dict = extract_normal_range_dict(df_with_range)
    df_without_range['范围'] = df_without_range[item_col].map(range_dict)  # 范围: Range, 项目名称: Item name

    return df_without_range



def concat_old_and_new_data(old_patient_info, old_data_mark, new_patient_info, new_data_mark):
    """
    Merge old version data and new version data together.
    
    Args:
    old_patient_info: Old version patient information dataframe
    old_data_mark: Old version data mark dataframe
    new_patient_info: New version patient information dataframe
    new_data_mark: New version data mark dataframe
    
    Returns:
    Merged dataframe
    """
    # 1. Add patient address and normal range

    new_patient_info = add_patient_address(old_patient_info, new_patient_info)
    new_data_mark = add_normal_range(old_data_mark, new_data_mark)
    

    # 1.1 Merge patient information
    merged_patient_info = pd.concat([old_patient_info, new_patient_info], ignore_index=True)
    
    # 1.2 Merge data marks
    merged_data_mark = pd.concat([old_data_mark, new_data_mark], ignore_index=True)


    return merged_patient_info, merged_data_mark
    

    


    
