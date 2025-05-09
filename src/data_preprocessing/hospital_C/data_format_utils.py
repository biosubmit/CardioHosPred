# Date change utility
from datetime import datetime
import pandas as pd
from typing import Union

def format_date(date_str: str) -> Union[str, None]:
    """Convert different formats of date strings to a unified format.

    Supports the following input formats:
    - '%Y/%m/%d %H:%M' (full format with time, like '2022/10/24 9:07')
    - '%Y/%m/%d' (date-only format, like '2022/9/20')
    - '%Y-%m-%d %H:%M:%S' (database format, like '2022-10-24 09:07:00')
    - '0000-00-00 00:00:00' (special empty date format, will return None)

    Args:
        date_str: Date string to be formatted

    Returns:
        Union[str, None]: Formatted date string in '%Y-%m-%d %H:%M' format,
                          returns None for invalid dates or special empty dates

    Examples:
        >>> format_date("2022/9/7 8:32")
        '2022-09-07 08:32'
        >>> format_date("2022/9/20")
        '2022-09-20 00:00'
        >>> format_date("0000-00-00 00:00:00")
        None
    """
    # Check if it's empty or a special empty date format
    if not date_str or date_str == '0000-00-00 00:00:00':
        return None
        
    # Try different date formats
    formats_to_try = [
        '%Y/%m/%d %H:%M',    # 2022/10/24 9:07
        '%Y/%m/%d',          # 2022/9/20
        '%Y-%m-%d %H:%M:%S', # 2022-10-24 09:07:00
        '%Y-%m-%d %H:%M',    # 2022-10-24 09:07
        '%Y-%m-%d',          # 2022-10-24
    ]
    
    for date_format in formats_to_try:
        try:
            date_obj = datetime.strptime(date_str, date_format)
            return date_obj.strftime('%Y-%m-%d %H:%M')
        except ValueError:
            continue
    
    # If all formats fail, raise an exception
    raise ValueError(f"Unable to parse date format: {date_str}")


def format_address(address_str: str) -> str:
    """Format address string.

    Extract province and city level address from the full address, 
    only keep characters up to the city level (e.g., from "Shandong Province Qingdao City xxx" only keep "Shandong Province Qingdao City")

    Args:
        address_str: Address string
        
    Returns:
        str: Formatted address string
    """
    return address_str.split('市')[0]+'市'  # 市: City


def format_age(age_str: str) -> Union[int, None]:
    """Format age string.
    
    Convert age string to integer format by removing any non-numeric characters.
    
    Args:
        age_str: Age string that may contain '岁' (years) or other characters
        
    Returns:
        Union[int, None]: Formatted age as integer, or None if conversion fails
    """
    # Check if age string is a number or a number with '岁' (years)
    if age_str.isdigit():
        return int(age_str)
    elif age_str.isdigit() and '岁' in age_str:  # 岁: Years
        return int(age_str.split('岁')[0])
    else:
        return None




def format_data(data: pd.DataFrame) -> pd.DataFrame:
    """Format date columns in the DataFrame.
    
    Format specified date columns to a unified date format.
    
    Args:
        data: DataFrame containing date columns
        
    Returns:
        pd.DataFrame: DataFrame with formatted date columns
    """
    # Handle possible NaN values
    for col in ['入院时间', '出院时间']:  # 入院时间: Admission time, 出院时间: Discharge time
        if col in data.columns:
            data[col] = data[col].apply(lambda x: format_date(x) if pd.notna(x) else None)

    # Handle age column
    # First convert age column to string
    data['年龄'] = data['年龄'].astype(str)  # 年龄: Age
    data['年龄'] = data['年龄'].apply(lambda x: format_age(x) if pd.notna(x) else None)  # 年龄: Age


    
 
    return data

