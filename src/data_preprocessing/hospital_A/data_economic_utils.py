import pandas as pd
from loguru import logger
from typing import Dict, List, Optional, Union

# ---------------------- 1. Read Data (Unchanged) ----------------------

def read_checkup_data(file_path: str = 'check_up_merged_result.csv') -> pd.DataFrame:
    return pd.read_csv(file_path)

def get_adresss_dict(file_path: str = '../../data/hospital_A/Province_city_correct.csv') -> dict:
    df = pd.read_csv(file_path)
    df_unique = df.drop_duplicates(subset=['病案号'], keep='first')  # 病案号: Medical record number
    address_dict = dict(zip(df_unique['病案号'], df_unique['城市']))  # 病案号: Medical record number, 城市: City
    return address_dict

def read_citybase_data(file_path: str = '../../data/hospital_A/city_database.xlsx') -> pd.DataFrame:
    df = pd.read_excel(file_path)
    selected_cols = [
        '年份', '行政区划代码', '地区', '人均地区生产总值(元)',
        '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)', '职工平均工资(元)'
    ]
    # 年份: Year, 行政区划代码: Administrative division code, 地区: Region
    # 人均地区生产总值(元): GDP per capita (yuan), 医院、卫生院数(个): Number of hospitals and health centers
    # 医院、卫生院床位数(张): Number of hospital and health center beds, 医生数(人): Number of doctors
    # 职工平均工资(元): Average employee salary (yuan)
    
    df = df[selected_cols].copy() # Use .copy() to avoid SettingWithCopyWarning
    df.dropna(subset=['年份'], inplace=True)  # 年份: Year
    df['年份'] = df['年份'].astype(int)  # 年份: Year
    # df['行政区划代码'] = df['行政区划代码'].astype(str) # If not used later, can be commented out
    # df['省级代码'] = df['行政区划代码'].str.slice(0, 2) # If not used later, can be commented out

    # --- Directly standardize province names when reading ---
    province_code_to_name = {
        '11': '北京', '12': '天津', '13': '河北', '14': '山西', '15': '内蒙古',
        '21': '辽宁', '22': '吉林', '23': '黑龙江', '31': '上海', '32': '江苏',
        '33': '浙江', '34': '安徽', '35': '福建', '36': '江西', '37': '山东',
        '41': '河南', '42': '湖北', '43': '湖南', '44': '广东', '45': '广西',
        '46': '海南', '50': '重庆', '51': '四川', '52': '贵州', '53': '云南',
        '54': '西藏', '61': '陕西', '62': '甘肃', '63': '青海', '64': '宁夏',
        '65': '新疆'
    }
    # Calculate province code (if needed)
    df['省级代码'] = df['行政区划代码'].astype(str).str.slice(0, 2)  # 省级代码: Province code, 行政区划代码: Administrative division code
    df['省份'] = df['省级代码'].map(province_code_to_name)  # 省份: Province, 省级代码: Province code
    # Clean province names (remove suffixes)
    df['省份_norm'] = df['省份'].apply(lambda x: normalize_province_name(x) if pd.notna(x) else None)  # 省份_norm: Normalized province, 省份: Province
    # Clean region names (may be needed, e.g., remove '市')
    # df['地区_norm'] = df['地区'].apply(lambda x: x.replace('市', '') if isinstance(x, str) and x.endswith('市') else x)
    df['地区_norm'] = df['地区'] # Assume '地区' in citybase_data is already standardized city name  # 地区_norm: Normalized region, 地区: Region

    # --- Ensure economic indicator columns are numeric for later averaging ---
    economic_columns = [
        '人均地区生产总值(元)', '医院、卫生院数(个)',
        '医院、卫生院床位数(张)', '医生数(人)', '职工平均工资(元)'
    ]
    for col in economic_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# ---------------------- 2. Parsing and Extraction Functions (Unchanged) ----------------------

def extract_province(address: str) -> Optional[str]:
    if pd.isnull(address):
        return None
    if '省' in address:  # 省: Province
        return address.split('省')[0]
    if '自治区' in address:  # 自治区: Autonomous Region
        return address.split('自治区')[0]
    # Handle direct-administered municipalities and other special cases
    special_cities = ['北京', '上海', '天津', '重庆']  # Beijing, Shanghai, Tianjin, Chongqing
    for city in special_cities:
        if address.startswith(city):
            return city
    # If address directly starts with province name but no suffix (e.g., '内蒙古呼和浩特市')
    provinces_short = ['内蒙古', '黑龙江', '西藏', '新疆', '广西', '宁夏'] # Examples
    for prov in provinces_short:
         if address.startswith(prov):
             return prov
    # Try splitting by city, applicable for cases like '北京市海淀区'
    if '市' in address:  # 市: City
        return address.split('市')[0]

    return address # Fallback

def extract_city(address: str) -> Optional[str]:
    if pd.isnull(address):
        return None
    # First handle special prefixes
    prefixes_to_remove = ['内蒙古自治区', '广西壮族自治区', '西藏自治区', '宁夏回族自治区', '新疆维吾尔自治区']
    for prefix in prefixes_to_remove:
        if address.startswith(prefix):
            address = address[len(prefix):]
            break # Remove the first matching prefix only
    # Then remove by province
    if '省' in address:  # 省: Province
        address = address.split('省', 1)[-1] # Split once, take the latter part

    # Finally extract city
    if '市' in address:  # 市: City
        return address.split('市')[0].strip() # .strip() removes possible spaces

    # Handle cases without '市' but possibly city-level units (like autonomous prefectures, regions, leagues)
    # Simplified handling: If after extracting province, the remaining part doesn't contain county/district etc., it might be city-level
    # (This logic is complex, simplified here, may not be fully accurate)
    province = extract_province(address) # Need original address
    if province and address.startswith(province):
         maybe_city = address[len(province):].strip()
         if '市' not in maybe_city and '县' not in maybe_city and '区' not in maybe_city and '旗' not in maybe_city:
              return maybe_city # Might be 'region' or 'autonomous prefecture'

    return None

def normalize_province_name(province_name: str) -> Optional[str]:
    if pd.isnull(province_name):
        return None
    suffixes = ['省', '市', '自治区', '壮族自治区', '回族自治区', '维吾尔自治区', '特别行政区']
    # 省: Province, 市: City, 自治区: Autonomous Region, 壮族自治区: Zhuang Autonomous Region, 
    # 回族自治区: Hui Autonomous Region, 维吾尔自治区: Uyghur Autonomous Region, 特别行政区: Special Administrative Region
    name = province_name.strip()
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


# ---------------------- 3. Optimized Integration Functions ----------------------
# ---------------------- 3. Optimized Integration Functions (Corrected Version) ----------------------

def integrate_economy_data_optimized(
    normalized_data_mark: pd.DataFrame, # Main data containing Medical record number and Examination time
    citybase_data: pd.DataFrame, # Preprocessed city economic data
    address_dict: dict           # Dictionary mapping Medical record number -> Province-city level address
) -> pd.DataFrame:
    """
    Efficiently integrate economic data into examination data using vectorized operations.
    """
    # 1. Prepare main data (normalized_data_mark)
    logger.info("Starting to prepare main data...")
    main_df = normalized_data_mark.copy()
    
    # Ensure examination time column is datetime type
    if '检查时间' in main_df.columns:  # 检查时间: Examination time
        if not pd.api.types.is_datetime64_any_dtype(main_df['检查时间']):
            logger.info("Converting examination time column to datetime type...")
            main_df['检查时间'] = pd.to_datetime(main_df['检查时间'], errors='coerce')  # 检查时间: Examination time
    
    main_df['年份'] = main_df['检查时间'].dt.year # Extract year  # 年份: Year, 检查时间: Examination time
    main_df['省市级地址'] = main_df['病案号'].map(address_dict) # Add address  # 省市级地址: Province-city level address, 病案号: Medical record number

    main_df.loc[:, '省份_raw'] = main_df['省市级地址'].apply(extract_province)  # 省份_raw: Raw province, 省市级地址: Province-city level address
    main_df.loc[:, '省份'] = main_df['省份_raw'].apply(normalize_province_name)  # 省份: Province, 省份_raw: Raw province
    main_df.loc[:, '城市'] = main_df['省市级地址'].apply(extract_city)  # 城市: City, 省市级地址: Province-city level address
    main_df.dropna(subset=['省市级地址', '年份'], inplace=True)  # 省市级地址: Province-city level address, 年份: Year

    # 2. Prepare economic data
    logger.info("Preparing economic data...")
    economic_columns = [
        '人均地区生产总值(元)', '医院、卫生院数(个)',
        '医院、卫生院床位数(张)', '医生数(人)', '职工平均工资(元)'
    ]
    # 人均地区生产总值(元): GDP per capita (yuan), 医院、卫生院数(个): Number of hospitals and health centers
    # 医院、卫生院床位数(张): Number of hospital and health center beds, 医生数(人): Number of doctors
    # 职工平均工资(元): Average employee salary (yuan)
    
    city_data_to_merge = citybase_data[['年份', '地区_norm'] + economic_columns].copy()  # 年份: Year, 地区_norm: Normalized region
    city_data_to_merge = city_data_to_merge.rename(columns={'地区_norm': '城市'})  # 城市: City

    province_avg_data = citybase_data.groupby(['年份', '省份_norm'])[economic_columns].mean().reset_index()
    # 年份: Year, 省份_norm: Normalized province
    province_avg_data = province_avg_data.rename(columns={'省份_norm': '省份'})  # 省份: Province

    # 3. Execute merge
    logger.info("Starting data merge...")
    logger.info("  Merging city data...")
    # First merge: City economic columns enter without suffix (no conflicting columns in main_df)
    merged_df = pd.merge(
        main_df,
        city_data_to_merge,
        on=['年份', '城市'],  # 年份: Year, 城市: City
        how='left'
        # suffixes parameter can be omitted or retained here, same effect, as there's no conflict
    )

    logger.info("  Merging province average data...")
    # Second merge: Province economic columns enter with conflict with existing city columns, add _prov suffix
    merged_df = pd.merge(
        merged_df,
        province_avg_data,
        on=['年份', '省份'],  # 年份: Year, 省份: Province
        how='left',
        suffixes=('', '_prov') # Left side (city) keeps original name, right side (province) adds _prov
    )

    # 4. Combine results: (corrected logic)
    logger.info("Combining final results...")
    for col in economic_columns:
        prov_col = col + '_prov'
        # City data in col column, province data in prov_col column
        # Use combine_first: If col (city data) is NaN, fill with prov_col
        # Result written directly to col column
        if prov_col in merged_df.columns: # Check if province column exists, just in case
             merged_df[col] = merged_df[col].combine_first(merged_df[prov_col])
             # Delete temporary province column
             merged_df.drop(columns=[prov_col], inplace=True)
        # else: # If province column doesn't exist (e.g., a province has no data for a year), city column is final result, no action needed

    # 5. Clean up helper columns that are no longer needed (optional)
    # Ensure only delete existing columns
    cols_to_drop = ['省份_raw', '省市级地址', '省份', '城市', '年份']
    # 省份_raw: Raw province, 省市级地址: Province-city level address, 省份: Province, 城市: City, 年份: Year
    merged_df.drop(columns=[col for col in cols_to_drop if col in merged_df.columns], inplace=True, errors='ignore')


    logger.info("Data integration complete.")
    return merged_df

def integrate_economy_data(
    normalized_data_mark: pd.DataFrame,
) -> pd.DataFrame:
    
    address_dict = get_adresss_dict('../../data/hospital_A/Province_city_correct.csv')
    citybase_data_preprocessed = read_citybase_data('../../data/hospital_A/city_database.xlsx')

    # First add province-city level address
    logger.info("Adding province-city level address information...")
    data_with_address = normalized_data_mark.copy()
    data_with_address['省市级地址'] = data_with_address['病案号'].map(address_dict)  # 省市级地址: Province-city level address, 病案号: Medical record number
    
    # Then call optimized integration function
    result_df = integrate_economy_data_optimized(
        normalized_data_mark=data_with_address,
        citybase_data=citybase_data_preprocessed,
        address_dict=address_dict
    )

    return result_df



# --- Main Program Flow (Unchanged) ---
if __name__ == '__main__':

    logger.info("Step A: Reading examination data")
    data_mark_df = pd.read_csv('../../data/hospital_A/tmp_preprocessed_data/formatted_data_mark.csv', parse_dates=['检查时间'])  # 检查时间: Examination time

    logger.info("Step B: Reading address dictionary")
    address_dict = get_adresss_dict('../../data/hospital_A/Province_city_correct.csv')

    logger.info("Step C: Reading and preprocessing city economic data")
    citybase_data_preprocessed = read_citybase_data('../../data/hospital_A/city_database.xlsx')

    logger.info("Step D & E: Integrating economic data (optimized version)")
    result_df = integrate_economy_data_optimized(
        normalized_data_mark=data_mark_df,
        citybase_data=citybase_data_preprocessed,
        address_dict=address_dict
    )

    logger.info("Step F: Outputting results")
    logger.info(f"\n{result_df.head()}")
    logger.info(f"Number of records after processing: {len(result_df)}")
    logger.info("Missing value statistics for economic indicator columns:")
    economic_cols_final = [
        '人均地区生产总值(元)', '医院、卫生院数(个)',
        '医院、卫生院床位数(张)', '医生数(人)', '职工平均工资(元)'
    ]
    # 人均地区生产总值(元): GDP per capita (yuan), 医院、卫生院数(个): Number of hospitals and health centers
    # 医院、卫生院床位数(张): Number of hospital and health center beds, 医生数(人): Number of doctors
    # 职工平均工资(元): Average employee salary (yuan)
    
    # Check if columns exist before calculating statistics
    existing_economic_cols = [col for col in economic_cols_final if col in result_df.columns]
    if existing_economic_cols:
        logger.info(f"\n{result_df[existing_economic_cols].isnull().sum()}")
    else:
        logger.info("Failed to add economic indicator columns.")


    output_file = 'main_data_fix_time_label_new_data_all_2_optimized.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"Results saved to {output_file}")