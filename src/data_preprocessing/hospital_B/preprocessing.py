import pandas as pd
import numpy as np
import sys
import os

# Step 1: Load Tables
admissions = pd.read_csv('./data/MIMIC/raw_data/ADMISSIONS.csv', parse_dates=['ADMITTIME', 'DISCHTIME'])
labevents = pd.read_csv('./data/MIMIC/raw_data/LABEVENTS.csv', parse_dates=['CHARTTIME'])
d_items = pd.read_csv('./data/MIMIC/raw_data/D_LABITEMS.csv')
d_icd_diagnoses = pd.read_csv('./data/MIMIC/raw_data/D_ICD_DIAGNOSES.csv')
diagnoses_icd = pd.read_csv('./data/MIMIC/raw_data/DIAGNOSES_ICD.csv')
patients = pd.read_csv('./data/MIMIC/raw_data/PATIENTS.csv', parse_dates=['DOB', 'DOD'])
d_items_io = pd.read_csv('./data/MIMIC/raw_data/D_ITEMS.csv')

# Step 2: Filter Multiple Admissions
admission_counts = admissions['SUBJECT_ID'].value_counts()
multiple_admissions = admission_counts[admission_counts > 1]
print(f"Number of patients with multiple admission records: {len(multiple_admissions)}")

admissions = admissions[admissions['SUBJECT_ID'].isin(multiple_admissions.index)]
admissions = admissions.sort_values(by=['SUBJECT_ID', 'ADMITTIME']).reset_index(drop=True)
admissions['NEXT_ADMITTIME'] = admissions.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
admissions['TIME_TO_NEXT_ADMISSION'] = (admissions['NEXT_ADMITTIME'] - admissions['DISCHTIME']).dt.days
admissions['TIME_TO_NEXT_ADMISSION'].fillna(9999, inplace=True)
print(admissions[['SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'NEXT_ADMITTIME', 'TIME_TO_NEXT_ADMISSION']].head(10))

# Step 3: Map ITEMID to LAB_NAME
dictionary = d_items[['ITEMID', 'LABEL']]
labevents = labevents.merge(dictionary, on='ITEMID', how='left')
labevents.rename(columns={'LABEL': 'LAB_NAME'}, inplace=True)
missing_lab_names = labevents['LAB_NAME'].isnull().sum()
labevents = labevents[labevents['LAB_NAME'].notnull()]

# Step 4: Clean VALUENUM
labevents['VALUENUM'] = pd.to_numeric(labevents['VALUENUM'], errors='coerce')
labevents = labevents[labevents['VALUENUM'].notnull()]
print(f"Number of records in LABEVENTS after conversion and cleaning: {len(labevents)}")

# Step 5: Aggregate Lab Data
lab_avg = labevents.groupby(['HADM_ID', 'LAB_NAME'])['VALUENUM'].mean().unstack()
lab_last = labevents.sort_values('CHARTTIME').groupby(['HADM_ID', 'LAB_NAME'])['VALUENUM'].last().unstack()
lab_features = pd.concat([lab_avg.add_suffix('_avg'), lab_last.add_suffix('_last')], axis=1)
print(lab_features.head())

# Step 6: Merge Lab Features with Admissions
lab_features.reset_index(inplace=True)
data = admissions.merge(lab_features, on='HADM_ID', how='left')
print(data.head())

# Process Diagnoses
diagnoses = diagnoses_icd.merge(
    d_icd_diagnoses[['ICD9_CODE', 'SHORT_TITLE']],
    on='ICD9_CODE',
    how='left'
)
diagnoses.rename(columns={'SHORT_TITLE': 'DIAGNOSIS_NAME'}, inplace=True)

# --- New Section: Filter target cardiovascular diseases based on ICD9 codes ---
# Define target ICD9 code prefixes:
# Coronary heart disease: 410-414
# Hypertension: 401-405
# Heart failure: 428
# Arrhythmia: 427
# Valvular heart disease: 424
# Cardiomyopathy: 425
# Congenital heart disease: 745-747
# Vascular disease: 443, 440
target_icd_prefixes = ['410', '411', '412', '413', '414',   # Coronary heart disease
                         '401', '402', '403', '404', '405',   # Hypertension
                         '428',                             # Heart failure
                         '427',                             # Arrhythmia
                         '424',                             # Valvular heart disease
                         '425',                             # Cardiomyopathy
                         '745', '746', '747',               # Congenital heart disease
                         '443', '440']                      # Vascular disease

def is_target_icd(icd_code):
    icd_str = str(icd_code)
    for prefix in target_icd_prefixes:
        if icd_str.startswith(prefix):
            return True
    return False

diagnoses = diagnoses[diagnoses['ICD9_CODE'].apply(is_target_icd)]
# --------------------------------------------------------------

missing_diagnoses = diagnoses['DIAGNOSIS_NAME'].isnull().sum()
print(f"\nNumber of missing DIAGNOSIS_NAME: {missing_diagnoses}")

if missing_diagnoses > 0:
    print(f"There are {missing_diagnoses} records with missing DIAGNOSIS_NAME mappings.")
    missing_icd_codes = diagnoses[diagnoses['DIAGNOSIS_NAME'].isnull()]['ICD9_CODE'].unique()
    print(f"Number of ICD9_CODEs with missing DIAGNOSIS_NAME: {len(missing_icd_codes)}")
    print("Examples of missing ICD9_CODEs:", list(missing_icd_codes)[:10])
    diagnoses['DIAGNOSIS_NAME'] = diagnoses['DIAGNOSIS_NAME'].fillna('Unknown')
    remaining_missing = diagnoses['DIAGNOSIS_NAME'].isnull().sum()
    print(f"\nNumber of missing DIAGNOSIS_NAME after filling: {remaining_missing}")
else:
    print("All ICD9_CODEs were successfully mapped to DIAGNOSIS_NAME.")

# Select Top Diagnoses
diagnosis_counts = diagnoses['DIAGNOSIS_NAME'].value_counts()
top_n = 1000
top_diagnoses = diagnosis_counts.nlargest(top_n).index.tolist()

diagnoses_filtered = diagnoses[diagnoses['DIAGNOSIS_NAME'].isin(top_diagnoses) | (diagnoses['DIAGNOSIS_NAME'] == 'Unknown')]
diagnoses_filtered['DIAGNOSIS_NAME'] = diagnoses_filtered['DIAGNOSIS_NAME'].apply(
    lambda x: x if x in top_diagnoses or x == 'Unknown' else 'Other'
)

# One-Hot Encode DIAGNOSIS_NAME
diagnoses_one_hot = pd.get_dummies(diagnoses_filtered[['HADM_ID', 'DIAGNOSIS_NAME']], prefix='DIAGNOSIS_NAME')
diagnoses_agg = diagnoses_one_hot.groupby('HADM_ID').max().reset_index()
print("\nAggregated diagnosis features:")
print(diagnoses_agg.head())

# Merge Diagnoses with Data
data = data.merge(diagnoses_agg, on='HADM_ID', how='left')
diagnosis_feature_columns = diagnoses_agg.columns.tolist()
diagnosis_feature_columns.remove('HADM_ID')
data[diagnosis_feature_columns] = data[diagnosis_feature_columns].fillna(0)
data[diagnosis_feature_columns] = data[diagnosis_feature_columns].astype(int)
print("\nMerged data (ADMISSIONS + LABEVENTS + DIAGNOSES):")
print(data.head())

# Load INPUTEVENTS_CV and INPUTEVENTS_MV
try:
    inputevents_cv = pd.read_csv('./data/MIMIC/raw_data/INPUTEVENTS_CV.csv', parse_dates=['CHARTTIME'])
    print("\nSuccessfully loaded INPUTEVENTS_CV.csv and parsed 'CHARTTIME'.")
except ValueError as e:
    print("\nError reading INPUTEVENTS_CV.csv:", e)
    inputevents_cv = pd.read_csv('./data/MIMIC/raw_data/INPUTEVENTS_CV.csv')
    inputevents_cv.columns = inputevents_cv.columns.str.upper().str.strip()
    if 'CHARTTIME' in inputevents_cv.columns:
        inputevents_cv['CHARTTIME'] = pd.to_datetime(inputevents_cv['CHARTTIME'], errors='coerce')
        print("Column CHARTTIME successfully parsed as datetime type.")
    else:
        print("ERROR: 'CHARTTIME' column does not exist in INPUTEVENTS_CV.csv. Please check the file content.")
        raise

try:
    inputevents_mv = pd.read_csv('./data/MIMIC/raw_data/INPUTEVENTS_MV.csv', parse_dates=['STARTTIME', 'ENDTIME'])
    print("\nSuccessfully loaded INPUTEVENTS_MV.csv and parsed 'STARTTIME' and 'ENDTIME'.")
except ValueError as e:
    print("\nError reading INPUTEVENTS_MV.csv:", e)
    inputevents_mv = pd.read_csv('./data/MIMIC/raw_data/INPUTEVENTS_MV.csv')
    inputevents_mv.columns = inputevents_mv.columns.str.upper().str.strip()
    required_columns_mv = ['STARTTIME', 'ENDTIME']
    missing_columns_mv = [col for col in required_columns_mv if col not in inputevents_mv.columns]
    if missing_columns_mv:
        print(f"ERROR: Missing columns {missing_columns_mv} in INPUTEVENTS_MV.csv. Please check the file content.")
        raise ValueError("Required date columns missing, please check INPUTEVENTS_MV.csv file.")
    else:
        inputevents_mv['STARTTIME'] = pd.to_datetime(inputevents_mv['STARTTIME'], errors='coerce')
        inputevents_mv['ENDTIME'] = pd.to_datetime(inputevents_mv['ENDTIME'], errors='coerce')
        print("STARTTIME and ENDTIME columns successfully parsed as datetime type.")

# Map ITEMID to INPUT_NAME
inputevents_cv = inputevents_cv.merge(d_items_io[['ITEMID', 'LABEL']], on='ITEMID', how='left')
inputevents_cv.rename(columns={'LABEL': 'INPUT_NAME'}, inplace=True)
inputevents_mv = inputevents_mv.merge(d_items_io[['ITEMID', 'LABEL']], on='ITEMID', how='left')
inputevents_mv.rename(columns={'LABEL': 'INPUT_NAME'}, inplace=True)

# Handle Missing INPUT_NAME
missing_input_cv = inputevents_cv['INPUT_NAME'].isnull().sum()
missing_input_mv = inputevents_mv['INPUT_NAME'].isnull().sum()
print(f"\nMissing INPUT_NAME (CV): {missing_input_cv}")
print(f"Missing INPUT_NAME (MV): {missing_input_mv}")

inputevents_cv['INPUT_NAME'] = inputevents_cv['INPUT_NAME'].fillna('Unknown')
inputevents_mv['INPUT_NAME'] = inputevents_mv['INPUT_NAME'].fillna('Unknown')

print(f"Number of CV records after filling missing INPUT_NAME: {len(inputevents_cv)}")
print(f"Number of MV records after filling missing INPUT_NAME: {len(inputevents_mv)}")

# Add SOURCE Column
inputevents_cv['SOURCE'] = 'CV'
inputevents_mv['SOURCE'] = 'MV'

# Create Unified 'EVENTTIME' Column
inputevents_cv.rename(columns={'CHARTTIME': 'EVENTTIME'}, inplace=True)
inputevents_mv.rename(columns={'STARTTIME': 'EVENTTIME'}, inplace=True)

# Concatenate INPUTEVENTS
inputevents = pd.concat([inputevents_cv, inputevents_mv], ignore_index=True)
print("\nFirst few rows of merged INPUTEVENTS table:")
print(inputevents.head())

# Select Top INPUT_NAME
input_name_counts = inputevents['INPUT_NAME'].value_counts()
top_n_inputs = 1000
top_inputs = input_name_counts.nlargest(top_n_inputs).index.tolist()

inputevents_filtered = inputevents[inputevents['INPUT_NAME'].isin(top_inputs) | (inputevents['INPUT_NAME'] == 'Unknown')]
inputevents_filtered['INPUT_NAME'] = inputevents_filtered['INPUT_NAME'].apply(
    lambda x: x if x in top_inputs or x == 'Unknown' else 'Other'
)

# One-Hot Encode INPUT_NAME
input_one_hot = pd.get_dummies(inputevents_filtered[['HADM_ID', 'INPUT_NAME']], prefix='INPUT_NAME')
input_agg = input_one_hot.groupby('HADM_ID').max().reset_index()
print("\nAggregated input features:")
print(input_agg.head())

# Merge Input Features with Data
data = data.merge(input_agg, on='HADM_ID', how='left')
input_feature_columns = input_agg.columns.tolist()
input_feature_columns.remove('HADM_ID')
data[input_feature_columns] = data[input_feature_columns].fillna(0)
data[input_feature_columns] = data[input_feature_columns].astype(int)
print("\nMerged data (ADMISSIONS + DIAGNOSES + INPUTEVENTS):")
print(data.head())

# Load OUTPUTEVENTS
try:
    outputevents = pd.read_csv('./data/MIMIC/raw_data/OUTPUTEVENTS.csv', parse_dates=['CHARTTIME'])
    print("\nSuccessfully loaded OUTPUTEVENTS.csv and parsed 'CHARTTIME'.")
except ValueError as e:
    print("\nError reading OUTPUTEVENTS.csv:", e)
    outputevents = pd.read_csv('./data/MIMIC/raw_data/OUTPUTEVENTS.csv')
    outputevents.columns = outputevents.columns.str.upper().str.strip()
    if 'CHARTTIME' in outputevents.columns:
        outputevents['CHARTTIME'] = pd.to_datetime(outputevents['CHARTTIME'], errors='coerce')
        print("Column CHARTTIME successfully parsed as datetime type.")
    else:
        print("ERROR: 'CHARTTIME' column does not exist in OUTPUTEVENTS.csv. Please check the file content.")
        raise

# Map ITEMID to OUTPUT_NAME
outputevents = outputevents.merge(d_items_io[['ITEMID', 'LABEL']], on='ITEMID', how='left')
outputevents.rename(columns={'LABEL': 'OUTPUT_NAME'}, inplace=True)

# Handle Missing OUTPUT_NAME
missing_output = outputevents['OUTPUT_NAME'].isnull().sum()
print(f"\nNumber of missing OUTPUT_NAME: {missing_output}")
outputevents['OUTPUT_NAME'] = outputevents['OUTPUT_NAME'].fillna('Unknown')
print(f"Number of records after filling missing OUTPUT_NAME: {len(outputevents)}")

# Ensure VALUENUM is Numeric
if 'VALUENUM' in outputevents.columns:
    outputevents['VALUENUM'] = pd.to_numeric(outputevents['VALUENUM'], errors='coerce')
    outputevents = outputevents[outputevents['VALUENUM'].notnull()]
    print(f"Number of OUTPUTEVENTS records after conversion and cleaning: {len(outputevents)}")
else:
    print("INFO: 'VALUENUM' column does not exist in OUTPUTEVENTS.csv or does not need processing.")

# Select Top OUTPUT_NAME
output_name_counts = outputevents['OUTPUT_NAME'].value_counts()
top_n_outputs = 1000
top_outputs = output_name_counts.nlargest(top_n_outputs).index.tolist()

outputevents_filtered = outputevents[outputevents['OUTPUT_NAME'].isin(top_outputs) | (outputevents['OUTPUT_NAME'] == 'Unknown')]
outputevents_filtered['OUTPUT_NAME'] = outputevents_filtered['OUTPUT_NAME'].apply(
    lambda x: x if x in top_outputs or x == 'Unknown' else 'Other'
)

# One-Hot Encode OUTPUT_NAME
output_one_hot = pd.get_dummies(outputevents_filtered[['HADM_ID', 'OUTPUT_NAME']], prefix='OUTPUT_NAME')
output_agg = output_one_hot.groupby('HADM_ID').max().reset_index()
print("\nAggregated output features:")
print(output_agg.head())

# Merge Output Features with Data
data = data.merge(output_agg, on='HADM_ID', how='left')
output_feature_columns = output_agg.columns.tolist()
output_feature_columns.remove('HADM_ID')
data[output_feature_columns] = data[output_feature_columns].fillna(0)
data[output_feature_columns] = data[output_feature_columns].astype(int)
print("\nMerged data (ADMISSIONS + DIAGNOSES + INPUTEVENTS + OUTPUTEVENTS):")
print(data.head())

# Step 5: Add Static Features (PATIENTS)
print("\nFirst few rows of PATIENTS table:")
print(patients.head())

# Merge PATIENTS with Data
data = data.merge(patients[['SUBJECT_ID', 'DOB', 'GENDER']], on='SUBJECT_ID', how='left')
print(f"\nNumber of rows in data after merging PATIENTS table: {len(data)}")
print(data.head())

def calculate_age(admit_time, dob):
    if pd.isnull(admit_time) or pd.isnull(dob):
        return np.nan
    age = admit_time.year - dob.year - ((admit_time.month, admit_time.day) < (dob.month, dob.day))
    return age

data['AGE'] = data.apply(lambda row: calculate_age(row['ADMITTIME'], row['DOB']), axis=1)
print("\nCalculated ages:")
print(data[['ADMITTIME', 'DOB', 'AGE']].head())

# Cap AGE at 120
data['AGE'] = data['AGE'].apply(lambda x: x if 0 <= x <= 120 else np.nan)
print("\nAGE statistics:")
print(data['AGE'].describe())

# Drop Records with NaN AGE
data = data.dropna(subset=['AGE'])
print(f"\nNumber of records after removing rows with NaN AGE: {len(data)}")

# Define Feature Columns
feature_columns = diagnosis_feature_columns + input_feature_columns + output_feature_columns + ['AGE', 'GENDER', 'SUBJECT_ID', 'HADM_ID']

# Check for Missing Features
missing_features = [col for col in feature_columns if col not in data.columns]
if missing_features:
    print(f"\nWarning: The following feature columns do not exist in the dataset and will be removed: {missing_features}")
    feature_columns = [col for col in feature_columns if col not in missing_features]

# Compute Target Variable 'TIME_TO_NEXT_ADMISSION'
data['DISCHTIME'] = pd.to_datetime(data['DISCHTIME'], errors='coerce')
data['NEXT_ADMITTIME'] = pd.to_datetime(data['NEXT_ADMITTIME'], errors='coerce')
data['TIME_TO_NEXT_ADMISSION'] = (data['NEXT_ADMITTIME'] - data['DISCHTIME']).dt.days
data['TIME_TO_NEXT_ADMISSION'] = data['TIME_TO_NEXT_ADMISSION'].fillna(9999).astype(int)

# Define Feature Matrix X and Target Variable y
X = data[feature_columns]
y = data['TIME_TO_NEXT_ADMISSION']

print("\nFirst few rows of feature matrix X:")
print(X.head())
print("\nFirst few rows of target variable y:")
print(y.head())

# Handle Missing Values
missing_values = X.isnull().sum().sum()
print(f"\nTotal number of missing values in feature matrix: {missing_values}")

if missing_values == 0:
    print("Confirmed: No missing values in feature matrix.")
else:
    print(f"Warning: There are {missing_values} missing values in feature matrix.")
    X = X.fillna(X.mean())
    print("Missing values have been filled with means.")

# Map 'GENDER' to Numeric
X['GENDER'] = X['GENDER'].map({'F': 0, 'M': 1})
print("\nConverted GENDER column:")
print(X['GENDER'].head())

# Analyze y Distribution
print("\nDescriptive statistics for target variable y:")
print(y.describe())

# Analyze y Value Counts
print("\nValue distribution of target variable y (top 20 most frequent values):")
print(y.value_counts().head(20))

# Count y=9999
num_no_next_admission = (y == 9999).sum()
print(f"\nNumber of patients with no further admission records (y=9999): {num_no_next_admission}")

# Create Classification Target Variable y_class
y_class = y.apply(lambda x: 1 if x <= 365 else 0)

# Adjust Dataset
X = X.loc[y_class.index]
print(f"\nShape of feature matrix X: {X.shape}")
print(f"Shape of target variable y_class: {y_class.shape}")
print("\nDistribution of target variable y_class:")
print(y_class.value_counts())

# Save feature matrix X
X.to_csv('./data/MIMIC/preprocessed_data/features_X.csv', index=False)
print("Feature matrix X has been saved as 'features_X.csv'.")

# View basic statistics for feature matrix X
print("\nDescriptive statistics for feature matrix X:")
print(X.describe())

from sklearn.preprocessing import StandardScaler

# Identify numeric features (excluding binary features from one-hot encoding)
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Exclude binary features (values are 0 or 1)
binary_cols = [col for col in numerical_cols if X[col].dropna().value_counts().index.isin([0,1]).all()]
numerical_cols = [col for col in numerical_cols if col not in binary_cols]

print(f"\nNumeric feature columns (excluding binary features): {numerical_cols}")

# Standardize numeric features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("\nNumeric features have been standardized.")

# Calculate missing rate for each feature
missing_rates = X.isnull().mean()
print("\nMissing rates for features:")
print(missing_rates.sort_values(ascending=False))

# Define missing rate threshold, assume 50%
missing_threshold = 0.5
cols_to_drop_missing = missing_rates[missing_rates > missing_threshold].index.tolist()

print(f"\nFeatures with missing rate above {missing_threshold*100}%: {cols_to_drop_missing}")

# Remove these features from feature matrix
X = X.drop(columns=cols_to_drop_missing)
print(f"\nShape of feature matrix after removing features with high missing rate: {X.shape}")

# Add target variable to feature matrix to calculate correlation
X_with_target = X.copy()
X_with_target['TARGET'] = y_class

# Calculate correlation coefficients between numeric features and target variable
correlations = X_with_target.corr()['TARGET'].abs().sort_values(ascending=False)

print("\nFeature correlation with target variable (sorted by absolute correlation coefficient):")
print(correlations[1:])  # Exclude 'TARGET' itself

# Define correlation threshold, assume 0.01
correlation_threshold = 0.01
cols_to_drop_corr = correlations[correlations < correlation_threshold].index.tolist()

print(f"\nFeatures with correlation below {correlation_threshold} with target variable: {cols_to_drop_corr}")

# Remove these features from feature matrix
X = X.drop(columns=cols_to_drop_corr)
print(f"\nShape of feature matrix after removing features with very low correlation: {X.shape}")

# Calculate correlation matrix between features
corr_matrix = X.corr().abs()

# Select upper triangle
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find feature pairs with correlation above threshold
redundancy_threshold = 0.95
to_drop = [column for column in upper.columns if any(upper[column] > redundancy_threshold)]

print(f"\nRedundant features (correlation with other features above {redundancy_threshold}): {to_drop}")

# Remove redundant features
X = X.drop(columns=to_drop)
print(f"\nShape of feature matrix after removing redundant features: {X.shape}")

# Calculate number of admissions for each patient
admission_counts = admissions.groupby('SUBJECT_ID')['HADM_ID'].nunique().reset_index()
admission_counts.rename(columns={'HADM_ID': 'ADMISSION_COUNT'}, inplace=True)

# Merge admission counts into dataset
data = data.merge(admission_counts, on='SUBJECT_ID', how='left')

# Add new feature to feature matrix X
X['ADMISSION_COUNT'] = data['ADMISSION_COUNT']

print("\nNew feature 'ADMISSION_COUNT' has been added.")

# Calculate admission time list for each patient
admission_times = admissions.groupby('SUBJECT_ID')['ADMITTIME'].apply(list).reset_index()

# Calculate mean interval between admissions for each patient
def calculate_mean_interval(admit_times):
    admit_times = sorted(admit_times)
    intervals = [(admit_times[i+1] - admit_times[i]).days for i in range(len(admit_times)-1)]
    if intervals:
        return np.mean(intervals)
    else:
        return np.nan

admission_times['MEAN_INTERVAL'] = admission_times['ADMITTIME'].apply(calculate_mean_interval)

# Merge mean admission interval into dataset
data = data.merge(admission_times[['SUBJECT_ID', 'MEAN_INTERVAL']], on='SUBJECT_ID', how='left')

# Add new feature to feature matrix X
X['MEAN_INTERVAL'] = data['MEAN_INTERVAL']

print("\nNew feature 'MEAN_INTERVAL' has been added.")

# Fill missing values for 'MEAN_INTERVAL'
X['MEAN_INTERVAL'] = X['MEAN_INTERVAL'].fillna(-1)
print("\nMissing values for 'MEAN_INTERVAL' have been filled.")

# Calculate number of diagnoses for each admission
diagnosis_counts = diagnoses_icd.groupby('HADM_ID').size().reset_index(name='DIAGNOSIS_COUNT')

# Merge diagnosis counts into dataset
data = data.merge(diagnosis_counts, on='HADM_ID', how='left')

# Add new feature to feature matrix X
X['DIAGNOSIS_COUNT'] = data['DIAGNOSIS_COUNT']

print("\nNew feature 'DIAGNOSIS_COUNT' has been added.")

# Fill missing values (some admission records may not have diagnosis information)
X['DIAGNOSIS_COUNT'] = X['DIAGNOSIS_COUNT'].fillna(0)

print("\nDescriptive statistics for new features:")
print(X[['ADMISSION_COUNT', 'MEAN_INTERVAL', 'DIAGNOSIS_COUNT']].describe())

# Standardize new features
X[['ADMISSION_COUNT', 'MEAN_INTERVAL', 'DIAGNOSIS_COUNT']] = scaler.fit_transform(X[['ADMISSION_COUNT', 'MEAN_INTERVAL', 'DIAGNOSIS_COUNT']])

print("\nNew features have been standardized.")

# Save processed feature matrix X and target variable y_class
X.to_csv('./data/MIMIC/preprocessed_data/processed_features_X.csv', index=False)
y_class.to_csv('./data/MIMIC/preprocessed_data/processed_target_y.csv', index=False)

print("Processed feature matrix X has been saved as 'processed_features_X.csv'.")
print("Processed target variable y_class has been saved as 'processed_target_y.csv'.")


# Step 1: Load data
def load_data(X_path='./data/MIMIC/preprocessed_data/processed_features_X.csv', y_path='./data/MIMIC/preprocessed_data/processed_target_y.csv'):
    print("Loading data...")
    try:
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).squeeze()
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    print(f"Shape of feature matrix X: {X.shape}")
    print(f"Shape of target variable y: {y.shape}")
    
    # Fill missing values for 'ADMISSION_COUNT' column
    if 'ADMISSION_COUNT' in X.columns:
        X['ADMISSION_COUNT'] = X['ADMISSION_COUNT'].fillna(1)
        print(f"Number of missing values in 'ADMISSION_COUNT' column in feature matrix X: {X['ADMISSION_COUNT'].isnull().sum()}")
    else:
        print("Warning: 'ADMISSION_COUNT' column does not exist in feature matrix X.")
    
    # Convert all feature names to string type
    X.columns = X.columns.astype(str)
    
    # Define illegal characters to remove
    illegal_chars = ['[', ']', '<', '>', '(', ')', '{', '}', '@', '#', '$', '%', '^', '&', '*',
                    '+', '=', '|', '\\', '/', '?', ',', '.', ';', ':', '"', "'", '`', '~', ' ']
    
    # Define a function to clean feature names
    def clean_column_name(col_name):
        for char in illegal_chars:
            col_name = col_name.replace(char, '_')
        return col_name
    
    # Clean feature names
    cleaned_columns = [clean_column_name(col) for col in X.columns]
    
    # Ensure column names are unique, add suffix if duplicated
    def make_unique(columns):
        seen = {}
        unique_columns = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                unique_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_columns.append(col)
        return unique_columns
    
    unique_cleaned_columns = make_unique(cleaned_columns)
    X.columns = unique_cleaned_columns
    
    print("Feature names have been cleaned and uniqueness ensured.")
    
    # Delete duplicate columns, keep the first occurrence (double check)
    duplicate_columns = X.columns[X.columns.duplicated()].tolist()
    if duplicate_columns:
        print(f"Found duplicate columns: {duplicate_columns}. Removing duplicate columns...")
        X = X.loc[:, ~X.columns.duplicated()]
        print(f"Shape of feature matrix X after removing duplicate columns: {X.shape}")
    else:
        print("No duplicate columns found.")
    
    # Verify all column names are unique
    if X.columns.duplicated().any():
        print("Error: There are still duplicate column names after cleaning. Please check the cleaning process.")
        print("Duplicate column names are:", X.columns[X.columns.duplicated()].tolist())
        sys.exit(1)
    else:
        print("All column names are unique.")
    
    return X, y

# save the data
X, y = load_data()
X.to_csv('./data/MIMIC/preprocessed_data/data_X_for_model.csv', index=False)
y.to_csv('./data/MIMIC/preprocessed_data/data_y_for_model.csv', index=False)

def load_medical_records(file_path):
    """
    Load medical record information file and select specific feature columns
    
    Selected feature columns:
    - Medical record number
    - Gender
    - Age
    - Address_1
    - Admission department
    - Discharge department
    - Main diagnosis
    - Other diagnosis
    - Other diagnosis_2
    """
    try:
        df = pd.read_csv(file_path)
        selected_columns = [
            '病案号', '性别', '年龄', '地址_1', 
            '入院科室', '出院科室', '主要诊断', 
            '其他诊断', '其他诊断_2'
        ]
        return df[selected_columns]
    except Exception as e:
        print(f"Error loading medical record information file: {str(e)}")
        return None

def load_data_mark(file_path):
    """
    Load data mark file and select specific feature columns
    
    Selected feature columns:
    - Hospital number
    - Date
    - Height
    - Weight
    - Item name
    - Range
    - Examination result
    """
    try:
        df = pd.read_csv(file_path)
        selected_columns = [
            '住院号', '日期', '身高', '体重',
            '项目名称', '范围', '检查结果'
        ]
        return df[selected_columns]
    except Exception as e:
        print(f"Error loading data mark file: {str(e)}")
        return None

def clean_medical_records(df):
    """Clean medical record data"""
    if df is None:
        return None
    
    # Remove duplicate records
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna({
        '性别': 'Unknown',
        '年龄': df['年龄'].median(),
        '地址_1': 'Unknown',
        '入院科室': 'Unknown',
        '出院科室': 'Unknown',
        '主要诊断': 'Unknown',
        '其他诊断': 'Unknown',
        '其他诊断_2': 'Unknown'
    })
    
    return df

def clean_data_mark(df):
    """Clean data mark data"""
    if df is None:
        return None
    
    # Remove duplicate records
    df = df.drop_duplicates()
    
    # Convert date column to datetime type
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    
    # Ensure numeric columns are numeric type
    numeric_columns = ['身高', '体重']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    df = df.fillna({
        '身高': df['身高'].median(),
        '体重': df['体重'].median(),
        '项目名称': 'Unknown',
        '范围': 'Unknown',
        '检查结果': 'Unknown'
    })
    
    return df

def preprocess_stage1(medical_record_path, data_mark_path):
    """
    Stage 1 data preprocessing
    
    Parameters:
        medical_record_path: Path to medical record information file
        data_mark_path: Path to data mark file
    
    Returns:
        tuple: (Processed medical record data, Processed data mark data)
    """
    # Load data
    medical_records = load_medical_records(medical_record_path)
    data_mark = load_data_mark(data_mark_path)
    
    # Clean data
    medical_records = clean_medical_records(medical_records)
    data_mark = clean_data_mark(data_mark)
    
    return medical_records, data_mark

if __name__ == "__main__":
    # Set input and output paths
    medical_record_path = "./data/MIMIC/raw_data/medical_records.csv"
    data_mark_path = "./data/MIMIC/raw_data/data_mark.csv"
    output_dir = "./data/MIMIC/preprocessed_data"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process data
    medical_records, data_mark = preprocess_stage1(medical_record_path, data_mark_path)
    
    # Save processed data
    if medical_records is not None and data_mark is not None:
        medical_records.to_csv(f"{output_dir}/processed_medical_records.csv", index=False)
        data_mark.to_csv(f"{output_dir}/processed_data_mark.csv", index=False)
        print("Stage 1 data preprocessing completed!")
        print(f"Processed files have been saved to: {output_dir}")
    else:
        print("Error occurred during data preprocessing")
