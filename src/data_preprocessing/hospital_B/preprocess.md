# MIMIC Data Preprocessing Pipeline - Hospital B

This file ([data_preprocessing/hospital_B/preprocess_mimic.py](data_preprocessing/hospital_B/preprocess_.py)) describes the preprocessing steps for the MIMIC-III database, aiming to provide clean, usable datasets for subsequent machine learning models. The following are detailed explanations for each step:

## 1. Data Loading

*   **Objective:** Load required MIMIC-III tables from CSV files.
*   **Method:** Use the `read_csv` function from the `pandas` library to read the following files:
    *   `ADMISSIONS.csv`: Admission records
    *   `LABEVENTS.csv`: Laboratory events
    *   `D_LABITEMS.csv`: Laboratory item definitions
    *   `D_ICD_DIAGNOSES.csv`: ICD diagnosis code definitions
    *   `DIAGNOSES_ICD.csv`: Patient diagnosis codes
    *   `PATIENTS.csv`: Patient information
    *   `D_ITEMS.csv`: Input/output item definitions
*   **Result:** Each CSV file is loaded into a `pandas` DataFrame. The `ADMITTIME`, `DISCHTIME`, `DOB`, `DOD` columns are parsed as datetime format.

## 2. Filtering Multiple Admission Records

*   **Objective:** Filter patients with multiple hospital admission records.
*   **Method:**
    *   Calculate the number of admissions for each `SUBJECT_ID`.
    *   Select patients with more than one admission.
    *   Calculate the time of next admission and fill in missing values.
*   **Result:** Obtain an `admissions` DataFrame containing only patients with multiple admission records, and calculate `TIME_TO_NEXT_ADMISSION`.

## 3. Mapping ITEMID to LAB_NAME

*   **Objective:** Map `ITEMID` in the `LABEVENTS` table to the corresponding `LAB_NAME`.
*   **Method:**
    *   Use the `D_LABITEMS.csv` table to map `ITEMID` to `LABEL` (renamed as `LAB_NAME`).
*   **Result:** The `labevents` DataFrame contains a `LAB_NAME` column representing the name of laboratory items.

## 4. Cleaning VALUENUM

*   **Objective:** Clean the `VALUENUM` column in the `LABEVENTS` table to ensure it is a numeric type.
*   **Method:**
    *   Use `pd.to_numeric` to convert `VALUENUM` to a numeric type, and set values that cannot be converted to `NaN`.
    *   Remove records where `VALUENUM` is `NaN`.
*   **Result:** The `labevents` DataFrame contains a cleaned numeric `VALUENUM` column.

## 5. Aggregating Laboratory Data

*   **Objective:** Aggregate laboratory data for each admission record.
*   **Method:**
    *   Calculate the mean and last measurement value of `VALUENUM` for each `HADM_ID` and `LAB_NAME`.
    *   Concatenate the mean and last values into a wide table.
*   **Result:** The `lab_features` DataFrame contains the average laboratory values and the last laboratory values for each admission record.

## 6. Merging Laboratory Features with Admission Records

*   **Objective:** Merge laboratory features into admission records.
*   **Method:**
    *   Use `HADM_ID` to merge the `lab_features` DataFrame into the `admissions` DataFrame.
*   **Result:** The `data` DataFrame contains admission records and laboratory features.

## 7. Processing Diagnosis Information

*   **Objective:** Process diagnosis information, including handling missing values, selecting Top N diagnoses, and performing One-Hot encoding.
*   **Method:**
    *   Merge the `DIAGNOSES_ICD` table with the `D_ICD_DIAGNOSES` table to obtain diagnosis names.
    *   Handle missing diagnosis names.
    *   Select the most common N diagnoses.
    *   Perform One-Hot encoding on diagnosis names.
    *   Merge diagnosis features into the `data` DataFrame.
*   **Result:** The `data` DataFrame contains admission records and diagnosis features.

## 8. Loading and Processing Infusion Event Data (INPUTEVENTS_CV and INPUTEVENTS_MV)

*   **Objective**: Load, clean, and merge `INPUTEVENTS_CV.csv` and `INPUTEVENTS_MV.csv` data, handle missing values, and perform One-Hot encoding.
*   **Method**:
    *   Load `INPUTEVENTS_CV.csv` and `INPUTEVENTS_MV.csv`, parsing date columns.
    *   Map `ITEMID` to `INPUT_NAME`.
    *   Handle missing `INPUT_NAME` values.
    *   Merge the two infusion event tables.
    *   Select the most common N infusion names.
    *   Perform One-Hot encoding on infusion names.
    *   Merge infusion features into the `data` DataFrame.
*   **Result**: The `data` DataFrame contains admission records and infusion event features.

## 9. Loading and Processing Output Event Data (OUTPUTEVENTS)

*   **Objective**: Load, clean, and process `OUTPUTEVENTS.csv` data, handle missing values, and perform One-Hot encoding.
*   **Method**:
    *   Load `OUTPUTEVENTS.csv`, parsing date columns.
    *   Map `ITEMID` to `OUTPUT_NAME`.
    *   Handle missing `OUTPUT_NAME` values.
    *   Ensure `VALUENUM` is a numeric type.
    *   Select the most common N output names.
    *   Perform One-Hot encoding on output names.
    *   Merge output features into the `data` DataFrame.
*   **Result**: The `data` DataFrame contains admission records and output event features.

## 10. Adding Static Features (PATIENTS)

*   **Objective:** Merge static patient information (age, gender).
*   **Method:**
    *   Merge the `PATIENTS` table with the `data` DataFrame.
    *   Calculate patient age.
    *   Handle abnormal age values.
*   **Result:** The `data` DataFrame contains patient age and gender information.

## 11. Feature Engineering and Target Variable Definition

*   **Objective:** Define feature matrix `X` and target variable `y`.
*   **Method:**
    *   Select feature columns.
    *   Handle missing values.
    *   Map `GENDER` to numeric values.
    *   Create categorical target variable `y_class`, indicating whether readmission occurs within 365 days.
*   **Result:** Obtain feature matrix `X` and target variable `y_class`.

## 12. Feature Selection and Standardization

*   **Objective:** Further clean and transform the feature matrix `X`.
*   **Method:**
    *   Save feature matrix X
    *   Identify numeric features and standardize them.
    *   Calculate the missing rate for each feature and remove features with high missing rates.
    *   Calculate the correlation between features and the target variable, and remove features with extremely low correlation.
    *   Calculate the correlation matrix between features and remove redundant features.
    *   Calculate the number of admissions and average admission interval for each patient, and add them to the feature matrix.
    *   Fill in missing values.
    *   Standardize new features.
*   **Result:** Obtain the final feature matrix `X` and target variable `y_class`, which are ready for model training.

## 13. Saving Processed Data

*   **Objective:** Save the processed feature matrix `X` and target variable `y_class`.
*   **Method:**
    *   Use the `to_csv` function to save `X` and `y_class` as CSV files.
*   **Result:** Generate `processed_features_X.csv` and `processed_target_y.csv` files.

