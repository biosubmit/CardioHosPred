#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Series Hospital Prediction Model Analysis Module.

This module splits data by inspection time in half-year steps,
trains models within each time period, generates ROC curve data
and feature importance analysis, and saves results to dedicated folders.
It also provides functionality for evaluating the model on the full dataset.
"""

import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from loguru import logger
from datetime import datetime, timedelta
import warnings
from typing import Tuple, Dict, List, Optional, Any
from scipy import stats

# Configure logging
logger.add("logs/time_series_analysis_en.log", rotation="500 MB") # Changed log file name for clarity

# Suppress specific pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def create_analysis_dirs() -> None:
    """
    Create the directory structure required for analysis results.
    
    Creates folders for saving ROC curves and feature importance analysis results.
    """
    dirs = [
        '../data/hospital_A/time_based_analysis/roc_curves',
        '../data/hospital_A/time_based_analysis/feature_importance/data',
        '../data/hospital_A/time_based_analysis/feature_importance/translated',
        '../data/hospital_A/full_data_analysis'  # Add directory for full data analysis
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created or already exists: {directory}")

def create_special_time_based_dirs() -> None:
    """
    Create the directory structure required for special time period analysis.
    
    Creates folders for saving results of special time period analysis.
    Note: This function defines paths but build_model_for_timeframe uses hardcoded paths.
    If special paths are needed, build_model_for_timeframe would need adjustment.
    """
    dirs = [
        '../data/hospital_A/special_time_based_analysis/roc_curves',
        '../data/hospital_A/special_time_based_analysis/feature_importance/data',
        '../data/hospital_A/special_time_based_analysis/feature_importance/translated',
        '../data/hospital_A/special_time_based_analysis/full_data_analysis'
    ]
    for dir_path in dirs: # Renamed 'dir' to 'dir_path' to avoid conflict with built-in
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory created or already exists: {dir_path}")



def process_data() -> pd.DataFrame:
    """
    Load and process data.
    
    Loads the dataset, handles extreme values (inf), and encodes categorical columns.
    
    Returns:
        pd.DataFrame: Processed dataset.
    """
    logger.info("Loading data...")
    data = pd.read_csv('../data/hospital_A/tmp_preprocessed_data/final_preprocessed_data.csv')
    
    # Handle extreme values, replace inf/-inf with nan
    logger.info("Handling extreme values...")
    data = data.replace([np.inf, -np.inf], np.nan)
    remove_col = [
        '人均地区生产总值(元)', # Per capita regional GDP (Yuan)
        '医院、卫生院数(个)',   # Number of hospitals and health centers (count)
        '医院、卫生院床位数(张)', # Number of beds in hospitals and health centers (count)
        '医生数(人)',       # Number of doctors (count)
        '职工平均工资(元)',   # Average employee salary (Yuan)
        '出生日期',       # Date of Birth
        '入院科室',       # Admission Department
        '出院科室',       # Discharge Department
        '主要诊断',       # Main Diagnosis
        '入院时间',       # Admission Time
        '出院时间',       # Discharge Time
        '上次诊断',       # Previous Diagnosis
        '上次入院科室',   # Previous Admission Department
        '上次出院科室',   # Previous Discharge Department
        '上次入院时间',   # Previous Admission Time
        '上次出院时间'    # Previous Discharge Time
    ]
    data.drop(columns=remove_col, inplace=True, errors='ignore') # Added errors='ignore' for robustness


    # Specify some columns as categorical
    # '性别' (Gender)
    if '性别' in data.columns:
        data['性别'] = data['性别'].astype('category').cat.codes # Gender
    
    logger.info("Encoding categorical columns...")
    # This will encode all remaining 'category' dtype columns.
    # If other object columns need encoding, they should be converted to 'category' first
    # or handled explicitly.
    cat_cols = data.select_dtypes(include=['category']).columns


    for col in cat_cols:
        if col != '病案号':  # '病案号' (Medical Record Number) - Keep Medical Record Number as ID
            data[col] = data[col].astype('category').cat.codes
            logger.debug(f"Column '{col}' has been categorically encoded.")

    # Ensure '检查时间' (Inspection_Time) is in datetime format
    if '检查时间' in data.columns: # '检查时间' (Inspection_Time)
        try:
            data['检查时间'] = pd.to_datetime(data['检查时间']) # Inspection_Time
            logger.info("'Inspection_Time' column converted to datetime format.")
        except Exception as e:
            logger.error(f"Error converting 'Inspection_Time' to datetime format: {str(e)}")
    else:
        logger.error("Column '检查时间' ('Inspection_Time') not found in data.") # '检查时间' (Inspection_Time)
    
    logger.info(f"Data processing complete. {len(data)} rows, {len(data.columns)} columns.")
    return data

def calculate_auc_ci(y_true: np.ndarray, y_score: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate AUC value and its confidence interval.
    
    Args:
        y_true: True labels.
        y_score: Predicted scores.
        confidence: Confidence level, default is 0.95 for 95% CI.
        
    Returns:
        Tuple[float, float, float]: (AUC value, CI lower bound, CI upper bound).
    """
    # Ensure inputs are numpy arrays to avoid pandas index issues
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
        
    fpr_original, tpr_original, _ = roc_curve(y_true, y_score)
    auc_value = auc(fpr_original, tpr_original)
    
    # Calculate AUC confidence interval using Bootstrap method
    n_bootstraps = 1000
    bootstrapped_aucs = []
    
    rng = np.random.RandomState(42)  # Set random seed for reproducibility
    
    for i in range(n_bootstraps):
        # Random sampling with replacement
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # If bootstrap sample has only one class, skip this iteration
            continue
        
        fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_score[indices])
        bootstrapped_aucs.append(auc(fpr_boot, tpr_boot))
    
    if not bootstrapped_aucs: # Handle case where all bootstrap samples had one class
        logger.warning("Could not compute bootstrapped AUCs (all samples had one class). CI will be N/A.")
        return auc_value, np.nan, np.nan

    # Calculate confidence interval
    alpha = (1.0 - confidence) / 2.0
    lower_bound = max(0.0, np.percentile(bootstrapped_aucs, 100 * alpha))
    upper_bound = min(1.0, np.percentile(bootstrapped_aucs, 100 * (1 - alpha)))
    
    return auc_value, lower_bound, upper_bound

def calculate_roc_ci(y_true: np.ndarray, y_score: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence interval for each point on the ROC curve.
    
    Args:
        y_true: True labels.
        y_score: Predicted scores.
        confidence: Confidence level, default is 0.95 for 95% CI.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            (mean_fpr, mean_tpr, tpr_lower_bound, tpr_upper_bound, original_thresholds).
    """
    # Ensure inputs are numpy arrays to avoid pandas index issues
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    
    # Calculate original ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Calculate ROC curve confidence interval using Bootstrap method
    n_bootstraps = 200  # Reduce iterations for performance (can be increased for more precision)
    rng = np.random.RandomState(42)  # Set random seed for reproducibility
    
    # Create a standard FPR space for interpolation
    all_fprs_interp_grid = np.linspace(0, 1, 100) # Standard grid for FPR
    
    bootstrap_tprs_interp = []
    
    for i in range(n_bootstraps):
        # Random sampling with replacement
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # If bootstrap sample has only one class, skip this iteration
            continue
        
        # Calculate ROC curve for bootstrap sample
        y_true_bootstrap = y_true[indices]
        y_score_bootstrap = y_score[indices]
        fpr_bootstrap, tpr_bootstrap, _ = roc_curve(y_true_bootstrap, y_score_bootstrap)
        
        # Use linear interpolation to standard FPR space
        tpr_interp = np.interp(all_fprs_interp_grid, fpr_bootstrap, tpr_bootstrap)
        tpr_interp[0] = 0.0  # Ensure starting from the origin
        
        bootstrap_tprs_interp.append(tpr_interp)
    
    if not bootstrap_tprs_interp: # Handle case where all bootstrap samples had one class
        logger.warning("Could not compute bootstrapped TPRs for ROC CI. Returning original ROC without CI.")
        # Return original ROC points padded to match expected output structure if CI calculation fails
        return fpr, tpr, tpr, tpr, thresholds 


    # Calculate mean and confidence interval for TPR
    tpr_mean_interp = np.mean(bootstrap_tprs_interp, axis=0)
    # tpr_std_interp = np.std(bootstrap_tprs_interp, axis=0) # Standard deviation can also be useful
    
    # Calculate confidence interval
    alpha = (1.0 - confidence) / 2.0
    tpr_lower_bound_interp = np.percentile(bootstrap_tprs_interp, 100 * alpha, axis=0)
    tpr_upper_bound_interp = np.percentile(bootstrap_tprs_interp, 100 * (1 - alpha), axis=0)
    
    return all_fprs_interp_grid, tpr_mean_interp, tpr_lower_bound_interp, tpr_upper_bound_interp, thresholds # Return original thresholds

def translate_feature_names(feature_file_path: str, translation_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Translate feature names in the feature importance file.
    
    Args:
        feature_file_path: Feature importance file path.
        translation_dict: Feature name translation dictionary. If None, uses default translation.
        
    Returns:
        str: Path to the translated file. Returns original path if translation fails or is not applicable.
    """
    # First, load the feature file
    try:
        feature_df = pd.read_csv(feature_file_path)
    except FileNotFoundError:
        logger.error(f"Feature file not found: {feature_file_path}")
        return feature_file_path # Return original path

    # If no translation dictionary is provided, load from default location
    if translation_dict is None:
        import json
        json_path = '../config/colname_translation.json'
        
        if not os.path.exists(json_path):
            logger.warning(f"Translation file not found: {json_path}. Original feature names will be used.")
            return feature_file_path
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                translation_dict = json.load(f)
            if translation_dict:  # Ensure dictionary is not None
                logger.info(f"Loaded translation dictionary from {json_path}, containing {len(translation_dict)} features.")
            else:
                logger.warning(f"Translation dictionary loaded from {json_path} is empty.")
        except Exception as e:
            logger.error(f"Error loading translation dictionary from {json_path}: {e}")
            return feature_file_path
            
    # Ensure 'feature' column exists
    if 'feature' not in feature_df.columns:
        logger.error(f"Column 'feature' not found in feature file {feature_file_path}.")
        return feature_file_path  # Return original path
    
    # Apply translation
    if translation_dict:  # Ensure dictionary is not None or empty before using
        feature_df['feature_translated'] = feature_df['feature'].apply( # Store in new column to preserve original
            lambda x: translation_dict.get(x, x)  # If no corresponding translation, keep original name
        )
    else:
        logger.info("No translation dictionary provided or loaded, using original feature names.")
        feature_df['feature_translated'] = feature_df['feature'] # Copy to new column
    
    # Generate path for the translated file
    base_dir = os.path.dirname(feature_file_path)
    file_name = os.path.basename(feature_file_path)
    
    # Adjust base_dir for translated files (example: from 'data' to 'translated' subdirectory)
    # Original code replaces '/data/' with '/translated/'. This assumes 'data' is part of path.
    # A more robust way might be to ensure the 'translated' subdir exists relative to the 'data' subdir.
    translated_dir = os.path.join(os.path.dirname(base_dir), 'translated') # Assumes 'data' is parent of 'feature_importance'
    if 'feature_importance/data' in base_dir: # Specific to this structure
         translated_dir = base_dir.replace('/feature_importance/data', '/feature_importance/translated')
    else: # Fallback or general case
         translated_dir = os.path.join(base_dir, 'translated')


    translated_path = os.path.join(translated_dir, f"translated_{file_name}")
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(translated_path), exist_ok=True)
    
    # Save the file with original and translated feature names
    feature_df.to_csv(translated_path, index=False)
    logger.info(f"Feature names (with translations) saved to {translated_path}")
    
    return translated_path
        


def build_model_for_timeframe(
    data: pd.DataFrame, 
    start_date: datetime, 
    end_date: datetime, 
    timeframe_name: str,
    output_base_dir: str = '../data/hospital_A/time_based_analysis' # Added for flexibility
) -> Tuple[Optional[RandomForestClassifier], float, pd.DataFrame]:
    """
    Build and evaluate a model for a specific time frame.
    
    Args:
        data: Preprocessed data.
        start_date: Time frame start date.
        end_date: Time frame end date.
        timeframe_name: Name for the time frame, used in output file names.
        output_base_dir: Base directory for saving results.
        
    Returns:
        tuple: (Trained model, accuracy, feature importance DataFrame).
    """
    logger.info(f"Preparing model data for time frame: {timeframe_name} ({start_date.date()} to {end_date.date()})")
    
    # Initialize variables to avoid "potentially unbound" errors
    X_train, X_test, y_train, y_test = None, None, None, None
    
    # Filter data for the time frame
    timeframe_data = data[(data['检查时间'] >= start_date) & (data['检查时间'] < end_date)].copy() # '检查时间' (Inspection_Time)
    if len(timeframe_data) == 0:
        logger.warning(f"No data for time frame {timeframe_name}.")
        return None, 0.0, pd.DataFrame()
    
    logger.info(f"Data volume for time frame {timeframe_name}: {len(timeframe_data)} rows.")
    
    # Clean problematic values in data
    timeframe_data = timeframe_data.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values for all numerical columns
    numeric_cols = timeframe_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if timeframe_data[col].isna().any():
            # Fill with median, or -1 if median cannot be computed (e.g., all NaNs)
            median_val = timeframe_data[col].median()
            timeframe_data[col] = timeframe_data[col].fillna(median_val if pd.notna(median_val) else -1)
    
    # Process categorical columns (ensure they are codes)
    cat_cols = timeframe_data.select_dtypes(include=['category', 'object']).columns
    for col in cat_cols:
        if col != '病案号':  # '病案号' (Medical Record Number) - Keep Medical Record Number as ID
            timeframe_data[col] = timeframe_data[col].astype('category').cat.codes
    
    # Process '时间差' (Time_Difference) for binary classification (1 if > 0, else 0)
    if '时间差' in timeframe_data.columns: # '时间差' (Time_Difference)
        timeframe_data['时间差_binary'] = timeframe_data['时间差'].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0) # '时间差_binary' (Binary Time_Difference) from '时间差' (Time_Difference)
        logger.info(f"Distribution of binary 'Time_Difference' for {timeframe_name}: {timeframe_data['时间差_binary'].value_counts(dropna=False)}")
        target_col_name = '时间差_binary'
    else:
        logger.error(f"Target variable '时间差' ('Time_Difference') not found in data for {timeframe_name}.")
        return None, 0.0, pd.DataFrame()
    
    # Prepare features (X) and target (y)
    # Drop original '时间差', '检查时间', and '病案号' (if present and used for grouping) from features
    features_to_drop = [target_col_name, '时间差', '检查时间'] # '时间差' (Time_Difference), '检查时间' (Inspection_Time)
    X = timeframe_data.drop(columns=[col for col in features_to_drop if col in timeframe_data.columns], errors='ignore')
    y = timeframe_data[target_col_name]
    
    # Group split based on '病案号' (Medical Record Number)
    groups = X['病案号'] if '病案号' in X.columns else None # '病案号' (Medical Record Number)
    
    if groups is not None:
        # Remove '病案号' from X before training, keep it for splitting
        X_features = X.drop(columns=['病案号'], errors='ignore') # '病案号' (Medical Record Number)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        try:
            train_idx, test_idx = next(gss.split(X_features, y, groups=groups))
            X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        except ValueError as e: # Handles cases like too few groups or samples per group
            logger.warning(f"GroupShuffleSplit failed for {timeframe_name}: {e}. Falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)
    else:
        X_features = X # No '病案号' (Medical Record Number) to drop
        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)
    
    del timeframe_data # Release memory
    gc.collect()
    
    if X_train is None or X_train.empty or y_train is None or y_train.empty:
        logger.error(f"Training data is empty for time frame {timeframe_name}.")
        return None, 0.0, pd.DataFrame()
    if len(y_train.unique()) < 2:
        logger.error(f"Training target for {timeframe_name} has only one class. Cannot train classifier.")
        return None, 0.0, pd.DataFrame()

    # Train model
    logger.info(f"Training model for time frame {timeframe_name}...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=0, # Set to 1 or higher for more verbosity during training
        max_depth=10,
        class_weight='balanced' # Added due to potential imbalance
    )
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error during model training for {timeframe_name}: {str(e)}")
        # Attempt with higher precision float type as a fallback (original code had this)
        try:
            logger.info(f"Retrying training for {timeframe_name} with float64 data type.")
            X_train_fp64 = X_train.astype(np.float64)
            model.fit(X_train_fp64, y_train)
        except Exception as e2:
            logger.error(f"Second training attempt failed for {timeframe_name}: {str(e2)}")
            return None, 0.0, pd.DataFrame()
    
    # Evaluate model
    logger.info(f"Evaluating model for time frame {timeframe_name}...")
    accuracy = 0.0
    feature_importance_df = pd.DataFrame()
    
    if X_test is None or X_test.empty or y_test is None or y_test.empty:
        logger.error(f"Test data is empty for time frame {timeframe_name}. Cannot evaluate.")
        return model, 0.0, pd.DataFrame() # Return trained model, but no accuracy
    if len(y_test.unique()) < 2:
        logger.error(f"Test target for {timeframe_name} has only one class. ROC/AUC metrics will be problematic.")
        # Accuracy can still be calculated
        y_pred_simple = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_simple)
        logger.info(f"Model accuracy (on single-class test set) for {timeframe_name}: {accuracy:.4f}")
        return model, accuracy, pd.DataFrame() # Return model and accuracy, but no ROC/feature importance

    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy for {timeframe_name}: {accuracy:.4f}")
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate AUC and its 95% CI
        roc_auc, auc_ci_lower, auc_ci_upper = calculate_auc_ci(y_test, y_pred_proba)
        logger.info(f"AUC for {timeframe_name}: {roc_auc:.4f} (95% CI: {auc_ci_lower:.4f}-{auc_ci_upper:.4f})")

        # Calculate ROC curve points with CIs
        std_fpr, mean_tpr, tpr_lower, tpr_upper, original_thresholds = calculate_roc_ci(y_test, y_pred_proba)
        
        # Save ROC curve data with CIs (interpolated)
        roc_data_interpolated = pd.DataFrame({
            'fpr_interpolated': std_fpr,
            'tpr_mean_interpolated': mean_tpr,
            'tpr_lower_ci': tpr_lower,
            'tpr_upper_ci': tpr_upper,
            'auc': roc_auc, # Add overall AUC and CI to this file
            'auc_ci_lower': auc_ci_lower,
            'auc_ci_upper': auc_ci_upper
        })
        
        roc_dir = os.path.join(output_base_dir, 'roc_curves')
        os.makedirs(roc_dir, exist_ok=True)
        roc_data_path_interpolated = os.path.join(roc_dir, f'roc_curve_interpolated_ci_{timeframe_name}.csv')
        roc_data_interpolated.to_csv(roc_data_path_interpolated, index=False)
        logger.info(f"Interpolated ROC curve data with CIs for {timeframe_name} saved to {roc_data_path_interpolated}")

        # Save original (non-interpolated) ROC data
        fpr_orig, tpr_orig, thresholds_orig = roc_curve(y_test, y_pred_proba) # Recalculate for clarity
        original_roc_data = pd.DataFrame({
            'fpr_original': fpr_orig,
            'tpr_original': tpr_orig,
            'thresholds_original': thresholds_orig
        })
        original_roc_data_path = os.path.join(roc_dir, f'roc_curve_original_points_{timeframe_name}.csv')
        original_roc_data.to_csv(original_roc_data_path, index=False)
        logger.info(f"Original ROC curve points for {timeframe_name} saved to {original_roc_data_path}")
        
        # Calculate feature importance
        importances = model.feature_importances_
        std_importances = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns[indices],
            'importance': importances[indices],
            'std_dev_importance': std_importances[indices]
        })
        
        fi_data_dir = os.path.join(output_base_dir, 'feature_importance', 'data')
        os.makedirs(fi_data_dir, exist_ok=True)
        importance_data_path = os.path.join(fi_data_dir, f'feature_importance_{timeframe_name}.csv')
        feature_importance_df.to_csv(importance_data_path, index=False)
        logger.info(f"Feature importance data for {timeframe_name} saved to {importance_data_path}")
        
        # Translate feature names and save
        # Pass the correct base directory for translation output
        fi_translated_dir = os.path.join(output_base_dir, 'feature_importance', 'translated')
        os.makedirs(fi_translated_dir, exist_ok=True)
        # Custom logic for translate_feature_names pathing might be needed if it's too rigid
        translated_path = translate_feature_names(importance_data_path) # Original function's path logic might need adjustment
        if translated_path and os.path.exists(translated_path): # Check if translation happened and file exists
            logger.info(f"Translated feature names for {timeframe_name} saved to {translated_path}")
            
    except Exception as e:
        logger.error(f"Error during model evaluation for {timeframe_name}: {str(e)}")
    
    return model, accuracy, feature_importance_df

def time_based_model_analysis(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split data by 'Inspection_Time' in half-year steps and train models in each time period.
    
    Args:
        data: Preprocessed data.
        
    Returns:
        Dict[str, pd.DataFrame]: Feature importance data for each time period.
    """
    # Ensure necessary directories are created
    create_analysis_dirs() # Uses default paths
    output_base_dir = '../data/hospital_A/time_based_analysis' # Consistent with create_analysis_dirs

    # Ensure '检查时间' (Inspection_Time) column exists and is in datetime format
    if '检查时间' not in data.columns or not pd.api.types.is_datetime64_dtype(data['检查时间']): # '检查时间' (Inspection_Time)
        logger.error("'检查时间' ('Inspection_Time') column is missing or not in datetime format. Aborting time-based analysis.")
        return {}
    
    min_date = data['检查时间'].min() # '检查时间' (Inspection_Time)
    max_date = data['检查时间'].max() # '检查时间' (Inspection_Time)
    
    if pd.isna(min_date) or pd.isna(max_date):
        logger.error("Min/Max dates for '检查时间' ('Inspection_Time') are NaT. Cannot proceed with time-based analysis.")
        return {}

    logger.info(f"Data time range from {min_date.date()} to {max_date.date()}")
    
    # Calculate time periods (half-year step)
    current_date = min_date
    time_periods = []
    
    while current_date < max_date:
        next_date = current_date + pd.DateOffset(months=6)
        time_periods.append((current_date, next_date))
        current_date = next_date
    
    if not time_periods:
        logger.warning("No time periods generated. Data range might be too short.")
        return {}
    logger.info(f"Divided into {len(time_periods)} time periods.")
    
    feature_importance_results = {}
    
    for i, (start_date, end_date) in enumerate(time_periods):
        timeframe_name = f"period_{i+1}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        logger.info(f"Processing time period {i+1}/{len(time_periods)}: {start_date.date()} to {end_date.date()}")
        
        model, accuracy, fi_df = build_model_for_timeframe(
            data, start_date, end_date, timeframe_name, output_base_dir=output_base_dir
        )
        
        if model is not None and not fi_df.empty:
            feature_importance_results[timeframe_name] = fi_df
            logger.info(f"Model accuracy for {timeframe_name}: {accuracy:.4f}")
        else:
            logger.warning(f"Model training failed or no feature importance data for {timeframe_name}.")
    
    generate_summary_report(feature_importance_results, output_base_dir)
    
    return feature_importance_results

def generate_summary_report(feature_importance_results: Dict[str, pd.DataFrame], output_base_dir: str) -> None:
    """
    Generate a summary report for the time period analysis.
    
    Args:
        feature_importance_results: Feature importance data for each time period.
        output_base_dir: Base directory to save the summary.
    """
    logger.info("Generating summary report of feature importances...")
    
    if not feature_importance_results:
        logger.warning("No feature importance results available, cannot generate summary report.")
        return
    
    all_features = set()
    for df in feature_importance_results.values():
        if 'feature' in df.columns:
            all_features.update(df['feature'].unique())
    
    summary_data = []
    
    for feature_name in all_features:
        feature_data_row = {'feature': feature_name}
        for period_name, df in feature_importance_results.items():
            if 'feature' in df.columns and 'importance' in df.columns:
                importance_series = df.loc[df['feature'] == feature_name, 'importance']
                feature_data_row[period_name] = importance_series.iloc[0] if not importance_series.empty else 0.0
            else:
                feature_data_row[period_name] = 0.0
        summary_data.append(feature_data_row)
    
    summary_df = pd.DataFrame(summary_data)
    
    importance_cols = [col for col in summary_df.columns if col.startswith('period_')]
    if importance_cols:
        summary_df['avg_importance'] = summary_df[importance_cols].mean(axis=1)
        summary_df['std_dev_temporal_importance'] = summary_df[importance_cols].std(axis=1)
        summary_df = summary_df.sort_values('avg_importance', ascending=False)
    
    summary_path = os.path.join(output_base_dir, 'feature_importance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Feature importance summary report saved to {summary_path}")
    
    translated_summary_path = translate_feature_names(summary_path) # Uses its internal logic for translated path
    if translated_summary_path and os.path.exists(translated_summary_path):
        logger.info(f"Translated summary report saved to {translated_summary_path}")

def build_full_data_model(data: pd.DataFrame) -> Tuple[Optional[RandomForestClassifier], float, pd.DataFrame]:
    """
    Build and evaluate a model using the full dataset.
    
    Args:
        data: Preprocessed data.
        
    Returns:
        tuple: (Trained model, accuracy, feature importance DataFrame).
    """
    logger.info("Preparing full data model...")
    output_base_dir_full_data = '../data/hospital_A/full_data_analysis'
    os.makedirs(output_base_dir_full_data, exist_ok=True)
    
    # Call build_model_for_timeframe with a dummy full timeframe name
    # This reuses the logic, but it's a bit of a workaround.
    # A dedicated function might be cleaner if logic diverges significantly.
    # For now, to match original structure:
    # Need to create a "timeframe_name" that represents the full data.
    # The start/end dates passed to build_model_for_timeframe won't filter if they span the whole dataset.
    min_full_date = data['检查时间'].min() if '检查时间' in data.columns else datetime.min # '检查时间' (Inspection_Time)
    max_full_date = data['检查时间'].max() if '检查时间' in data.columns else datetime.max # '检查时间' (Inspection_Time)
    
    if pd.isna(min_full_date) or pd.isna(max_full_date):
        logger.error("Cannot determine date range for full data model. '检查时间' ('Inspection_Time') might be problematic.")
        return None, 0.0, pd.DataFrame()

    full_data_timeframe_name = f"full_dataset_{min_full_date.strftime('%Y%m%d')}_{max_full_date.strftime('%Y%m%d')}"

    model, accuracy, feature_importance_df = build_model_for_timeframe(
        data, 
        min_full_date, # Start date that includes all data
        max_full_date + timedelta(days=1), # End date that includes all data
        full_data_timeframe_name,
        output_base_dir=output_base_dir_full_data
    )
    
    # Additional metrics specific to full data model evaluation (original code had this here)
    if model and not feature_importance_df.empty: # Check if model training and initial eval was successful
        logger.info("Calculating additional metrics for the full data model...")
        # To get X_test, y_test for full data, we need to redo split or get it from build_model_for_timeframe
        # The current build_model_for_timeframe doesn't return X_test, y_test.
        # For simplicity, let's re-perform the split here for additional metrics.
        # This is slightly inefficient but ensures data consistency.
        
        model_data_copy = data.copy()
        model_data_copy = model_data_copy.replace([np.inf, -np.inf], np.nan)
        numeric_cols = model_data_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if model_data_copy[col].isna().any():
                median_val = model_data_copy[col].median()
                model_data_copy[col] = model_data_copy[col].fillna(median_val if pd.notna(median_val) else -1)
        
        cat_cols = model_data_copy.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            if col != '病案号': # '病案号' (Medical Record Number)
                model_data_copy[col] = model_data_copy[col].astype('category').cat.codes

        if '时间差' in model_data_copy.columns: # '时间差' (Time_Difference)
            model_data_copy['时间差_binary'] = model_data_copy['时间差'].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0) # '时间差_binary' from '时间差'
            target_col_name = '时间差_binary'
        else:
            logger.error("Target '时间差' ('Time_Difference') missing in full data for additional metrics.")
            return model, accuracy, feature_importance_df # Return what we have

        features_to_drop = [target_col_name, '时间差', '检查时间'] # '时间差' (Time_Difference), '检查时间' (Inspection_Time)
        X_full = model_data_copy.drop(columns=[col for col in features_to_drop if col in model_data_copy.columns], errors='ignore')
        y_full = model_data_copy[target_col_name]

        groups_full = X_full['病案号'] if '病案号' in X_full.columns else None # '病案号' (Medical Record Number)
        X_train_full, X_test_full, y_train_full, y_test_full = None, None, None, None

        if groups_full is not None:
            X_features_full = X_full.drop(columns=['病案号'], errors='ignore') # '病案号' (Medical Record Number)
            gss_full = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            try:
                train_idx_f, test_idx_f = next(gss_full.split(X_features_full, y_full, groups=groups_full))
                X_test_full, y_test_full = X_features_full.iloc[test_idx_f], y_full.iloc[test_idx_f]
            except ValueError as e:
                logger.warning(f"GroupShuffleSplit failed for full data metrics: {e}. Falling back to random split.")
                _, X_test_full, _, y_test_full = train_test_split(X_features_full, y_full, test_size=0.2, random_state=42, stratify=y_full if len(y_full.unique()) > 1 else None)
        else:
            X_features_full = X_full
            _, X_test_full, _, y_test_full = train_test_split(X_features_full, y_full, test_size=0.2, random_state=42, stratify=y_full if len(y_full.unique()) > 1 else None)

        if X_test_full is not None and not X_test_full.empty and y_test_full is not None and not y_test_full.empty and len(y_test_full.unique()) > 1:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            y_pred_full_metrics = model.predict(X_test_full)
            
            precision = precision_score(y_test_full, y_pred_full_metrics, zero_division=0)
            recall = recall_score(y_test_full, y_pred_full_metrics, zero_division=0)
            f1 = f1_score(y_test_full, y_pred_full_metrics, zero_division=0)
            cm = confusion_matrix(y_test_full, y_pred_full_metrics)
            
            specificity = 0.0
            if cm.size == 4 : # Ensure it's a 2x2 matrix
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # AUC already calculated by build_model_for_timeframe and saved in its ROC file.
            # We can extract it from the saved file or re-calculate. For now, let's focus on new metrics.
            roc_auc_full, auc_ci_lower_full, auc_ci_upper_full = calculate_auc_ci(y_test_full, model.predict_proba(X_test_full)[:,1])

            metrics_data = {
                'accuracy': [accuracy], # This is accuracy from build_model_for_timeframe's split
                'precision': [precision],
                'recall': [recall],
                'f1_score': [f1],
                'specificity': [specificity],
                'auc': [roc_auc_full], # AUC from this specific split
                'auc_ci_lower': [auc_ci_lower_full],
                'auc_ci_upper': [auc_ci_upper_full]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = os.path.join(output_base_dir_full_data, 'evaluation_metrics_full_data.csv')
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Additional evaluation metrics for full data model saved to {metrics_path}")
        else:
            logger.warning("Could not calculate additional metrics for full data model due to test data issues.")

    return model, accuracy, feature_importance_df


def special_time_based_model_analysis(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Perform model analysis for special, predefined time periods.
    
    Args:
        data: Preprocessed data.    
    Returns:
        Dict[str, pd.DataFrame]: Feature importance data for each special time period.
    """
    logger.info("Starting special time period model analysis...")
    # create_special_time_based_dirs() # Call this if distinct output paths are desired and implemented
    # For now, using the same output structure as regular time_based_model_analysis
    # This means it might overwrite files if timeframe_names overlap or are not distinguished.
    # The original code uses 'create_analysis_dirs()' inside special_time_based_model_analysis.
    create_analysis_dirs() # Creates general dirs, not special ones.
    output_base_dir_special = '../data/hospital_A/time_based_analysis' # Defaulting to common dir
    # If truly special output directories are needed:
    # output_base_dir_special = '../data/hospital_A/special_time_based_analysis'
    # And create_special_time_based_dirs() should be called.
    
    if '检查时间' not in data.columns or not pd.api.types.is_datetime64_dtype(data['检查时间']): # '检查时间' (Inspection_Time)
        logger.error("'检查时间' ('Inspection_Time') column is missing or not in datetime format. Aborting special analysis.")
        return {}
    
    min_date_data = data['检查时间'].min() # '检查时间' (Inspection_Time)
    max_date_data = data['检查时间'].max() # '检查时间' (Inspection_Time)

    if pd.isna(min_date_data) or pd.isna(max_date_data):
        logger.error("Min/Max dates for '检查时间' ('Inspection_Time') are NaT. Cannot proceed with special time-based analysis.")
        return {}
    logger.info(f"Data time range for special analysis: {min_date_data.date()} to {max_date_data.date()}")
    
    # Define special time points for splitting data
    time1_str = "2015-12-31"
    time2_str = "2018-12-31"
    try:
        time1 = datetime.strptime(time1_str, "%Y-%m-%d")
        time2 = datetime.strptime(time2_str, "%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid date format for special time points: {time1_str}, {time2_str}")
        return {}

    # Define time periods based on these points and data range
    time_periods_special = [
        (min_date_data, time1), 
        (time1, time2), 
        (time2, max_date_data + timedelta(days=1)) # Ensure last period includes max_date_data
    ]
    # Filter out periods that are invalid (e.g. end_date <= start_date)
    time_periods_special = [(s, e) for s, e in time_periods_special if e > s]


    if not time_periods_special:
        logger.warning("No valid special time periods generated.")
        return {}
    logger.info(f"Divided into {len(time_periods_special)} special time periods.")
    
    feature_importance_results_special = {}
    
    for i, (start_date, end_date) in enumerate(time_periods_special):
        # Add "special" prefix to timeframe_name to distinguish from regular periods if saved in same dir
        timeframe_name = f"special_period_{i+1}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        logger.info(f"Processing special time period {i+1}/{len(time_periods_special)}: {start_date.date()} to {end_date.date()}")
        
        model, accuracy, fi_df = build_model_for_timeframe(
            data, start_date, end_date, timeframe_name, output_base_dir=output_base_dir_special
        )
        
        if model is not None and not fi_df.empty:
            feature_importance_results_special[timeframe_name] = fi_df
            logger.info(f"Model accuracy for special period {timeframe_name}: {accuracy:.4f}")
        else:
            logger.warning(f"Model training failed or no feature importance data for special period {timeframe_name}.")
    
    # Generate a summary report specifically for these special periods
    # The output filename should reflect it's for special periods.
    # generate_summary_report might need an argument to customize filename.
    # For now, it will use 'feature_importance_summary.csv' in output_base_dir_special.
    # To make it distinct:
    summary_output_dir = os.path.join(output_base_dir_special, "special_summary") # Example different dir
    os.makedirs(summary_output_dir, exist_ok=True)
    # This would require generate_summary_report to accept a full path or more flexible dir structure
    generate_summary_report(feature_importance_results_special, output_base_dir_special) # Still uses the general summary func

    return feature_importance_results_special


if __name__ == "__main__":
    logger.info("Starting time-period-based data analysis and model training.")
    
    try:
        # Ensure all necessary general directories are created
        create_analysis_dirs()
        
        processed_data_df = process_data()

        if processed_data_df.empty or '检查时间' not in processed_data_df.columns: # '检查时间' (Inspection_Time)
            logger.error("Data processing failed or '检查时间' ('Inspection_Time') column is missing. Exiting.")
        else:
            logger.info("Starting full data model training and evaluation...")
            build_full_data_model(processed_data_df)
            logger.info("Full data model training and evaluation completed.")

            # Perform time-based model analysis (half-year steps)
            logger.info("Starting time-based (half-year steps) model analysis...")
            time_based_fi_results = time_based_model_analysis(processed_data_df)
            logger.info("Time-based (half-year steps) model analysis completed.")

            # Perform special time period model analysis (commented out in original)
            # logger.info("Starting special time period model analysis...")
            # create_special_time_based_dirs() # Call if special directories are to be used
            # special_fi_results = special_time_based_model_analysis(processed_data_df)
            # logger.info("Special time period model analysis completed.")

        logger.info("Analysis finished.")
        
    except Exception as e:
        logger.exception(f"An error occurred during the analysis process: {str(e)}")