#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Training and Evaluation Module.

This module provides a unified interface for training and evaluating various types of classification models,
including traditional machine learning models, deep learning sequence models, and Transformer models.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root directory to system path
sys.path.append('..')
sys.path.append('.')

# Import custom model modules
# from models.regression_models import train_and_evaluate_models, get_model as get_ml_model # Not used directly
try:
    from models.dl_regression_models import get_dl_model 
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    logger.warning("Deep learning module import failed, using only traditional machine learning models")
    DEEP_LEARNING_AVAILABLE = False


class ModelTrainer:
    """Model training and evaluation class.
    
    Provides a unified interface for training and evaluating multiple classification models and saving results.
    
    Attributes:
        model_dir: Directory for saving models
        results_dir: Directory for saving results
        X_train: Training features (DataFrame before scaling)
        X_test: Testing features (DataFrame before scaling)
        y_train: Training targets (Series)
        y_test: Testing targets (Series)
        X_train_scaled: Standardized training features (DataFrame)
        X_test_scaled: Standardized testing features (DataFrame)
        scaler: Feature standardizer
        ml_models: Traditional machine learning models dictionary
        dl_models: Deep learning models dictionary
        ml_results: Traditional machine learning model evaluation results
        dl_results: Deep learning model evaluation results
    """
    
    def __init__(self, model_dir: str = '../models_hospital_C', results_dir: str = '../results_hospital_C') -> None:
        self.model_dir = model_dir
        self.results_dir = results_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.X_train_scaled: Optional[pd.DataFrame] = None
        self.X_test_scaled: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        
        self.ml_models = {}
        self.dl_models = {}
        self.ml_results = {}
        self.dl_results = {}
    
    def get_model(self, model_type: str, **kwargs: Any) -> Any:
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**kwargs)
        elif model_type == 'xgb':
            from xgboost import XGBClassifier
            return XGBClassifier(**kwargs)
        elif model_type == 'lgb':
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**kwargs)
        elif model_type == 'linear':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**kwargs)
        elif model_type == 'ridge':
            from sklearn.linear_model import RidgeClassifier
            return RidgeClassifier(**kwargs)
        elif model_type == 'svc':
            from sklearn.svm import SVC
            return SVC(**kwargs)
        elif model_type == 'hgb':
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def prepare_data(self, 
                 X: pd.DataFrame, 
                 y: pd.Series, 
                 test_size: float = 0.2, 
                 random_state: int = 42,
                 group_column: Optional[str] = None
                ) -> None:
        from sklearn.model_selection import GroupShuffleSplit
        logger.info("Preparing model data...")

        if group_column and group_column in X.columns:
            logger.info(f"Grouping by {group_column} using GroupShuffleSplit to split dataset...")
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=X[group_column]))
            
            self.X_train = X.iloc[train_idx]
            self.X_test = X.iloc[test_idx]
            self.y_train = y.iloc[train_idx]
            self.y_test = y.iloc[test_idx]
            
            if self.X_train is not None: # mypy check
                train_groups = self.X_train[group_column].unique()
                logger.info(f"Grouping complete, training set contains {len(train_groups)} unique '{group_column}' values")
            if self.X_test is not None: # mypy check
                test_groups = self.X_test[group_column].unique()
                logger.info(f"Test set contains {len(test_groups)} unique '{group_column}' values")
        else:
            logger.info("No valid group_column specified or found, using regular random train_test_split...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        if self.X_train is None or self.X_test is None:
             logger.error("X_train or X_test is None after splitting. Cannot proceed with scaling.")
             return

        X_train_to_scale = self.X_train.copy()
        X_test_to_scale = self.X_test.copy()

        features_for_scaling_train = X_train_to_scale.columns.tolist()
        features_for_scaling_test = X_test_to_scale.columns.tolist()

        if group_column and group_column in features_for_scaling_train:
            X_train_to_scale = X_train_to_scale.drop(columns=[group_column])
            features_for_scaling_train.remove(group_column)
        if group_column and group_column in features_for_scaling_test:
            X_test_to_scale = X_test_to_scale.drop(columns=[group_column])
            features_for_scaling_test.remove(group_column)
        
        numeric_cols_train = X_train_to_scale.select_dtypes(include=np.number).columns
        numeric_cols_test = X_test_to_scale.select_dtypes(include=np.number).columns

        if not numeric_cols_train.empty:
            self.X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train_to_scale[numeric_cols_train]),
                columns=numeric_cols_train,
                index=self.X_train.index
            )
        else:
            logger.warning("No numeric columns in training set for standardization. X_train_scaled will be empty.")
            self.X_train_scaled = pd.DataFrame(index=self.X_train.index)


        if not numeric_cols_test.empty:
            self.X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test_to_scale[numeric_cols_test]),
                columns=numeric_cols_test,
                index=self.X_test.index
            )
        else:
            logger.warning("No numeric columns in test set for standardization. X_test_scaled will be empty.")
            self.X_test_scaled = pd.DataFrame(index=self.X_test.index)


        if self.X_train_scaled is not None and self.X_test_scaled is not None:
            logger.info(f"Data preparation complete, training set: {len(self.X_train_scaled)} samples, test set: {len(self.X_test_scaled)} samples")
        if self.y_train is not None and self.y_test is not None:
            logger.info(f"Positive sample ratio - Training set: {self.y_train.mean():.2%}, Test set: {self.y_test.mean():.2%}")

    def evaluate_model(self, model: Any, model_type: str) -> Dict[str, float]:
        if self.X_test_scaled is None or self.X_test_scaled.empty or self.y_test is None:
            logger.error("X_test_scaled is empty or y_test is undefined, cannot evaluate.")
            return {metric: 0.0 for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']}

        is_dl_model_type = model_type in ['lstm', 'gru', 'transformer'] 

        y_pred_for_metrics: np.ndarray
        y_prob_for_auc: Optional[np.ndarray] = None
        has_prob = False
        
        input_data_for_eval: Union[pd.DataFrame, np.ndarray]
        if is_dl_model_type:
            input_data_for_eval = self.X_test_scaled # Pass DataFrame for DL models
        else:
            input_data_for_eval = self.X_test_scaled.values # Pass NumPy array for ML models

        # Check if input_data_for_eval actually has columns/features before proceeding
        if isinstance(input_data_for_eval, pd.DataFrame) and input_data_for_eval.shape[1] == 0:
             logger.error(f"Input data (DataFrame) for model {model_type} has 0 columns, cannot evaluate.")
             return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0, 'error': "Input DataFrame has 0 columns"}
        elif isinstance(input_data_for_eval, np.ndarray) and input_data_for_eval.ndim > 1 and input_data_for_eval.shape[1] == 0 :
             logger.error(f"Input data (NumPy array) for model {model_type} has 0 columns, cannot evaluate.")
             return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0, 'error': "Input NumPy array has 0 columns"}
        elif isinstance(input_data_for_eval, np.ndarray) and input_data_for_eval.ndim == 1 and input_data_for_eval.shape[0] > 0 and self.X_test_scaled.shape[1] > 0 :
             # This case might be problematic if a 1D array is passed but model expects 2D (N_samples, N_features)
             # However, X_test_scaled.values should produce 2D if X_test_scaled has columns.
             # If X_test_scaled was empty dataframe, .values might be 1D or 2D with 0 columns.
             # The checks above should catch 0 columns.
             pass


        raw_predictions = model.predict(input_data_for_eval)

        if is_dl_model_type:
            dl_output_np = raw_predictions
            if hasattr(dl_output_np, 'cpu'): dl_output_np = dl_output_np.cpu().detach() # Added detach for pytorch
            if hasattr(dl_output_np, 'numpy'): dl_output_np = dl_output_np.numpy()
            
            if not isinstance(dl_output_np, np.ndarray):
                try: dl_output_np = np.array(dl_output_np)
                except Exception as e:
                    logger.error(f"DL model {model_type} output {type(dl_output_np)} not convertible to NumPy: {e}")
                    return {m: 0.0 for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}

            if dl_output_np.ndim == 2:
                if dl_output_np.shape[1] == 1: y_prob_for_auc = dl_output_np.flatten()
                elif dl_output_np.shape[1] == 2: y_prob_for_auc = dl_output_np[:, 1]
                else:
                    logger.error(f"DL model {model_type} predict() 2D output shape {dl_output_np.shape} unhandled.")
                    return {m: 0.0 for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
            elif dl_output_np.ndim == 1: y_prob_for_auc = dl_output_np
            else:
                logger.error(f"DL model {model_type} predict() output ndim {dl_output_np.ndim} (shape {dl_output_np.shape}) unhandled.")
                return {m: 0.0 for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
            
            if y_prob_for_auc is not None:
                 y_pred_for_metrics = (y_prob_for_auc > 0.5).astype(int)
                 has_prob = True
            else: # Should not happen if logic above is correct
                logger.error(f"Failed to derive y_prob_for_auc for DL model {model_type}.")
                return {m: 0.0 for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
        else: # Traditional ML models
            y_pred_for_metrics = raw_predictions 
            try:
                # input_data_for_eval is ndarray here for ML models
                y_prob_for_auc = model.predict_proba(input_data_for_eval)[:, 1] 
                has_prob = True
            except (AttributeError, NotImplementedError):
                y_prob_for_auc = None 
                logger.warning(f"Model {model_type} does not support predict_proba, cannot calculate AUC.")

        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(self.y_test, y_pred_for_metrics)
        metrics['precision'] = precision_score(self.y_test, y_pred_for_metrics, zero_division=0) 
        metrics['recall'] = recall_score(self.y_test, y_pred_for_metrics, zero_division=0)
        metrics['f1'] = f1_score(self.y_test, y_pred_for_metrics, zero_division=0)
        
        # Calculate AUC if probabilities are available
        if has_prob and y_prob_for_auc is not None:
            try:
                metrics['auc'] = roc_auc_score(self.y_test, y_prob_for_auc)
            except ValueError as e:
                logger.error(f"Error calculating AUC: {str(e)}")
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
            
        return metrics

    def plot_results(self, metric: str = 'f1', save_fig: bool = True) -> None:
        """
        Plot performance comparison between models.
        
        Args:
            metric: Metric to plot ('accuracy', 'precision', 'recall', 'f1', 'auc')
            save_fig: Whether to save the figure to the results directory
        """
        if not self.ml_results and not self.dl_results:
            logger.warning("No results to plot. Train models first.")
            return
            
        # Combine results
        all_results = {}
        all_results.update(self.ml_results)
        all_results.update(self.dl_results)
        
        if not all_results:
            logger.warning(f"No models have results for metric '{metric}'.")
            return
            
        # Sort by metric value
        sorted_models = sorted(all_results.items(), key=lambda x: x[1].get(metric, 0), reverse=True)
        model_names = [name for name, _ in sorted_models]
        metric_values = [results.get(metric, 0) for _, results in sorted_models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title(f'Model Performance Comparison ({metric.upper()})')
        plt.xlabel('Model')
        plt.ylabel(f'{metric.upper()} Score')
        plt.ylim(0, max(metric_values) * 1.1)  # Add 10% space above highest bar
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_fig:
            os.makedirs(self.results_dir, exist_ok=True)
            plt.savefig(f'{self.results_dir}/model_comparison_{metric}.png')
            logger.info(f"Saved comparison figure to {self.results_dir}/model_comparison_{metric}.png")
        
        plt.close()
    
    def train_ml_models(self, model_types: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        """
        Train traditional machine learning models.
        
        Args:
            model_types: List of model types to train. If None, trains all available types.
            **kwargs: Additional parameters to pass to model constructors.
            
        Returns:
            Dictionary mapping model names to their evaluation metrics.
        """
        if self.X_train_scaled is None or self.y_train is None:
            logger.error("Data not prepared yet. Call prepare_data() first.")
            return {}
            
        if self.X_train_scaled.empty:
            logger.error("X_train_scaled is empty. Cannot train models.")
            return {}
            
        if model_types is None:
            model_types = ['rf', 'xgb', 'lgb', 'hgb', 'linear', 'ridge', 'svc']
        
        results = {}
        
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} model...")
                model = self.get_model(model_type, **kwargs.get(model_type, {}))
                model.fit(self.X_train_scaled.values, self.y_train.values)
                self.ml_models[model_type] = model
                
                # Evaluate model
                eval_metrics = self.evaluate_model(model, model_type)
                results[model_type] = eval_metrics
                self.ml_results[model_type] = eval_metrics
                
                logger.info(f"{model_type} model results: {eval_metrics}")
            except Exception as e:
                logger.error(f"Error training {model_type} model: {str(e)}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def train_dl_models(self, model_types: Optional[List[str]] = None, 
                       **kwargs: Any) -> Dict[str, Dict[str, float]]:
        """
        Train deep learning models.
        
        Args:
            model_types: List of model types to train. If None, trains all available types.
            **kwargs: Additional parameters to pass to model constructors.
            
        Returns:
            Dictionary mapping model names to their evaluation metrics.
        """
        if not DEEP_LEARNING_AVAILABLE:
            logger.warning("Deep learning modules not available. Cannot train DL models.")
            return {}
            
        if self.X_train_scaled is None or self.y_train is None:
            logger.error("Data not prepared yet. Call prepare_data() first.")
            return {}
            
        if self.X_train_scaled.empty:
            logger.error("X_train_scaled is empty. Cannot train DL models.")
            return {}
            
        if model_types is None:
            model_types = ['lstm', 'gru', 'transformer']
            
        results = {}
        
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} model...")
                
                # Create the model
                model_config = {
                    'input_size': self.X_train_scaled.shape[1],
                    'output_size': 1,  # Binary classification
                    **kwargs.get(model_type, {})
                }
                
                model = get_dl_model(model_type, model_config)
                
                # Train the model
                model.fit(self.X_train_scaled, self.y_train)
                self.dl_models[model_type] = model
                
                # Evaluate model
                eval_metrics = self.evaluate_model(model, model_type)
                results[model_type] = eval_metrics
                self.dl_results[model_type] = eval_metrics
                
                logger.info(f"{model_type} model results: {eval_metrics}")
            except Exception as e:
                logger.error(f"Error training {model_type} model: {str(e)}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def train_all_models(self, ml_types: Optional[List[str]] = None, 
                        dl_types: Optional[List[str]] = None,
                        **kwargs: Any) -> Dict[str, Dict[str, float]]:
        """
        Train both traditional and deep learning models.
        
        Args:
            ml_types: List of traditional machine learning model types to train.
            dl_types: List of deep learning model types to train.
            **kwargs: Additional parameters to pass to model constructors.
            
        Returns:
            Dictionary mapping model names to their evaluation metrics.
        """
        # Train ML models
        logger.info("Training traditional machine learning models...")
        ml_results = self.train_ml_models(model_types=ml_types, **kwargs)
        
        # Train DL models if available
        if DEEP_LEARNING_AVAILABLE:
            logger.info("Training deep learning models...")
            dl_results = self.train_dl_models(model_types=dl_types, **kwargs)
        else:
            dl_results = {}
            logger.warning("Deep learning modules not available. Skipping DL models.")
        
        # Combine results
        results = {}
        results.update(ml_results)
        results.update(dl_results)
        
        # Create comparison plot
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            self.plot_results(metric=metric)
        
        return results
    
    def get_best_model_overall(self) -> Tuple[Optional[str], Union[Any, None], bool]:
        """
        Get the best performing model across all trained models.
        
        Returns:
            Tuple: (model_name, model_object, is_dl_model)
            If no models have been trained, returns (None, None, False)
        """
        all_results = {}
        all_results.update(self.ml_results)
        all_results.update(self.dl_results)
        
        if not all_results:
            logger.warning("No trained models found. Train models first.")
            return None, None, False
        
        # Get the model with the highest F1 score
        best_model_name = max(all_results.items(), key=lambda x: x[1].get('f1', 0))[0]
        is_dl_model = best_model_name in self.dl_models
        
        model = self.dl_models.get(best_model_name) if is_dl_model else self.ml_models.get(best_model_name)
        
        logger.info(f"Best model: {best_model_name} with F1 score: {all_results[best_model_name].get('f1', 0):.4f}")
        
        return best_model_name, model, is_dl_model
    
    def save_best_model(self, model_type: str, is_dl_model: bool = False) -> None:
        """
        Save the specified model to disk.
        
        Args:
            model_type: Type of model to save
            is_dl_model: Whether the model is a deep learning model
        """
        model_dict = self.dl_models if is_dl_model else self.ml_models
        if model_type not in model_dict:
            logger.error(f"Model {model_type} not found in {'DL' if is_dl_model else 'ML'} models.")
            return
            
        model = model_dict[model_type]
        filename = f"{self.model_dir}/{model_type}_model.pkl"
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save the model
        if is_dl_model:
            # Deep learning models may have custom save methods
            if hasattr(model, 'save'):
                model.save(filename.replace('.pkl', ''))
                logger.info(f"Deep learning model {model_type} saved to {filename.replace('.pkl', '')}")
            else:
                joblib.dump(model, filename)
                logger.info(f"Deep learning model {model_type} saved to {filename}")
        else:
            # Standard sklearn models
            joblib.dump(model, filename)
            logger.info(f"Machine learning model {model_type} saved to {filename}")
    
    def save_results(self) -> None:
        """
        Save evaluation results to CSV and performance plot images.
        """
        # Combine results
        all_results = {}
        for model_type, metrics in self.ml_results.items():
            all_results[model_type] = metrics
        for model_type, metrics in self.dl_results.items():
            all_results[model_type] = metrics
            
        if not all_results:
            logger.warning("No results to save.")
            return
            
        # Convert to DataFrame
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'model'
        results_df.reset_index(inplace=True)
        
        # Save to CSV
        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = f"{self.results_dir}/model_performance.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved model performance results to {csv_path}")
        
        # Create plots for each metric
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            self.plot_results(metric=metric, save_fig=True)

def train_hospital_models(data_path: str, target_col: str = 'time_difference', 
                         **kwargs: Any) -> Optional[ModelTrainer]:
    """
    Train models for hospital readmission prediction.
    
    Args:
        data_path: Path to the preprocessed dataset
        target_col: Name of the target column
        **kwargs: Additional parameters for model training
        
    Returns:
        Trained ModelTrainer object or None if training fails
    """
    logger.info(f"Loading data from {data_path}")
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None
        
    # Check if target column exists
    if target_col not in data.columns:
        logger.error(f"Target column '{target_col}' not found in dataset. Available columns: {data.columns.tolist()}")
        return None
        
    # Create binary target (1 if readmission within 30 days, 0 otherwise)
    y = (data[target_col] <= 30).astype(int)
    logger.info(f"Created binary target (readmission within 30 days). Positive ratio: {y.mean():.2%}")
    
    # Remove target column from features
    X = data.drop(columns=[target_col])
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    group_column = kwargs.pop('group_column', '病案号')  # 'medical_record_number'
    if group_column in X.columns:
        logger.info(f"Using {group_column} for group-based data splitting")
    else:
        logger.warning(f"Group column {group_column} not found in dataset. Using random splitting.")
        group_column = None
        
    trainer.prepare_data(X, y, test_size=0.2, random_state=42, group_column=group_column)
    
    # Train models
    ml_types = kwargs.pop('ml_types', ['rf', 'xgb', 'lgb', 'hgb'])
    dl_types = kwargs.pop('dl_types', ['lstm']) if DEEP_LEARNING_AVAILABLE else []
    
    trainer.train_all_models(ml_types=ml_types, dl_types=dl_types, **kwargs)
    
    # Save results
    trainer.save_results()
    
    # Save best model
    best_model_name, _, is_dl_model = trainer.get_best_model_overall()
    if best_model_name is not None:
        trainer.save_best_model(best_model_name, is_dl_model)
        
    return trainer

if __name__ == "__main__":
    # Set up logging
    logger.add("logs/train_hospital_C.log", rotation="500 MB")
    
    # Train models
    data_path = "../data/hospital_C/tmp_preprocessed_data/final_data.csv"
    trainer = train_hospital_models(
        data_path=data_path,
        target_col='时间差',  # 'time_difference'
        group_column='病案号',  # 'medical_record_number'
        ml_types=['rf', 'xgb', 'lgb', 'hgb'],
        dl_types=['lstm'] if DEEP_LEARNING_AVAILABLE else []
    )
    
    if trainer is not None:
        logger.info("Model training completed successfully.")
    else:
        logger.error("Model training failed.")