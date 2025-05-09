#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model training and evaluation module.

This module provides a unified interface to train and evaluate various types of regression models,
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root directory to system path
sys.path.append('..')
sys.path.append('.')

# Import custom model modules
from models.regression_models import train_and_evaluate_models, get_model as get_ml_model
try:
    from models.dl_regression_models import train_and_evaluate_dl_models, get_dl_model
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    logger.warning("Deep learning module import failed, using only traditional machine learning models.")
    DEEP_LEARNING_AVAILABLE = False


class ModelTrainer:
    """Model training and evaluation class.
    
    Provides a unified interface to train and evaluate multiple regression models and save results.
    
    Attributes:
        model_dir: Directory to save models.
        results_dir: Directory to save results.
        X_train: Training features.
        X_test: Testing features.
        y_train: Training target.
        y_test: Testing target.
        scaler: Feature scaler.
        ml_models: Dictionary of traditional machine learning models.
        dl_models: Dictionary of deep learning models.
        ml_results: Evaluation results for traditional machine learning models.
        dl_results: Evaluation results for deep learning models.
    """
    
    def __init__(self, model_dir: str = '../models', results_dir: str = '../results_dl') -> None:
        """Initialize the ModelTrainer class.
        
        Args:
            model_dir: Directory to save models.
            results_dir: Directory to save results.
        """
        self.model_dir = model_dir
        self.results_dir = results_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
        # Model and results dictionaries
        self.ml_models = {}
        self.dl_models = {}
        self.ml_results = {}
        self.dl_results = {}
    
    def get_model(self, model_type: str, **kwargs: Any) -> Any:
        """Get a model instance of the specified type.
        
        Args:
            model_type: Model type.
            **kwargs: Model parameters.
            
        Returns:
            Any: Model instance.
        """
        return get_ml_model(model_type, **kwargs)

    def prepare_data(self, 
                 X: pd.DataFrame, 
                 y: pd.Series, 
                 test_size: float = 0.2, 
                 random_state: int = 42,
                 group_column: Optional[str] = None
                ) -> None:
        """
        Prepare data for model training.

        Uses GroupShuffleSplit to split data by groups if group_column is specified and exists in X,
        ensuring that the same group (e.g., same medical record number) does not appear in both
        training and testing sets. Otherwise, uses regular random split.
        After splitting, features are standardized.
        
        Args:
            X: Feature data.
            y: Target variable.
            test_size: Proportion of the dataset to include in the test split (default 0.2).
            random_state: Random seed.
            group_column: Column name used for grouping, ensuring data from the same group
                          does not appear in both training and testing sets.
        """
        from sklearn.model_selection import GroupShuffleSplit

        logger.info("Preparing model data...")

        # If group_column is specified and exists in X's columns, use GroupShuffleSplit for grouped splitting
        if group_column and group_column in X.columns:
            logger.info(f"Splitting dataset using GroupShuffleSplit, grouping by {group_column}...")

            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=X[group_column]))
            
            # Get training/testing sets
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Record the number of groups in training/testing sets
            train_groups = X_train[group_column].unique()
            test_groups = X_test[group_column].unique()
            logger.info(f"Group split complete. Training set contains {len(train_groups)} unique {group_column}s, "
                        f"testing set contains {len(test_groups)} unique {group_column}s.")

            # If needed, remove group_column (when it's not 'index') 
            # (Note: This comment was in the original code, seems like a placeholder for potential future logic)

            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
        else:
            # If group_column is not specified or not found in X, use regular train_test_split
            logger.info("No valid group_column specified or found. Using regular random split (train_test_split)...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Standardize features
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns
        )

        logger.info(f"Data preparation complete. Training set: {len(self.X_train)} samples, Test set: {len(self.X_test)} samples.")

    
    def train_ml_models(self, model_types: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        """Train traditional machine learning models.
        
        Args:
            model_types: List of model types to train. If None, trains all supported models.
            **kwargs: Parameters to pass to model constructors.
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each model.
        """
        if model_types is None:
            model_types = ['linear', 'ridge', 'svr', 'rf', 'xgb', 'lgb', 'hgb']
        
        results = {}
        
        # Define specific parameters for each model type
        model_specific_params = {
            'linear': {'fit_intercept'},
            'ridge': {'alpha', 'fit_intercept'},
            'svr': {'kernel', 'C', 'epsilon', 'gamma'},
            'rf': {'n_estimators', 'max_depth', 'min_samples_leaf', 'n_jobs', 'random_state'},
            'xgb': {'n_estimators', 'max_depth', 'learning_rate', 'objective', 'verbosity', 'random_state'},
            'lgb': {'n_estimators', 'max_depth', 'learning_rate', 'objective', 'verbose', 'random_state'},
            'hgb': {'max_iter', 'learning_rate', 'l2_regularization', 'max_bins', 'random_state'}
        }
        
        for model_type in model_types:
            try:
                # Filter out specific parameters for the current model type
                valid_params = {k: v for k, v in kwargs.items() 
                              if k in model_specific_params.get(model_type, set())}
                
                logger.info(f"Training {model_type} model with parameters: {valid_params}")
                
                # Get and train the model
                model = self.get_model(model_type, **valid_params)
                model.fit(self.X_train_scaled, self.y_train)  # Use standardized training data
                
                # Evaluate the model
                metrics = self.evaluate_model(model, model_type)
                results[model_type] = metrics
                
                # Save the model
                model_path = os.path.join(self.model_dir, f"{model_type}_model.pkl")
                joblib.dump(model, model_path)
                logger.info(f"{model_type} model saved to {model_path}")
                
                # Store in the model dictionary
                self.ml_models[model_type] = model
                self.ml_results[model_type] = metrics # Store results here
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {str(e)}")
                results[model_type] = {
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': float('-inf'),
                    'error': str(e)
                }
                self.ml_results[model_type] = results[model_type] # Store error results here
        
        # Find the best model
        if self.ml_results: # Use self.ml_results which is correctly populated
            # Filter out entries with errors before finding min
            valid_ml_results = {k: v for k, v in self.ml_results.items() if 'error' not in v}
            if valid_ml_results:
                best_model_type = min(valid_ml_results.items(), key=lambda x: x[1]['rmse'])[0]
                logger.info(f"Best traditional machine learning model: {best_model_type}, RMSE: {self.ml_results[best_model_type]['rmse']}")
        
        return self.ml_results # Return the stored results
    
    def train_dl_models(self, model_types: Optional[List[str]] = None, 
                       **kwargs: Any) -> Dict[str, Dict[str, float]]:
        """Train deep learning models.
        
        Args:
            model_types: List of model types to train, defaults to None.
            **kwargs: Parameters to pass to model constructors.
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each model.
        """
        if not DEEP_LEARNING_AVAILABLE:
            logger.warning("Deep learning module is not available, skipping deep learning model training.")
            return {}
        
        logger.info("Training deep learning models...")
        
        if model_types is None:
            model_types = ['lstm', 'gru', 'transformer', 'histboost'] # 'histboost' is often considered ML, but it's in dl_regression_models.py
        
        # Train and evaluate multiple models
        # This function is expected to return trained models and their results
        # Assuming train_and_evaluate_dl_models populates self.dl_models and self.dl_results
        # Or, it returns models and results, which we then store.
        # The original code assigns to self.dl_results, let's clarify how models are stored.
        
        # The original `train_and_evaluate_dl_models` was likely intended to return results.
        # And `get_dl_model` to get an instance.
        # Let's assume `train_and_evaluate_dl_models` returns a dictionary of results similar to ML.
        # And we need to get and store the models separately if needed for `self.dl_models`.

        all_dl_results = {}
        for model_type in model_types:
            try:
                # Prepare data for sequence models (reshaping if necessary)
                # This might need to be handled inside get_dl_model or fit method of DL models
                # For now, assume X_train_scaled and X_test_scaled are suitable or handled by the DL model class
                
                logger.info(f"Training {model_type} deep learning model...")
                
                # Using the structure from dl_regression_models.py (hypothetical)
                # This part needs to align with how dl_regression_models.py actually works.
                # The original code seems to call train_and_evaluate_dl_models once for all types.
                # Let's stick to the original structure for train_and_evaluate_dl_models.
                pass # This loop is not how the original code was structured for DL models.
            except Exception as e:
                logger.error(f"Error preparing or training DL model {model_type}: {e}")
                all_dl_results[model_type] = {'rmse': float('inf'), 'mae': float('inf'), 'r2': float('-inf'), 'error': str(e)}
        
        # This is the original call pattern.
        # It implies `train_and_evaluate_dl_models` handles multiple model types, trains, evaluates, and returns all results.
        # It also implies models are saved or accessible from within that function or class if needed.
        self.dl_results = train_and_evaluate_dl_models(
            self.X_train_scaled, self.y_train, 
            self.X_test_scaled, self.y_test,
            model_types=model_types, # Pass the list of model types
            model_dir = self.model_dir, # Pass model_dir for saving
            **kwargs # Pass other DL specific params
        )
        
        # The original code then re-trains the best DL model.
        # This seems redundant if train_and_evaluate_dl_models already trained and possibly saved it.
        # Let's assume train_and_evaluate_dl_models returns the trained model objects as well, or we can load them.
        # For now, let's refine `self.dl_models` population.
        # If `train_and_evaluate_dl_models` returns a dict like: {'lstm': {'model': model_obj, 'metrics': {}}},
        # then we can populate self.dl_models and self.dl_results.
        # Based on the current structure, `train_and_evaluate_dl_models` seems to return only metrics.
        # And then the best model is retrained and stored in `self.dl_models`. This is somewhat inefficient.

        if self.dl_results:
            valid_dl_results = {k: v for k, v in self.dl_results.items() if 'error' not in v}
            if valid_dl_results:
                best_model_type = min(valid_dl_results.items(), key=lambda x: x[1]['rmse'])[0]
                logger.info(f"Identified best deep learning model type from evaluation: {best_model_type}")
                
                # Re-train the best model to store it in self.dl_models
                # This assumes get_dl_model gives a fresh, untrained model.
                logger.info(f"Re-training best deep learning model ({best_model_type}) to store instance...")
                best_model = get_dl_model(best_model_type, input_shape=(self.X_train_scaled.shape[1],), **kwargs) # Pass input_shape if needed
                
                # Reshape data if necessary for DL models (e.g., for LSTM, GRU)
                X_train_dl = self.X_train_scaled.values
                if model_type in ['lstm', 'gru', 'transformer']: # Transformer might also need 3D
                     X_train_dl = X_train_dl.reshape((X_train_dl.shape[0], X_train_dl.shape[1], 1))

                best_model.fit(X_train_dl, self.y_train.values) # Use .values for numpy arrays
                
                # Save to model dictionary
                self.dl_models[best_model_type] = best_model
                logger.info(f"Best deep learning model ({best_model_type}) instance stored.")
            else:
                logger.warning("No valid deep learning model results to determine the best model.")
        else:
            logger.info("No deep learning models were trained or results returned.")
            
        return self.dl_results
    
    def train_all_models(self, ml_types: Optional[List[str]] = None, 
                        dl_types: Optional[List[str]] = None,
                        best_metric: str = 'rmse',
                        **kwargs: Any) -> Dict[str, Dict[str, float]]:
        """Train all models (ML and DL).
        
        Args:
            ml_types: List of traditional machine learning model types to train.
            dl_types: List of deep learning model types to train.
            best_metric: Metric used to select the best model, defaults to 'rmse'.
            **kwargs: Parameters to pass to model constructors.
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for all models.
        """
        # Separate traditional ML and Deep Learning parameters
        dl_params = {}
        ml_params = {}
        
        # Deep Learning specific parameters
        dl_specific_params = {
            'epochs', 'batch_size', 'hidden_dim', 
            'num_layers', 'dropout', 'nhead', 'dim_feedforward' 
            # 'learning_rate' can be common but often tuned differently for DL
        }
        
        # Traditional Machine Learning specific parameters (example subset)
        ml_specific_params = {
            # RandomForest params
            'n_estimators', 'max_depth', 'min_samples_leaf',
            # XGBoost params
            # 'learning_rate', # Also in DL, handle carefully
            'objective', 'verbosity',
            # LightGBM params
            # 'objective', 'verbose',
            # Linear model params
            'fit_intercept', 'alpha',
            # SVR params
            'kernel', 'C', 'epsilon', 'gamma',
            # HistGradientBoosting params
            'max_iter', 'l2_regularization', 'max_bins'
        }
        
        # Common parameters
        common_params = {
            'random_state', 'n_jobs', 'learning_rate' # Adding learning_rate here as it's common
        }
        
        # Assign parameters
        for param, value in kwargs.items():
            if param in dl_specific_params:
                dl_params[param] = value
            if param in ml_specific_params: # Check specific first
                ml_params[param] = value
            # If in common, add to both if not already set by specific
            if param in common_params:
                if param not in dl_params: # Prioritize specific if already set
                    dl_params[param] = value
                if param not in ml_params:
                    ml_params[param] = value
        
        # Train traditional machine learning models
        self.ml_results = self.train_ml_models(model_types=ml_types, **ml_params) # Store results
        
        # Train deep learning models
        self.dl_results = self.train_dl_models(model_types=dl_types, **dl_params) # Store results
        
        # Merge results
        all_results = {**self.ml_results, **self.dl_results}
        
        # Find the best model among all
        if all_results:
            # Filter out entries with errors
            valid_results = {k: v for k, v in all_results.items() if isinstance(v, dict) and 'error' not in v}
            if not valid_results:
                logger.warning("No valid model results to determine the overall best model.")
                return all_results

            # Determine best model based on the specified metric
            if best_metric == 'r2':
                # R2: higher is better
                best_model_key = max(valid_results.items(), key=lambda x: x[1].get(best_metric, float('-inf')))[0]
            else:
                # Other metrics (RMSE, MAE, etc.): lower is better
                best_model_key = min(valid_results.items(), key=lambda x: x[1].get(best_metric, float('inf')))[0]
            
            logger.info(f"Based on the {best_metric} metric, the best overall model is: {best_model_key}, "
                        f"{best_metric.upper()}: {all_results[best_model_key].get(best_metric, float('nan')):.4f}")
            
            # Convert results to DataFrame and save
            results_df = pd.DataFrame.from_dict(all_results, orient='index')
            # Ensure dl_types is a list even if None
            _dl_types_list = dl_types if dl_types is not None else []
            results_df['model_category'] = ['dl' if model in _dl_types_list else 'ml' for model in results_df.index]
            results_df['timestamp'] = pd.Timestamp.now()
            
            # Save results to CSV
            results_path = os.path.join(self.results_dir, 'model_evaluation_history.csv')
            
            # If file exists, append results; otherwise, create a new file
            if os.path.exists(results_path):
                results_df.to_csv(results_path, mode='a', header=False, index=True)
            else:
                results_df.to_csv(results_path, index=True)
            
            logger.info(f"Model evaluation results saved to {results_path}")
        
        return all_results
    
    def save_best_model(self, model_type: str, is_dl_model: bool = False) -> None:
        """Save the best model.
        
        Args:
            model_type: Model type.
            is_dl_model: Whether the model is a deep learning model.
        """
        model_to_save = None
        if is_dl_model and model_type in self.dl_models:
            model_to_save = self.dl_models[model_type]
            model_path = os.path.join(self.model_dir, f'dl_{model_type}_regressor.pkl')
        elif not is_dl_model and model_type in self.ml_models:
            model_to_save = self.ml_models[model_type]
            model_path = os.path.join(self.model_dir, f'ml_{model_type}_regressor.pkl')
        else:
            logger.error(f"Model {model_type} (DL: {is_dl_model}) not found in stored models, cannot save.")
            return
        
        if model_to_save is None:
             logger.error(f"Model object for {model_type} is None, cannot save.")
             return

        # Save model
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_to_save, f)
            logger.info(f"Best model ({model_type}) saved to {model_path}")
        except Exception as e:
            # For some complex DL models, pickle might fail. Consider specific saving methods (e.g., Keras' model.save())
            logger.error(f"Failed to save model {model_type} using pickle: {e}. Consider model-specific saving methods for DL models.")
            if is_dl_model and hasattr(model_to_save, 'save'): # Keras-like save
                try:
                    tf_model_path = os.path.join(self.model_dir, f'dl_{model_type}_regressor_tf')
                    model_to_save.save(tf_model_path)
                    logger.info(f"Best DL model ({model_type}) saved in TensorFlow format to {tf_model_path}")
                except Exception as e_tf:
                    logger.error(f"Failed to save DL model {model_type} using TensorFlow format: {e_tf}")


        # Save scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Scaler saved to {scaler_path}")
    
    def save_results(self) -> None:
        """Save evaluation results."""
        # Save traditional machine learning model results
        ml_results_path = os.path.join(self.results_dir, 'ml_results.pkl')
        with open(ml_results_path, 'wb') as f:
            pickle.dump(self.ml_results, f)
        
        # Save deep learning model results
        dl_results_path = os.path.join(self.results_dir, 'dl_results.pkl')
        with open(dl_results_path, 'wb') as f:
            pickle.dump(self.dl_results, f)
        
        # Merge all results
        all_results = {**self.ml_results, **self.dl_results}
        all_results_path = os.path.join(self.results_dir, 'all_results.pkl')
        with open(all_results_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        logger.info(f"Evaluation results saved to {self.results_dir}")
    
    def plot_results(self, metric: str = 'rmse', save_fig: bool = True) -> None:
        """Plot evaluation results.
        
        Args:
            metric: Evaluation metric, supports 'mse', 'rmse', 'mae', 'r2', 'mae_std_ratio'.
            save_fig: Whether to save the plot.
        """
        # Merge all results
        all_results = {**self.ml_results, **self.dl_results}
        
        if not all_results:
            logger.warning("No evaluation results available to plot.")
            return
        
        # Filter out results that might be just error strings
        plot_data = {model: res for model, res in all_results.items() if isinstance(res, dict) and metric in res}

        if not plot_data:
            logger.warning(f"No valid results found for metric '{metric}' to plot.")
            return

        # Check if the specified evaluation metric is valid
        valid_metrics = ['mse', 'rmse', 'mae', 'r2', 'mae_std_ratio']
        if metric not in valid_metrics:
            logger.warning(f"Invalid evaluation metric: {metric}. Using 'rmse' instead.")
            metric = 'rmse'
            plot_data = {model: res for model, res in all_results.items() if isinstance(res, dict) and metric in res} # Re-filter
            if not plot_data:
                logger.warning(f"No valid results found for default metric 'rmse' to plot.")
                return
        
        # Extract evaluation metrics
        model_names = list(plot_data.keys())
        metric_values = [result[metric] for result in plot_data.values()]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values)
        
        # Add titles and labels
        metric_titles = {
            'mse': 'MSE (Mean Squared Error)', 
            'rmse': 'RMSE (Root Mean Squared Error)', 
            'mae': 'MAE (Mean Absolute Error)', 
            'r2': 'R² (R-squared)',
            'mae_std_ratio': 'MAE/STD Ratio'
        }
        title_str = metric_titles.get(metric, metric.upper())
        plt.title(f'Comparison of {title_str} for Different Models', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel(title_str, fontsize=14)
        plt.xticks(rotation=45, ha="right") # Adjust rotation for better readability
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)
        
        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save_fig:
            fig_path = os.path.join(self.results_dir, f'model_comparison_{metric}.png')
            plt.savefig(fig_path, dpi=300)
            logger.info(f"Comparison plot saved to {fig_path}")
        
        # Show plot
        plt.show()
    
    def get_best_model_overall(self, metric: str = 'rmse') -> Tuple[Optional[str], Union[Any, None], bool]:
        """Get the overall best model.
        
        Args:
            metric: Metric used to select the best model, defaults to 'rmse'. 
                    Also supports 'mse', 'mae', 'r2', 'mae_std_ratio'.
            
        Returns:
            Tuple[Optional[str], Union[Any, None], bool]: (model type, model instance, is_dl_model).
                                                          Returns (None, None, False) if no models are available.
        """
        # Merge all results
        all_results = {**self.ml_results, **self.dl_results}
        
        if not all_results:
            logger.warning("No evaluation results available to select the best model.")
            return None, None, False
        
        # Filter out entries with errors or missing metric
        valid_results = {
            k: v for k, v in all_results.items() 
            if isinstance(v, dict) and metric in v and not isinstance(v[metric], str) # ensure metric value is a number
        }

        if not valid_results:
            logger.warning(f"No valid results found for metric '{metric}' to determine the best model.")
            return None, None, False
        
        # Determine best model based on the specified metric
        if metric == 'r2':
            # R2: higher is better
            best_model_type = max(valid_results.items(), key=lambda x: x[1][metric])[0]
        else:
            # Other metrics (RMSE, MAE, etc.): lower is better
            best_model_type = min(valid_results.items(), key=lambda x: x[1][metric])[0]
        
        logger.info(f"Best model selected based on {metric}: {best_model_type}")
        
        # Determine if it's a DL or ML model
        is_dl_model = best_model_type in self.dl_models # Check if key exists in dl_models dict
        
        best_model_instance = None
        if is_dl_model:
            best_model_instance = self.dl_models.get(best_model_type)
        else: # If not in dl_models, assume it's in ml_models (or should be)
            best_model_instance = self.ml_models.get(best_model_type)
            if best_model_type in self.dl_results and best_model_type not in self.ml_models: # e.g. histboost if categorized as DL
                 is_dl_model = True # Correct is_dl_model if it was a DL model not yet in self.dl_models
                 best_model_instance = self.dl_models.get(best_model_type) # Try dl_models again

        if best_model_instance is None:
            logger.warning(f"Best model type {best_model_type} found, but its instance is not available in stored models.")

        return best_model_type, best_model_instance, is_dl_model

    def evaluate_model(self, model: Any, model_type: str) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            model: Trained model instance.
            model_type: Model type.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        # Use test set for prediction
        # Reshape data if necessary for DL models (e.g., for LSTM, GRU)
        X_test_eval = self.X_test_scaled.values
        # Heuristic: if model_type suggests a sequence model and model expects 3D input
        if model_type in ['lstm', 'gru', 'transformer'] and hasattr(model, 'input_shape') and len(model.input_shape) == 3:
             X_test_eval = X_test_eval.reshape((X_test_eval.shape[0], X_test_eval.shape[1], 1))
        
        y_pred = model.predict(X_test_eval)
        
        # Calculate evaluation metrics
        metrics = {}
        metrics['mse'] = mean_squared_error(self.y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(self.y_test, y_pred)
        metrics['r2'] = r2_score(self.y_test, y_pred)
        
        # Calculate MAE/STD metric - newly added metric
        y_std = np.std(self.y_test)
        metrics['mae_std_ratio'] = metrics['mae'] / y_std if y_std > 0 else float('inf')
        
        logger.info(f"{model_type} model evaluation results:")
        logger.info(f"  Mean Squared Error (MSE): {metrics['mse']:.4f}")
        logger.info(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        logger.info(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        logger.info(f"  R-squared (R²): {metrics['r2']:.4f}")
        logger.info(f"  MAE/STD Ratio: {metrics['mae_std_ratio']:.4f}")
        
        return metrics


def train_hospital_models(data_path: str, target_col: str = '时间差', # Time Difference
                         drop_cols: Optional[List[str]] = None,
                         best_metric: str = 'rmse',
                         plot_metric: str = 'rmse',
                         **kwargs: Any) -> ModelTrainer:
    """Convenience function to train models on hospital data.
    
    Args:
        data_path: Path to the data file.
        target_col: Name of the target column. Default: '时间差' (Time Difference)
        drop_cols: List of column names to exclude.
        best_metric: Metric used to select the best model, defaults to 'rmse'.
        plot_metric: Metric used for plotting results, defaults to 'rmse'.
        **kwargs: Parameters to pass to model training functions.
    
    Returns:
        ModelTrainer: Trained ModelTrainer instance.
    """
    logger.info(f"Loading data from {data_path}...")
    
    # Load data
    try:
        data = pd.read_csv(data_path, parse_dates=['上次入院时间', '上次出院时间']) # Previous Admission Time, Previous Discharge Time
        logger.info(f"Data loaded successfully. {len(data)} rows, {len(data.columns)} columns.")
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise
    
    # Handle missing values and problematic values
    data = data.replace([np.inf, -np.inf], np.nan)

   
    # Remove unnecessary columns
    remove_cols_1 = [
        '出生日期',      # Date of Birth
        '入院科室',      # Admission Department
        '出院科室',      # Discharge Department
        '主要诊断',      # Main Diagnosis
        '入院时间',      # Admission Time
        '出院时间',      # Discharge Time
        '上次诊断',      # Previous Diagnosis
        '上次入院时间',  # Previous Admission Time
        '上次出院时间'   # Previous Discharge Time
    ]


    data['上次住院天数'] = (data['上次出院时间'] - data['上次入院时间']).dt.days # Column '上次住院天数' (Previous Length of Stay days)
                                                                          # Uses '上次出院时间' (Previous Discharge Time) and '上次入院时间' (Previous Admission Time)

    # Read unimportant features
    unimportant_features_df = pd.read_csv('../data/hospital_A/tmp_preprocessed_data/unimportant_features.csv')
    unimportant_features = unimportant_features_df['feature'].tolist()

    data.drop(columns=remove_cols_1 + unimportant_features, inplace=True, errors='ignore')


    # '上次入院科室', '上次出院科室' (Previous Admission Department, Previous Discharge Department)
    if '上次入院科室' in data.columns: # Previous Admission Department
        data['上次入院科室'] = data['上次入院科室'].astype('category').cat.codes
    if '上次出院科室' in data.columns: # Previous Discharge Department
        data['上次出院科室'] = data['上次出院科室'].astype('category').cat.codes

    logger.info("Encoding categorical columns...")

    # Fill missing values for all numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isna().any():
            # data[col] = data[col].fillna(data[col].median() if data[col].count() > 0 else -1) # Original commented out: fill with median or -1
            # Fill with -999
            data[col] = data[col].fillna(-999)

    cat_cols = data.select_dtypes(include=['object', 'category']).columns # Include object for safety
    for col in cat_cols:
        if col != '病案号':  # '病案号' (Medical Record Number) # Keep Medical Record Number as ID
            try:
                data[col] = data[col].astype('category').cat.codes
                logger.debug(f"Column '{col}' has been category encoded.")
            except Exception as e:
                logger.warning(f"Could not encode column '{col}': {e}. It might be dropped or require specific handling.")
        

    # Create features and target
    # '时间差' (Time Difference), '检查时间' (Inspection Time)
    X = data.drop(columns=[target_col, '检查时间', '病案号'], errors='ignore') # Drop target, '检查时间', and '病案号' from features
    if '病案号' in data.columns: # '病案号' (Medical Record Number)
        X_with_group = data.drop(columns=[target_col, '检查时间'], errors='ignore') # Keep '病案号' for splitting
    else:
        X_with_group = X.copy() # If no '病案号', proceed without it for grouping

    y = data[target_col] # target_col is '时间差' (Time Difference) by default

    # Filter out extreme values for target variable '时间差' (Time Difference)
    valid_indices = y < 1500
    X = X[valid_indices]
    X_with_group = X_with_group[valid_indices]
    y = y[valid_indices]
    
    # Create ModelTrainer instance
    trainer = ModelTrainer()
    
    # Prepare data - group by '病案号' (Medical Record Number)
    group_column_name = '病案号' # '病案号' (Medical Record Number)
    # Use X_with_group (which contains group_column_name) for prepare_data, then ensure self.X_train/test don't have it
    trainer.prepare_data(X_with_group, y, group_column=group_column_name if group_column_name in X_with_group.columns else None)

    # After prepare_data, X_train and X_test might still contain the group_column.
    # It should be removed from the features used for training if it's not a predictive feature.
    if group_column_name in trainer.X_train.columns:
        trainer.X_train = trainer.X_train.drop(columns=[group_column_name])
        trainer.X_test = trainer.X_test.drop(columns=[group_column_name])
        # Rescale because columns changed
        trainer.X_train_scaled = pd.DataFrame(
            trainer.scaler.fit_transform(trainer.X_train),
            columns=trainer.X_train.columns
        )
        trainer.X_test_scaled = pd.DataFrame(
            trainer.scaler.transform(trainer.X_test),
            columns=trainer.X_test.columns
        )
        logger.info(f"Removed group column '{group_column_name}' from X_train/X_test after splitting and re-scaled.")
    
    # Train all models
    ml_types_default = ['rf', 'xgb', 'lgb', 'linear', 'ridge']
    dl_types_default = ['lstm', 'gru', 'transformer', 'histboost'] if DEEP_LEARNING_AVAILABLE else None
    
    ml_types_to_train = kwargs.pop('ml_types', ml_types_default)
    dl_types_to_train = kwargs.pop('dl_types', dl_types_default)
    
    trainer.train_all_models(ml_types=ml_types_to_train, dl_types=dl_types_to_train, best_metric=best_metric, **kwargs)
    
    # Save results
    trainer.save_results()
    
    # Get and save the best model
    best_model_type, _, is_dl_model = trainer.get_best_model_overall(metric=best_metric)
    if best_model_type:
        trainer.save_best_model(best_model_type, is_dl_model)
        
        # Plot results
        trainer.plot_results(metric=plot_metric)
    
    return trainer


if __name__ == "__main__":
    # Example usage
    data_path = '../data/hospital_A/tmp_preprocessed_data/final_preprocessed_data.csv'
    
    try:
        # Train models
        trainer_instance = train_hospital_models(
            data_path=data_path,
            target_col='时间差', # Target column: '时间差' (Time Difference)
            ml_types=['rf', 'xgb', 'lgb', 'linear', 'ridge', 'svr', 'hgb'],
            dl_types=['lstm', 'gru', 'transformer'] if DEEP_LEARNING_AVAILABLE else None,
            # Choose to use mae_std_ratio as the metric for selecting the best model
            best_metric='mae_std_ratio',
            # Plot comparison charts for both rmse and mae_std_ratio
            plot_metric='rmse', # Initial plot metric
            # Model parameters
            epochs=10,  # Number of training epochs for deep learning models (reduced for quick test)
            batch_size=32,  # Batch size for deep learning models
            learning_rate=0.01, # Learning rate (can be used by both ML and DL models)
            # HistBoost specific parameters (also relevant for some ML models like RF, XGB, LGB)
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            # HGB specific
            l2_regularization=1.0,
            max_bins=255,
            # DL specific
            hidden_dim=64,
            num_layers=2,
            dropout=0.2
        )
        
        # Get the best model
        best_model_name, B_model, is_deep_learning = trainer_instance.get_best_model_overall(metric='mae_std_ratio')
        if best_model_name:
            logger.info(f"Overall best model: {best_model_name} ({'Deep Learning' if is_deep_learning else 'Traditional Machine Learning'})")
        else:
            logger.info("No best model could be determined.")
        
        # Additionally plot MAE/STD ratio comparison chart
        trainer_instance.plot_results(metric='mae_std_ratio')
        
    except Exception as e:
        logger.exception(f"Error during model training process: {str(e)}")