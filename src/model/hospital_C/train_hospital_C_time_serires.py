#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Series Hospital Prediction Model Training Module.

This module provides functionality for training deep learning models for time series
hospital prediction tasks, including data loading, preprocessing, model training,
and evaluation.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from loguru import logger
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root directory to system path
sys.path.append('..')
sys.path.append('.')

# Import custom model modules
try:
    from models.dl_time_series_models import get_model as get_dl_model
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    logger.warning("Failed to import deep learning module, will only use traditional machine learning models")
    DEEP_LEARNING_AVAILABLE = False

class TimeSeriesModelTrainer:
    """Time series model training class.
    
    Provides functionality for training deep learning models on time series data
    for hospital prediction tasks.
    
    Attributes:
        model_dir: Directory for saving models
        results_dir: Directory for saving results
        X_train: Training features (array of sequences)
        X_test: Testing features (array of sequences)
        y_train: Training targets
        y_test: Testing targets
        scaler: Feature standardizer
        models: Dictionary of trained models
        results: Dictionary of model evaluation results
    """
    
    def __init__(self, model_dir: str = '../models_hospital_C_time_series', 
                results_dir: str = '../results_hospital_C_time_series') -> None:
        self.model_dir = model_dir
        self.results_dir = results_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        
        self.models = {}
        self.results = {}
    
    def prepare_data(self, 
                     sequences: List[np.ndarray], 
                     labels: List[int],
                     test_size: float = 0.2,
                     random_state: int = 42) -> None:
        """
        Prepare time series data for model training.
        
        Args:
            sequences: List of time series sequences.
            labels: List of target labels.
            test_size: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
        """
        logger.info("Preparing time series data...")
        
        # Convert to numpy arrays
        sequences_array = np.array(sequences)
        labels_array = np.array(labels)
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            sequences_array, labels_array, test_size=test_size, random_state=random_state,
            stratify=labels_array  # Ensure balanced classes in train and test sets
        )
        
        # Log data shapes
        logger.info(f"Data preparation complete, training set: {len(self.X_train)} samples, test set: {len(self.X_test)} samples")
        logger.info(f"Sequence shape: {self.X_train[0].shape}")
        logger.info(f"Positive sample ratio - Training set: {np.mean(self.y_train):.2%}, Test set: {np.mean(self.y_test):.2%}")

    def evaluate_model(self, model: Any, model_name: str) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model to evaluate.
            model_name: Name of the model.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.X_test is None or self.y_test is None:
            logger.error("Test data not available, cannot evaluate model.")
            return {}
        
        try:
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Handle different output formats
            if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
                if y_pred.shape[1] > 1:  # Multi-class predictions
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_pred_proba = y_pred[:, 1] if y_pred.shape[1] >= 2 else None
                else:  # Binary prediction with shape (n_samples, 1)
                    y_pred_classes = (y_pred.flatten() > 0.5).astype(int)
                    y_pred_proba = y_pred.flatten()
            else:  # Binary prediction with shape (n_samples,)
                y_pred_classes = (y_pred > 0.5).astype(int)
                y_pred_proba = y_pred
            
            # Calculate metrics
            metrics = {}
            metrics['accuracy'] = accuracy_score(self.y_test, y_pred_classes)
            metrics['precision'] = precision_score(self.y_test, y_pred_classes, zero_division=0)
            metrics['recall'] = recall_score(self.y_test, y_pred_classes, zero_division=0)
            metrics['f1'] = f1_score(self.y_test, y_pred_classes, zero_division=0)
            
            # Calculate AUC if probability scores are available
            if y_pred_proba is not None:
                try:
                    metrics['auc'] = roc_auc_score(self.y_test, y_pred_proba)
                except:
                    logger.warning(f"Could not calculate AUC for model {model_name}.")
                    metrics['auc'] = 0.0
            else:
                metrics['auc'] = 0.0
            
            logger.info(f"Model {model_name} evaluation results:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name.upper()}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            return {}
    
    def train_model(self, model_type: str, **kwargs: Any) -> Dict[str, float]:
        """
        Train a time series model.
        
        Args:
            model_type: Type of model to train.
            **kwargs: Additional parameters for model configuration.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if not DEEP_LEARNING_AVAILABLE:
            logger.error("Deep learning module not available, cannot train time series models.")
            return {}
            
        if self.X_train is None or self.y_train is None:
            logger.error("Training data not prepared, call prepare_data() first.")
            return {}
        
        try:
            logger.info(f"Training {model_type} model...")
            
            # Get input shape from the first training example
            input_shape = self.X_train[0].shape
            
            # Create model configuration
            model_config = {
                'input_shape': input_shape,
                'output_size': 1,  # Binary classification
                **kwargs
            }
            
            # Create and train the model
            model = get_dl_model(model_type, model_config)
            model.fit(self.X_train, self.y_train)
            
            # Save the model
            self.models[model_type] = model
            
            # Evaluate the model
            metrics = self.evaluate_model(model, model_type)
            self.results[model_type] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")
            return {}
    
    def train_multiple_models(self, model_types: Optional[List[str]] = None, 
                             **kwargs: Any) -> Dict[str, Dict[str, float]]:
        """
        Train multiple time series models.
        
        Args:
            model_types: List of model types to train.
            **kwargs: Additional parameters for model configurations.
            
        Returns:
            Dictionary mapping model names to their evaluation metrics.
        """
        if not DEEP_LEARNING_AVAILABLE:
            logger.error("Deep learning module not available, cannot train time series models.")
            return {}
            
        if model_types is None:
            model_types = ['lstm', 'bilstm', 'gru', 'tcn', 'transformer']
            
        results = {}
        
        for model_type in model_types:
            model_kwargs = kwargs.get(model_type, {})
            metrics = self.train_model(model_type, **model_kwargs)
            results[model_type] = metrics
            
        # Plot comparison of model performances
        self.plot_results()
            
        return results
    
    def plot_results(self, metric: str = 'f1') -> None:
        """
        Plot model performance comparison.
        
        Args:
            metric: Metric to plot, default is 'f1'.
        """
        if not self.results:
            logger.warning("No results to plot. Train models first.")
            return
            
        # Get results for the specified metric
        model_names = []
        metric_values = []
        
        for model_name, metrics in self.results.items():
            if metric in metrics:
                model_names.append(model_name)
                metric_values.append(metrics[metric])
        
        if not model_names:
            logger.warning(f"No models have results for metric '{metric}'.")
            return
            
        # Create the plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, metric_values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title(f'Time Series Model Performance Comparison ({metric.upper()})')
        plt.xlabel('Model')
        plt.ylabel(f'{metric.upper()} Score')
        plt.ylim(0, max(metric_values) * 1.1)  # Add 10% space above highest bar
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(self.results_dir, exist_ok=True)
        plt.savefig(f'{self.results_dir}/model_comparison_{metric}.png')
        logger.info(f"Saved comparison figure to {self.results_dir}/model_comparison_{metric}.png")
        
        plt.close()
    
    def save_best_model(self, metric: str = 'f1') -> None:
        """
        Save the best performing model.
        
        Args:
            metric: Metric to use for selecting the best model, default is 'f1'.
        """
        if not self.results:
            logger.warning("No results available. Train models first.")
            return
            
        # Find the best model
        best_model_name = None
        best_metric_value = -1
        
        for model_name, metrics in self.results.items():
            if metric in metrics and metrics[metric] > best_metric_value:
                best_metric_value = metrics[metric]
                best_model_name = model_name
        
        if best_model_name is None:
            logger.warning(f"No model found with metric '{metric}'.")
            return
            
        # Save the best model
        if best_model_name in self.models:
            model = self.models[best_model_name]
            filename = f"{self.model_dir}/{best_model_name}_best_model.pkl"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save the model
            try:
                if hasattr(model, 'save'):
                    model.save(filename.replace('.pkl', ''))
                    logger.info(f"Best model ({best_model_name}) saved to {filename.replace('.pkl', '')}")
                else:
                    joblib.dump(model, filename)
                    logger.info(f"Best model ({best_model_name}) saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving best model: {str(e)}")
        else:
            logger.warning(f"Best model ({best_model_name}) not found in models dictionary.")
    
    def save_results(self) -> None:
        """
        Save evaluation results to CSV.
        """
        if not self.results:
            logger.warning("No results to save.")
            return
            
        # Convert results to DataFrame
        data = []
        for model_name, metrics in self.results.items():
            row = {'model': model_name}
            row.update(metrics)
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Save to CSV
        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = f"{self.results_dir}/model_performance.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved model performance results to {csv_path}")

def prepare_sequences(data: pd.DataFrame, 
                     time_col: str,
                     feature_cols: List[str],
                     target_col: str,
                     group_col: str,
                     sequence_length: int = 5,
                     prediction_horizon: int = 1,
                     min_samples: int = 3) -> Tuple[List[np.ndarray], List[int]]:
    """
    Prepare time series sequences from tabular data.
    
    Args:
        data: DataFrame containing time series data.
        time_col: Name of the column containing timestamps.
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        group_col: Name of the column used to group sequences (e.g., patient ID).
        sequence_length: Length of each sequence.
        prediction_horizon: Number of steps ahead to predict.
        min_samples: Minimum number of samples required for a group.
        
    Returns:
        Tuple of (sequences, labels) where sequences is a list of numpy arrays
        and labels is a list of target values.
    """
    logger.info("Preparing time series sequences...")
    
    # Make sure time column is sorted
    data = data.copy()
    if not pd.api.types.is_datetime64_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col])
    
    # Get unique groups
    groups = data[group_col].unique()
    logger.info(f"Found {len(groups)} unique groups.")
    
    sequences = []
    labels = []
    
    for group in groups:
        # Get data for this group
        group_data = data[data[group_col] == group].sort_values(time_col)
        
        if len(group_data) < min_samples:
            continue
            
        # Create sequences
        for i in range(len(group_data) - sequence_length - prediction_horizon + 1):
            sequence = group_data.iloc[i:i+sequence_length][feature_cols].values
            target = group_data.iloc[i+sequence_length+prediction_horizon-1][target_col]
            
            # Convert target to binary if needed
            if not isinstance(target, (int, np.integer, bool, np.bool_)):
                if isinstance(target, (float, np.floating)):
                    target = int(target > 0.5)
                else:
                    try:
                        target = int(target)
                    except:
                        continue
            
            sequences.append(sequence)
            labels.append(target)
    
    logger.info(f"Created {len(sequences)} sequences with length {sequence_length}.")
    logger.info(f"Positive sample ratio: {sum(labels)/len(labels):.2%}")
    
    return sequences, labels

def train_time_series_models(data_path: str,
                            time_col: str = 'admission_time',
                            feature_cols: Optional[List[str]] = None,
                            target_col: str = 'readmission_30d',
                            group_col: str = 'patient_id',
                            sequence_length: int = 5,
                            prediction_horizon: int = 1,
                            **kwargs: Any) -> TimeSeriesModelTrainer:
    """
    Train time series models for hospital prediction.
    
    Args:
        data_path: Path to the data file.
        time_col: Name of the time column.
        feature_cols: List of feature columns.
        target_col: Name of the target column.
        group_col: Name of the group column.
        sequence_length: Length of each sequence.
        prediction_horizon: Number of steps ahead to predict.
        **kwargs: Additional parameters for model training.
        
    Returns:
        Trained TimeSeriesModelTrainer object.
    """
    logger.info(f"Loading data from {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data with {len(data)} rows and {len(data.columns)} columns.")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None
    
    # Determine feature columns if not specified
    if feature_cols is None:
        # Use all numeric columns except the target, time, and group columns
        exclude_cols = [time_col, target_col, group_col]
        feature_cols = [col for col in data.select_dtypes(include=['number']).columns 
                       if col not in exclude_cols]
        logger.info(f"Automatically selected {len(feature_cols)} feature columns.")
        
    # Prepare sequences
    sequences, labels = prepare_sequences(
        data=data,
        time_col=time_col,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    if not sequences:
        logger.error("No sequences created. Check your data and parameters.")
        return None
        
    # Initialize trainer
    trainer = TimeSeriesModelTrainer()
    
    # Prepare data
    trainer.prepare_data(sequences, labels, test_size=0.2, random_state=42)
    
    # Train models
    model_types = kwargs.pop('model_types', ['lstm', 'bilstm', 'gru'])
    
    trainer.train_multiple_models(model_types=model_types, **kwargs)
    
    # Save results and best model
    trainer.save_results()
    trainer.save_best_model()
    
    return trainer

if __name__ == "__main__":
    # Set up logging
    logger.add("logs/train_hospital_C_time_series.log", rotation="500 MB")
    
    # Train models
    data_path = "../data/hospital_C/tmp_preprocessed_data/time_series_data.csv"
    
    # Example of custom model configurations
    model_configs = {
        'lstm': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 50
        },
        'bilstm': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 50
        },
        'gru': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 50
        }
    }
    
    trainer = train_time_series_models(
        data_path=data_path,
        time_col='入院时间',  # admission_time
        target_col='时间差',  # time_difference (target will be binarized)
        group_col='病案号',   # medical_record_number
        sequence_length=5,
        prediction_horizon=1,
        model_types=['lstm', 'bilstm', 'gru'],
        **model_configs
    )
    
    if trainer is not None:
        logger.info("Time series model training completed successfully.")
    else:
        logger.error("Time series model training failed.") 