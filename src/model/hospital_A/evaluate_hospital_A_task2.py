#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时间序列医院预测模型评估模块 - 按时间维度评估。

该模块将数据集按比例划分为训练集和测试集，
并分别从月度和季度维度分析模型在测试集上的表现。
同时进行时序特征重要性分析。
"""

from typing import Dict, List, Tuple, Union, Literal
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
from loguru import logger
import warnings
from datetime import datetime
from utils.decorators import translate_to_csv

# 配置日志记录
logger.add("logs/temporal_evaluation.log", rotation="500 MB")

# 抑制警告
warnings.filterwarnings('ignore')

class TemporalEvaluator:
    """时间维度评估器类"""
    
    def __init__(
        self,
        task_type: Literal['regression', 'classification'] = 'regression',
        model_type: Literal['xgboost', 'randomforest'] = 'xgboost',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """
        初始化评估器。
        
        Args:
            task_type: 任务类型，'regression' 或 'classification'
            model_type: 模型类型，'xgboost' 或 'randomforest'（仅用于分类任务）
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.task_type = task_type
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        
        # 根据任务类型和模型类型初始化模型
        if task_type == 'regression':
            self.model = XGBRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
                max_depth=10,
                learning_rate=0.1,
                booster='gbtree',
                objective='reg:squarederror'
            )
        else:  # classification
            if model_type == 'xgboost':
                self.model = XGBClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1,
                    max_depth=10,
                    learning_rate=0.1,
                    booster='gbtree'
                )
            else:  # randomforest
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1,
                    max_depth=10
                )
        
        self.feature_names: List[str] = []
        
    def load_data(self) -> pd.DataFrame:
        """加载和预处理数据"""
        logger.info("加载数据...")
        data = pd.read_csv(
            '../data/hospital_A/tmp_preprocessed_data/final_preprocessed_data.csv',
            parse_dates=['上次入院时间', '上次出院时间', '检查时间']
        )
        
        # 处理极端值
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # 移除不需要的列
        remove_cols = [
            '出生日期', '入院科室', '出院科室', '主要诊断', '入院时间', '出院时间',
            '上次诊断', '上次入院时间', '上次出院时间'
        ]
        
        # 计算上次住院天数
        data['上次住院天数'] = (data['上次出院时间'] - data['上次入院时间']).dt.days
        
        # 读取不重要特征
        unimportant_features = pd.read_csv(
            '../data/hospital_A/tmp_preprocessed_data/unimportant_features.csv'
        )
        unimportant_features = unimportant_features['feature'].tolist()
        if self.task_type == 'classification':
            unimportant_features=[]
            
        data.drop(columns=remove_cols+unimportant_features, inplace=True)
        
        # 处理分类特征
        cat_cols = ['上次入院科室', '上次出院科室']
        for col in cat_cols:
            data[col] = data[col].astype('category').cat.codes
            
        # 过滤异常值
        data = data[data['时间差'] < 1500]
        
        logger.info(f"数据处理完成: {len(data)} 行, {len(data.columns)} 列")
        return data
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """准备特征和目标变量"""
        feature_cols = [
            col for col in data.columns 
            if col not in ['时间差', '检查时间']
        ]
        self.feature_names = feature_cols
        X = data[feature_cols]
        
        if self.task_type == 'regression':
            y = data['时间差']
        else:  # classification
            # 将时间差转换为二分类标签：大于0为1，等于0为0
            y = (data['时间差'] > 0).astype(int)
            
        return X, y
        
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        划分训练集和测试集
        
        Args:
            X: 特征矩阵
            y: 目标变量
            dates: 时间列
            
        Returns:
            训练集特征、测试集特征、训练集标签、测试集标签、训练集时间、测试集时间
        """
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test, dates_train, dates_test
        
    def evaluate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray = None,
        std_epsilon: float = 1  # 添加偏移量参数
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            y_train: 训练集真实值，用于计算MASE
            std_epsilon: 计算MAE/STD时用于避免除零的小偏移量
            
        Returns:
            包含各评估指标的字典
        """
        if self.task_type == 'regression':
            # 计算MAE
            mae = mean_absolute_error(y_true, y_pred)
            
            # 基础回归指标
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mae,
                'r2': r2_score(y_true, y_pred)
            }
            
            # 计算MAE/STD - 均方误差与标准差之比
            # 使用偏移量避免除零问题
            if len(y_true) > 1:
                std = np.std(y_true) + std_epsilon  # 添加偏移量
                metrics['mae_std_ratio'] = mae / std
            else:
                metrics['mae_std_ratio'] = np.nan
            
            # 计算 sMAPE (Symmetric Mean Absolute Percentage Error)
            # 避免除零问题
            denominator = np.abs(y_true) + np.abs(y_pred)
            mask = denominator != 0  # 过滤掉分母为零的情况
            if np.sum(mask) > 0:
                smape = 200.0 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])
                metrics['smape'] = smape
            else:
                metrics['smape'] = np.nan
                
            # 计算 MASE (Mean Absolute Scaled Error)
            # MASE需要历史数据作为基准，如果提供了训练集数据
            if y_train is not None and len(y_train) > 1:
                # 计算朴素预测的MAE (使用前一个时间点的值作为预测)
                naive_errors = np.abs(np.diff(y_train))
                naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else np.nan
                
                # 如果朴素预测的MAE不为0或NaN，则计算MASE
                if naive_mae is not None and naive_mae != 0 and not np.isnan(naive_mae):
                    mase = mae / naive_mae
                    metrics['mase'] = mase
                else:
                    metrics['mase'] = np.nan
            else:
                metrics['mase'] = np.nan
                logger.warning("无法计算MASE: 需要训练集数据作为朴素预测基准")
                
            return metrics
        else:  # classification
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            }
        
    def calculate_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        使用Bootstrap方法计算分类指标的置信区间。
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            n_bootstraps: bootstrap采样次数
            confidence_level: 置信水平
            
        Returns:
            包含各指标及其置信区间的字典
        """
        if self.task_type != 'classification':
            logger.warning("置信区间计算仅适用于分类任务")
            return {}
            
        # 计算置信区间的索引
        alpha = (1 - confidence_level) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100
        
        # 初始化指标结果容器
        bootstrap_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # 将输入转换为numpy数组以避免索引问题
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        # Bootstrap采样计算
        for _ in range(n_bootstraps):
            # 随机有放回抽样索引
            indices = np.random.choice(len(y_true_array), len(y_true_array), replace=True)
            
            # 获取bootstrap样本
            y_true_bootstrap = y_true_array[indices]
            y_pred_bootstrap = y_pred_array[indices]
            
            # 计算并存储指标
            bootstrap_metrics['accuracy'].append(accuracy_score(y_true_bootstrap, y_pred_bootstrap))
            bootstrap_metrics['precision'].append(precision_score(y_true_bootstrap, y_pred_bootstrap))
            bootstrap_metrics['recall'].append(recall_score(y_true_bootstrap, y_pred_bootstrap))
            bootstrap_metrics['f1'].append(f1_score(y_true_bootstrap, y_pred_bootstrap))
        
        # 计算置信区间
        result = {}
        for metric_name, values in bootstrap_metrics.items():
            values = np.array(values)
            result[metric_name] = {
                'point': np.mean(values),
                'lower': np.percentile(values, lower_percentile),
                'upper': np.percentile(values, upper_percentile)
            }
            
        return result
        
    def evaluate_by_period(
        self,
        dates: pd.Series,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        period: str = 'M',
        y_train: np.ndarray = None  # 添加训练集参数
    ) -> Dict[str, Dict[str, float]]:
        """
        按时间周期评估模型性能
        
        Args:
            dates: 时间序列
            y_true: 真实值
            y_pred: 预测值
            period: 评估周期 ('M' 为月度, 'Q' 为季度，'Y' 为年度)
            y_train: 训练集真实值，用于计算MASE
            
        Returns:
            各时间周期的评估指标
        """
        # 转换时间周期
        if period == 'M':
            period_dates = dates.dt.to_period('M')
        elif period == 'Q':
            period_dates = dates.dt.to_period('Q')
        elif period == 'Y':
            period_dates = dates.dt.to_period('Y')
            
        # 存储结果
        period_metrics = {}
        
        # 对每个周期进行评估
        for p in sorted(period_dates.unique()):
            mask = period_dates == p
            if sum(mask) > 0:  # 确保该周期有数据
                # 计算基本指标，传入训练集数据用于MASE计算
                metrics = self.evaluate_metrics(y_true[mask], y_pred[mask], y_train)
                
                # 如果是分类任务并且样本数足够，计算置信区间
                if self.task_type == 'classification' and sum(mask) >= 10:
                    try:
                        # 计算置信区间
                        ci_metrics = self.calculate_confidence_intervals(
                            y_true[mask], 
                            y_pred[mask], 
                            n_bootstraps=500  # 对每个周期使用较少的bootstrap样本
                        )
                        
                        # 将置信区间添加到指标中
                        for metric_name, values in ci_metrics.items():
                            metrics[f'{metric_name}_lower'] = values['lower']
                            metrics[f'{metric_name}_upper'] = values['upper']
                    except Exception as e:
                        logger.warning(f"计算周期 {p} 的置信区间时出错: {str(e)}")
                
                period_metrics[str(p)] = metrics
                
        return period_metrics
        
    def evaluate_by_custom_periods(
        self,
        dates: pd.Series,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cutpoints: List[Union[str, pd.Timestamp, datetime]],
        save_predictions: bool = True,
        save_dir: Union[str, Path, None] = None,
        y_train: np.ndarray = None
    ) -> Dict[str, Dict[str, float]]:
        """
        按自定义时间节点评估模型性能
        
        Args:
            dates: 时间序列
            y_true: 真实值
            y_pred: 预测值
            cutpoints: 自定义时间截断点列表
            save_predictions: 是否保存真实值和预测值
            save_dir: 保存预测结果的目录，如果为None则使用当前目录
            y_train: 训练集真实值，用于计算MASE
            
        Returns:
            各自定义时间段的评估指标
        """
        logger.info(f"按自定义时间节点评估: {cutpoints}")
        
        # 确保所有截断点都是datetime类型
        cutpoints_dt = []
        for cp in cutpoints:
            if isinstance(cp, str):
                cutpoints_dt.append(pd.to_datetime(cp))
            else:
                cutpoints_dt.append(pd.Timestamp(cp))
                
        # 按时间排序截断点
        cutpoints_dt.sort()
        
        # 添加数据的起始和结束时间作为边界点
        all_points = [dates.min()] + cutpoints_dt + [dates.max()]
        
        # 存储结果
        period_metrics = {}
        
        # 创建保存目录（如果需要）
        if save_predictions and save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # 对每个时间段进行评估
        for i in range(len(all_points) - 1):
            start_time = all_points[i]
            end_time = all_points[i+1]
            
            # 创建时间段标识
            period_name = f"{start_time.strftime('%Y-%m-%d')}_{end_time.strftime('%Y-%m-%d')}"
            
            # 筛选该时间段的数据
            mask = (dates >= start_time) & (dates < end_time)
            
            if sum(mask) > 0:  # 确保该时间段有数据
                # 获取该时间段的真实值和预测值
                period_y_true = y_true[mask]
                period_y_pred = y_pred[mask]
                period_dates = dates[mask]
                
                # 计算基本指标
                metrics = self.evaluate_metrics(period_y_true, period_y_pred, y_train)
                
                # 如果是分类任务并且样本数足够，计算置信区间
                if self.task_type == 'classification' and sum(mask) >= 10:
                    try:
                        # 计算置信区间
                        ci_metrics = self.calculate_confidence_intervals(
                            period_y_true, 
                            period_y_pred, 
                            n_bootstraps=500  # 对每个时间段使用较少的bootstrap样本
                        )
                        
                        # 将置信区间添加到指标中
                        for metric_name, values in ci_metrics.items():
                            metrics[f'{metric_name}_lower'] = values['lower']
                            metrics[f'{metric_name}_upper'] = values['upper']
                    except Exception as e:
                        logger.warning(f"计算时间段 {period_name} 的置信区间时出错: {str(e)}")
                
                # 添加样本数量信息
                metrics['sample_count'] = int(sum(mask))
                
                # 保存真实值和预测值
                if save_predictions:
                    # 创建包含日期、真实值和预测值的DataFrame
                    predictions_df = pd.DataFrame({
                        'date': period_dates,
                        'y_true': period_y_true,
                        'y_pred': period_y_pred
                    })
                    
                    if save_dir is not None:
                        # 保存到指定目录
                        filename = f"predictions_{period_name.replace('-', '')}.csv"
                        file_path = Path(save_dir) / filename
                        predictions_df.to_csv(file_path, index=False)
                        logger.info(f"时间段 {period_name} 的预测结果已保存到: {file_path}")
                    else:
                        # 不保存文件，但将预测结果添加到指标中
                        metrics['predictions'] = predictions_df
                
                period_metrics[period_name] = metrics
                logger.info(f"时间段 {period_name} 包含 {sum(mask)} 个样本")
            else:
                logger.warning(f"时间段 {period_name} 没有数据")
                
        return period_metrics
        
    def analyze_temporal_feature_importance(
        self,
        data: pd.DataFrame,
        window_size: str = 'Y',
        sliding: bool = False,
        window_name: Union[str, None] = None,
        max_samples_per_window: int = 1000  # 添加参数控制每个窗口的最大样本数
    ) -> pd.DataFrame:
        """
        分析特征重要性的时间变化。
        
        Args:
            data: 包含特征和时间戳的数据框
            window_size: 时间窗口大小
                - 'M': 月度
                - '3M': 季度
                - '6M': 半年
                - 'Y': 年度
                - '2Y': 两年
            sliding: 是否使用滑动窗口
            window_name: 窗口名称，用于保存文件时的命名
            max_samples_per_window: 每个窗口用于计算SHAP值的最大样本数
            
        Returns:
            特征重要性随时间变化的数据框
        """
        window_name = window_name or window_size
        logger.info(f"开始时序特征重要性分析，窗口大小: {window_size}, 滑动窗口: {sliding}")
        
        # 准备数据
        X, y = self.prepare_features(data)
        dates = data['检查时间']

        # 训练最终模型
        logger.info("训练最终模型...")
        self.model.fit(X, y)
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(
            self.model,
            feature_perturbation='interventional'
        )
        
        # 设置SHAP计算参数
        shap_kwargs = {
            'check_additivity': False
        }

        # 按时间窗口划分数据
        if sliding:
            # 使用滑动窗口
            window_starts = pd.date_range(
                start=dates.min(),
                end=dates.max() - pd.Timedelta(window_size),
                freq=window_size
            )
            windows = [(start, start + pd.Timedelta(window_size)) for start in window_starts]
        else:
            # 使用固定窗口
            window_starts = pd.date_range(
                start=dates.min(),
                end=dates.max(),
                freq=window_size
            )
            windows = list(zip(window_starts[:-1], window_starts[1:]))
            
        # 存储每个窗口的特征重要性
        importance_over_time = []
        
        # 对每个窗口计算SHAP值
        for start, end in windows:
            logger.info(f"分析时间窗口: {start.strftime('%Y-%m-%d')} 到 {end.strftime('%Y-%m-%d')}")
            
            try:
                # 获取窗口数据
                mask = (dates >= start) & (dates < end)
                X_window = X[mask]
                
                if len(X_window) == 0:
                    logger.warning(f"窗口 {start} - {end} 没有数据")
                    continue
                    
                # 如果样本数超过限制，随机采样
                if len(X_window) > max_samples_per_window:
                    sample_indices = np.random.choice(
                        len(X_window),
                        max_samples_per_window,
                        replace=False
                    )
                    X_window = X_window.iloc[sample_indices]
                
                # 计算SHAP值
                try:
                    # 首先尝试使用check_additivity参数
                    shap_values = explainer.shap_values(X_window, **shap_kwargs)
                except TypeError:
                    # 如果失败，则不使用该参数
                    logger.warning("SHAP计算不支持check_additivity参数，使用默认设置")
                    shap_values = explainer.shap_values(X_window)
                
                # 计算平均绝对SHAP值
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # 存储结果
                importance_over_time.append({
                    'window_start': start,
                    'window_end': end,
                    'sample_size': len(X_window),
                    **{
                        feature: importance 
                        for feature, importance in zip(self.feature_names, mean_abs_shap)
                    }
                })
                
            except Exception as e:
                logger.error(f"处理窗口 {start} - {end} 时出错: {str(e)}")
                continue
            
        # 转换为数据框
        importance_df = pd.DataFrame(importance_over_time)
        
        return importance_df
        
    def analyze_custom_temporal_feature_importance(
        self,
        data: pd.DataFrame,
        cutpoints: List[Union[str, pd.Timestamp, datetime]],
        max_samples_per_period: int = 1000
    ) -> pd.DataFrame:
        """
        基于自定义时间截断点分析特征重要性的时间变化。
        
        Args:
            data: 包含特征和时间戳的数据框
            cutpoints: 自定义时间截断点列表
            max_samples_per_period: 每个时间段用于计算SHAP值的最大样本数
            
        Returns:
            特征重要性随时间变化的数据框
        """
        logger.info(f"开始基于自定义时间截断点的特征重要性分析: {cutpoints}")
        
        # 准备数据
        X, y = self.prepare_features(data)
        dates = data['检查时间']
        
        # 训练最终模型
        logger.info("训练最终模型...")
        self.model.fit(X, y)
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(
            self.model,
            feature_perturbation='interventional'
        )
        
        # 设置SHAP计算参数
        shap_kwargs = {
            'check_additivity': False
        }
        
        # 确保所有截断点都是datetime类型
        cutpoints_dt = []
        for cp in cutpoints:
            if isinstance(cp, str):
                cutpoints_dt.append(pd.to_datetime(cp))
            else:
                cutpoints_dt.append(pd.Timestamp(cp))
                
        # 按时间排序截断点
        cutpoints_dt.sort()
        
        # 添加数据的起始和结束时间作为边界点
        all_points = [dates.min()] + cutpoints_dt + [dates.max()]
        
        # 存储每个时间段的特征重要性
        importance_over_time = []
        
        # 对每个时间段计算SHAP值
        for i in range(len(all_points) - 1):
            start_time = all_points[i]
            end_time = all_points[i+1]
            
            # 创建时间段标识
            period_name = f"{start_time.strftime('%Y-%m-%d')}_{end_time.strftime('%Y-%m-%d')}"
            logger.info(f"分析时间段: {period_name}")
            
            try:
                # 获取该时间段的数据
                mask = (dates >= start_time) & (dates < end_time)
                X_period = X[mask]
                
                if len(X_period) == 0:
                    logger.warning(f"时间段 {period_name} 没有数据")
                    continue
                    
                # 如果样本数超过限制，随机采样
                if len(X_period) > max_samples_per_period:
                    sample_indices = np.random.choice(
                        len(X_period),
                        max_samples_per_period,
                        replace=False
                    )
                    X_period = X_period.iloc[sample_indices]
                
                # 计算SHAP值
                try:
                    # 首先尝试使用check_additivity参数
                    shap_values = explainer.shap_values(X_period, **shap_kwargs)
                except TypeError:
                    # 如果失败，则不使用该参数
                    logger.warning("SHAP计算不支持check_additivity参数，使用默认设置")
                    shap_values = explainer.shap_values(X_period)
                
                # 计算平均绝对SHAP值
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # 存储结果
                importance_over_time.append({
                    'period_start': start_time,
                    'period_end': end_time,
                    'period_name': period_name,
                    'sample_size': len(X_period),
                    **{
                        feature: importance 
                        for feature, importance in zip(self.feature_names, mean_abs_shap)
                    }
                })
                
                logger.info(f"时间段 {period_name} 包含 {len(X_period)} 个样本")
                
            except Exception as e:
                logger.error(f"处理时间段 {period_name} 时出错: {str(e)}")
                continue
            
        # 转换为数据框
        importance_df = pd.DataFrame(importance_over_time)
        
        return importance_df
        
    @translate_to_csv()
    def save_importance_to_csv(self, importance_df: pd.DataFrame, filepath: str) -> None:
        """
        保存特征重要性数据到CSV文件。
        
        Args:
            importance_df: 特征重要性DataFrame
            filepath: 保存路径
        """
        # 检查是否为新格式（宽格式）的数据框
        if hasattr(importance_df, 'compatibility_view'):
            # 保存宽格式数据框 - 特征名作为列名
            importance_df.to_csv(filepath, index=False)
        elif 'feature' in importance_df.columns and 'importance' in importance_df.columns:
            # 旧格式（长格式）的数据框
            importance_df.to_csv(filepath, index=False)
        else:
            # 假设为宽格式但没有兼容性视图
            importance_df.to_csv(filepath, index=False)

    def analyze_overall_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        分析整体数据集的特征重要性。
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            特征重要性数据框
        """
        logger.info("分析整体特征重要性...")
        
        # 训练模型（使用全量数据）
        self.model.fit(X, y)
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(
            self.model,
            feature_perturbation='interventional'
        )
        
        # 设置SHAP计算参数
        shap_kwargs = {
            'check_additivity': False
        }
        
        # 如果样本数过多，取随机子集进行SHAP计算
        max_samples = 10000
        if len(X) > max_samples:
            sample_indices = np.random.choice(
                len(X),
                max_samples,
                replace=False
            )
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
            
        # 计算SHAP值
        try:
            shap_values = explainer.shap_values(X_sample, **shap_kwargs)
        except TypeError:
            logger.warning("SHAP计算不支持check_additivity参数，使用默认设置")
            shap_values = explainer.shap_values(X_sample)
            
        # 计算平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # 创建特征重要性数据框 - 将特征名作为列名，重要性值作为数据行
        importance_dict = {feature: [importance] for feature, importance in zip(self.feature_names, mean_abs_shap)}
        importance_df = pd.DataFrame(importance_dict)
        
        # 为了保持与其他代码的兼容性，我们创建一个包含原始信息的副本
        # 按重要性排序列名（特征）
        importance_values = importance_df.iloc[0].to_dict()
        sorted_features = sorted(importance_values.keys(), key=lambda x: importance_values[x], reverse=True)
        importance_df = importance_df[sorted_features]
        
        # 创建兼容性数据框 - 包含feature和importance两列的长格式数据
        compatibility_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        })
        compatibility_df = compatibility_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # 添加兼容性数据作为属性
        setattr(importance_df, 'compatibility_view', compatibility_df)
        
        return importance_df
        
    def run_evaluation(
        self,
        custom_cutpoints_file: Union[str, None] = None
    ) -> None:
        """
        运行完整的评估流程
        
        Args:
            custom_cutpoints_file: 自定义时间节点JSON文件路径
        """
        # 创建保存目录
        task_type = 'regression' if self.task_type == 'regression' else f'classification_{self.model_type}'
        save_dir = f'../data/hospital_A/temporal_evaluation_{task_type}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        data = self.load_data()
        
        # 准备特征
        X, y = self.prepare_features(data)
        
        # 划分数据集
        X_train, X_test, y_train, y_test, dates_train, dates_test = self.split_data(
            X, y, data['检查时间']
        )
        
        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 训练模型
        logger.info(f"开始训练{self.task_type}模型 ({self.model_type})...")
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算整体指标
        overall_metrics = self.evaluate_metrics(y_test, y_pred, y_train)
        logger.info(f"整体评估指标: {overall_metrics}")
        
        if self.task_type == 'classification':
            # 打印分类报告
            logger.info("\n分类报告:")
            logger.info(classification_report(y_test, y_pred))
            
            # 计算置信区间
            logger.info("计算95%置信区间...")
            ci_metrics = self.calculate_confidence_intervals(y_test, y_pred)
            
            # 输出置信区间结果
            for metric, values in ci_metrics.items():
                logger.info(f"{metric.upper()}: {values['point']:.4f} (95% CI: {values['lower']:.4f}-{values['upper']:.4f})")
                
            # 保存置信区间结果
            ci_df = pd.DataFrame({
                'metric': list(ci_metrics.keys()),
                'point': [values['point'] for values in ci_metrics.values()],
                'lower_95ci': [values['lower'] for values in ci_metrics.values()],
                'upper_95ci': [values['upper'] for values in ci_metrics.values()]
            })
            ci_df.to_csv(f'{save_dir}/classification_metrics_95ci.csv', index=False)
            logger.info(f"置信区间结果已保存到: {save_dir}/classification_metrics_95ci.csv")
        
        # 按月评估
        monthly_metrics = self.evaluate_by_period(dates_test, y_test, y_pred, 'M', y_train)
        monthly_df = pd.DataFrame.from_dict(monthly_metrics, orient='index')
        self.save_metrics_to_csv(monthly_df, f'{save_dir}/monthly_metrics.csv')
        '''
        # 按季度评估
        quarterly_metrics = self.evaluate_by_period(dates_test, y_test, y_pred, 'Q', y_train)
        quarterly_df = pd.DataFrame.from_dict(quarterly_metrics, orient='index')
        self.save_metrics_to_csv(quarterly_df, f'{save_dir}/quarterly_metrics.csv')

        # 按年评估
        yearly_metrics = self.evaluate_by_period(dates_test, y_test, y_pred, 'Y', y_train)
        yearly_df = pd.DataFrame.from_dict(yearly_metrics, orient='index')
        self.save_metrics_to_csv(yearly_df, f'{save_dir}/yearly_metrics.csv')
        
        

        # 如果没有提供自定义文件，使用默认的时间节点
        logger.info("使用默认自定义时间节点进行评估")
        custom_cutpoints = [
            '2015-12-31', 
            '2018-8-31',
        ]
        
        custom_metrics = self.evaluate_by_custom_periods(
            dates_test, 
            y_test, 
            y_pred, 
            custom_cutpoints, 
            save_dir=save_dir,
            y_train=y_train  # 传递训练集数据
        )
        custom_df = pd.DataFrame.from_dict(custom_metrics, orient='index')
        self.save_metrics_to_csv(custom_df, f'{save_dir}/custom_period_metrics_default.csv')
        
        # 基于默认自定义时间节点进行特征重要性分析
        logger.info("基于默认自定义时间节点进行特征重要性分析...")
        custom_importance = self.analyze_custom_temporal_feature_importance(
            data,
            custom_cutpoints,
            max_samples_per_period=100000
        )
        # 保存特征重要性数据
        custom_importance_path = str(Path(save_dir) / f'custom_feature_importance_default.csv')
        self.save_importance_to_csv(custom_importance, custom_importance_path)
        
        # 分析整体特征重要性
        logger.info("分析整体特征重要性...")
        overall_importance = self.analyze_overall_feature_importance(X, y)
        # 保存特征重要性数据
        csv_path = str(Path(save_dir) / f'overall_feature_importance_{task_type}.csv')
        self.save_importance_to_csv(overall_importance, csv_path)
        
        # 时序特征重要性分析
        logger.info("开始时序特征重要性分析...")
        
        # 定义不同的时间窗口配置
        window_configs = [
            ('M', 'Monthly'),
            # ('3M', 'Quarterly'),
            # ('6M', 'Semi-Annual'),
            # ('Y', 'Annual'),
            # ('2Y', 'Biennial')
        ]
        
        # 对每种窗口大小进行分析
        for window_size, window_name in window_configs:
            try:
                # 固定窗口分析
                importance_df = self.analyze_temporal_feature_importance(
                    data,
                    window_size=window_size,
                    sliding=False,
                    window_name=window_name,
                    max_samples_per_window=1000000  # 限制每个窗口的样本数
                )
                # 保存特征重要性数据
                fixed_filename_prefix = f"temporal_feature_importance_{window_name}_Fixed".lower()
                csv_path = str(Path(save_dir) / f'{fixed_filename_prefix}.csv')
                self.save_importance_to_csv(importance_df, csv_path)
                
                # 滑动窗口分析
                sliding_importance = self.analyze_temporal_feature_importance(
                    data,
                    window_size=window_size,
                    sliding=True,
                    window_name=window_name,
                    max_samples_per_window=100000  # 限制每个窗口的样本数
                )
                # 保存特征重要性数据
                sliding_filename_prefix = f"temporal_feature_importance_{window_name}_Sliding".lower()
                csv_path = str(Path(save_dir) / f'{sliding_filename_prefix}.csv')
                self.save_importance_to_csv(sliding_importance, csv_path)
            except Exception as e:
                logger.error(f"分析窗口 {window_name} 时出错: {str(e)}")
                continue
        '''
        logger.info("评估完成")

    def run_special_evaluation(self) -> None:
        """运行特殊评估流程"""
        # 创建保存目录
        save_dir = '../data/hospital_A/temporal_evaluation_after2017'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        data = self.load_data()
        
        # 过滤2018年后的数据
        data = data[data['检查时间'] >= '2017-01-01']
        
        # 准备特征  
        X, y = self.prepare_features(data)
        
        # 划分数据集
        X_train, X_test, y_train, y_test, dates_train, dates_test = self.split_data(
            X, y, data['检查时间']
        )   
        
        # 训练模型
        logger.info("开始训练模型...")
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算整体指标
        overall_metrics = self.evaluate_metrics(y_test, y_pred)
        logger.info(f"整体评估指标: {overall_metrics}")
        
        # 按月评估，传递训练集数据用于MASE计算
        monthly_metrics = self.evaluate_by_period(dates_test, y_test, y_pred, 'M', y_train)
        monthly_df = pd.DataFrame.from_dict(monthly_metrics, orient='index')
        self.save_metrics_to_csv(monthly_df, f'{save_dir}/monthly_metrics.csv')

    @translate_to_csv()
    def save_metrics_to_csv(self, metrics_df: pd.DataFrame, filepath: str) -> None:
        """
        保存指标到CSV文件。
        
        Args:
            metrics_df: 包含评估指标的DataFrame
            filepath: 保存路径
        """
        # 如果是分类任务，添加注释说明置信区间
        if self.task_type == 'classification':
            # 检查是否包含置信区间列
            has_ci = any('_lower' in col for col in metrics_df.columns)
            
            if has_ci:
                # 创建新的DataFrame，保留原始数据
                metrics_df_with_notes = metrics_df.copy()
                
                # 添加注释列
                notes = ["CI表示95%置信区间" if i == 0 else "" for i in range(len(metrics_df))]
                metrics_df_with_notes['注释'] = notes
                
                # 保存带注释的DataFrame
                metrics_df_with_notes.to_csv(filepath)
                return
        
        # 默认保存原始DataFrame
        metrics_df.to_csv(filepath)

if __name__ == "__main__":
    try:

        # XGBoost分类任务评估
        
        #evaluator_xgb_cls = TemporalEvaluator(task_type='classification', model_type='xgboost')
        #evaluator_xgb_cls.run_evaluation()


        
        # 回归任务评估
        evaluator_reg = TemporalEvaluator(task_type='regression')
        evaluator_reg.run_evaluation()
        

        
        # RandomForest分类任务评估
        #evaluator_rf_cls = TemporalEvaluator(task_type='classification', model_type='randomforest')
        #evaluator_rf_cls.run_evaluation()
        
    except Exception as e:
        logger.exception(f"评估过程中出错: {str(e)}") 