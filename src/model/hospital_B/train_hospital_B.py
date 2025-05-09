# Training script for Hospital B

import os
import sys
#from data_preprocessing.hospital_B.preprocessing import preprocess_data
#from models.his_boosting import HisBoosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelslib.rnn import RNNClassifier
from modelslib.lstm import LSTMClassifier
from modelslib.gru import GRUClassifier
from modelslib.transformer import TransformerClassifier
import numpy as np



def split_data_module_ml(random_state=42):
    X=pd.read_csv('./data/MIMIC/preprocessed_data/data_X_for_model.csv')
    y=pd.read_csv('./data/MIMIC/preprocessed_data/data_y_for_model.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test



def train_hospital_B(model_name, model, save_dir='./data/MIMIC'):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = split_data_module_ml()

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    model_save_path = f'{save_dir}/models/{model_name}_model.pkl'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Model {model_name} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Get metrics from classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = report['weighted avg']
    
    # Save the results by appending to existing file
    result_save_path = f'{save_dir}/results/all_result.csv'
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    result = pd.DataFrame({
        'model_name': [model_name],
        'precision': [metrics['precision']],
        'recall': [metrics['recall']],
        'f1-score': [metrics['f1-score']],
        'accuracy': [accuracy_score(y_test, y_pred)]
    })
    
    # Create file if it doesn't exist, otherwise append without headers
    if not os.path.exists(result_save_path):
        result.to_csv(result_save_path, index=False)
    else:
        result.to_csv(result_save_path, mode='a', header=False, index=False)

def train_multiple_models(save_dir='./data/MIMIC'):
    """
    Train and evaluate multiple machine learning models
    """
    # Dictionary of models to train
    X_train, X_test, y_train, y_test = split_data_module_ml()
    models = {
        'hist_gradient_boosting': HistGradientBoostingClassifier(),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'svm': SVC(random_state=42),
        'rnn': RNNClassifier(
            input_size=X_train.shape[1],  # Number of features
            hidden_size=64,
            output_size=len(np.unique(y_train)),  # Number of classes
            num_layers=2,
            num_epochs=200,
            batch_size=32,
            learning_rate=0.001
        ),
        'lstm': LSTMClassifier(
            input_size=X_train.shape[1],  # Number of features
            hidden_size=64,
            output_size=len(np.unique(y_train)),  # Number of classes
            num_layers=2,
            num_epochs=200,
            batch_size=32,
            learning_rate=0.001
        ),
        'gru': GRUClassifier(
            input_size=X_train.shape[1],  # Number of features
            hidden_size=64,
            output_size=len(np.unique(y_train)),  # Number of classes
            num_layers=2,
            num_epochs=200,
            batch_size=32,
            learning_rate=0.001
        ),
        'transformer': TransformerClassifier(
            input_size=X_train.shape[1],  # Number of features
            hidden_size=64,
            output_size=len(np.unique(y_train)),  # Number of classes
            num_layers=2,
            num_epochs=200,
            batch_size=32,
            learning_rate=0.001
        )
    }
    
    # Train each model
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        try:
            train_hospital_B(model_name, model, save_dir)
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
        print(f"Finished training {model_name}")
        print('='*50)

# Example usage in if __name__ == "__main__":
if __name__ == "__main__":
    # Import necessary models
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC


    
    # Run all models
    train_multiple_models()
    
    # Or train a specific model
    # model = HistGradientBoostingClassifier()
    # train_hospital_B('his_boosting', model)