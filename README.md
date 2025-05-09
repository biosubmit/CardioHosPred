# CardioHosPred

A machine learning project for hospital-based cardiovascular prediction models across multiple hospitals.

## Project Overview

This project analyzes medical data from multiple hospitals (A, B, and C) to build time-series predictive models for cardiovascular outcomes. The project includes data preprocessing, model training, evaluation, and analysis of feature importance across different hospitals and time periods.

## Directory Structure

```
CardioHosPred/
├── data/                      # Data storage
│   ├── hospital_A/            # Hospital A specific data
│   ├── hospital_B/            # Hospital B specific data
│   └── hospital_C/            # Hospital C specific data
├── src/                       # Source code
│   ├── data_preprocessing/    # Data preprocessing scripts
│   │   ├── hospital_A/        # Hospital A preprocessing
│   │   ├── hospital_B/        # Hospital B preprocessing
│   │   └── hospital_C/        # Hospital C preprocessing
│   └── model/                 # Model training and evaluation
│       ├── hospital_A/        # Hospital A models
│       ├── hospital_B/        # Hospital B models
│       └── hospital_C/        # Hospital C models
└── tests/                     # Test scripts (to be added)
```

## Features

- **Time-based Analysis**: Split data by inspection time in half-year steps and train models for each time period
- **Feature Importance Analysis**: Identify and visualize important features for prediction
- **ROC Curve Generation**: Generate and save ROC curves for model evaluation
- **Cross-hospital Comparison**: Compare model performance across different hospitals

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd CardioHosPred
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

To preprocess data for a specific hospital:

```bash
python src/data_preprocessing/hospital_A/preprocess.py
```

### Model Training

To train a model for a specific hospital and task:

```bash
python src/model/hospital_A/train_hospital_A_task1.py
```

### Model Evaluation

To evaluate a trained model:

```bash
python src/model/hospital_A/evaluate_hospital_A_task2.py
```

## Data Description

The project uses medical data from three hospitals, including patient demographics, medical history, laboratory tests, and cardiovascular outcomes. Specific features include:

- Patient demographics (age, gender)
- Medical history
- Laboratory test results
- Temporal data (inspection time, admission time)
- Outcome measures

## Model Description

The project primarily uses Random Forest classifiers to predict cardiovascular outcomes. The models are trained on different time periods to analyze temporal patterns in prediction performance.

## Results

Results from the model training and evaluation are saved in the respective hospital directories under `data/`. These include:

- ROC curves
- Feature importance analyses
- Performance metrics

## License

[Specify the license here]

## Contributors

[List the contributors here]

## Acknowledgments

[Acknowledgments for data sources, collaborators, etc.] 