# CardioHosPred

This repository contains the code accompanying our manuscript on modeling hospital admission behaviors using multi-source medical data. The project integrates data from three hospitals—Wuhan Union Hospital (Hospital A), the publicly available MIMIC-III dataset (Hospital B), and Shandong Cancer Hospital (Hospital C)—to develop predictive models for hospital admission timing and cardiovascular-related outcomes.

The repository includes complete pipelines for data preprocessing, time-aligned sequence modeling, evaluation, and feature importance analysis. Key tasks include same-day admission classification, 15-day and 365-day readmission prediction, and regression-based estimation of time gaps between diagnostic tests and hospital admissions.

Due to privacy restrictions, raw datasets cannot be released. Researchers interested in reproducing our results or accessing the data should contact the corresponding author. Preprocessed versions of the datasets (excluding raw identifiable fields) are stored under `data/hospital_X/tmp_preprocessed_data/` and used throughout the modeling pipeline.

## Directory Structure



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

- **Time-aligned Modeling**: Align test and admission records on a per-patient timeline
- **Feature Engineering**: Incorporate historical hospitalizations, socioeconomic indicators, and lab value normalization
- **Task Coverage**:
  - Same-day admission classification
  - 15-day and 365-day readmission prediction
  - Time-gap regression for next admission
- **Temporal Evaluation**: Analyze performance trends across months and custom-defined policy periods
- **SHAP-Based Feature Selection**: Temporal feature importance computed and visualized

## Installation

```bash
git clone <repository-url>
cd CardioHosPred
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## Usage

### Data Preprocessing

To preprocess data for a specific hospital:

```bash
python src/data_preprocessing/hospital_A/preprocess_pipeline.py
python src/data_preprocessing/hospital_B/preprocess_pipeline.py
python src/data_preprocessing/hospital_C/preprocess_pipeline.py
```

### Model Training

To train a model for a specific hospital and task:

```bash
python src/model/hospital_A/train_hospital_A_task1.py
```

### Model Evaluation

To evaluate a trained regression model in monthly timestep:

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

Results from the model training and evaluation are saved in the respective hospital directories under `Result/`. These include:

- ROC curves
- Feature importance analyses
- Performance metrics

## License

[Specify the license here]

## Contributors

[List the contributors here]

## Acknowledgments

[Acknowledgments for data sources, collaborators, etc.] 