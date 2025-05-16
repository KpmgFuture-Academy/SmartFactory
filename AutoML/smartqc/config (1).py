#GLOBAL CONFIGURATION DICTIONARY

CONFIG = {
    # General
    "batch_size": 100000,               # if dataset exceeds this size --> pandas chunking
    "random_state": 42,                 # reproducibility for models
    "verbose": True,                    # print extra logs or not

    # Data Quality Stage
    "anomaly_detection": True,          # enable or disable anomaly detection step
    "inf_handling": "knn",              # options: "replace_nan", "drop"

    # Preprocessing Stage
    "columns_to_drop": [],              # user can list columns to drop here
    "scale_numeric": True,              # enable numeric scaling (StandardScaler)
    
    # SMOTE Balancer Stage
    "use_smote": True,                  # only applies for classification tasks
    "smote_sampling_strategy": 0.5,     # 0.5 = balance to 50%

    # Target Selection Stage
    "auto_target_selection": False,     # system will not auto-guess target columns
    "target_column_names": [],          # OR manually define list of target column names

    # Pipeline Routing Stage
    "time_series_threshold": 0.3,       # if dataframe has datetime column + correlation > X → use timeseries

    # AutoML Stage
    "max_trials": 10,                   # max number of trials for Optuna hyperparameter tuning
    "automl_timeout_minutes": 5,       # stop tuning if exceeds N minutes
}
