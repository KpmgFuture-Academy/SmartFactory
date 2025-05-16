# smartqc/pipeline_router.py

from sklearn.utils.multiclass import type_of_target
import pandas as pd

class PipelineRouter:
    def __init__(self, df, target_columns):
        """
        Determines model task type:
        - Regression
        - Classification
        - Time Series (if timestamp column present)
        """
        self.df = df.copy()
        self.target_columns = target_columns

    def detect_task_type(self):
        """
        Detect based on target column(s).
        """
        if not self.target_columns:
            raise ValueError("No target columns provided.")

        y = self.df[self.target_columns]
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.squeeze()

        task_type = type_of_target(y)

        if task_type in ['binary', 'multiclass']:
            print("Detected task type: Classification")
            return "classification"
        elif task_type in ['continuous', 'continuous-multioutput']:
            print("Detected task type: Regression")
            return "regression"
        else:
            print(f"\nAmbiguous or unknown task type detected: {task_type}")
            print("This may be due to unusual values or multi-output formats.")
            user_choice = input("Would you like to proceed with (R)egression or (C)lassification? ").strip().lower()
            
            if user_choice == 'r':
                print("User selected: Regression")
                return "regression"
            elif user_choice == 'c':
                print("User selected: Classification")
                return "classification"
            else:
                print("Invalid input. Defaulting to 'unknown'.")
                return "unknown"

