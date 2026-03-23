# DATA TRANSFORMATION (FEATURE ENGINEERING + CLEANING)

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        # Define once → reuse everywhere
        self.numerical_columns = ["writing_score", "reading_score"]
        self.categorical_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

    def get_data_transformer_object(self):
        try:
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical pipeline (fixed)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            logging.info(f"Numerical columns: {self.numerical_columns}")
            logging.info(f"Categorical columns: {self.categorical_columns}")

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, self.numerical_columns),
                    ("cat_pipeline", cat_pipeline, self.categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully.")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"

            # Split input and target
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing on train and test data.")

            # FIT only on train
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # TRANSFORM on test (no leakage)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Save preprocessor
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info("Preprocessing object saved successfully.")

            return (
                input_feature_train_arr,
                input_feature_test_arr,
                target_feature_train_df.values,
                target_feature_test_df.values,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)