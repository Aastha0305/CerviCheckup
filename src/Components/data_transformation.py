import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'C:/Users/Jatin/Desktop/Aastha/Python-ml/project')
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    processor_obj_file_path = os.path.join('artifacts', "Processor.pkl")

# Custom transformers
class ConvertToNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_to_convert = [
            'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
            'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives',
            'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)',
            'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
            'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
            'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B',
            'STDs:HPV', 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'
        ]
        X[cols_to_convert] = X[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        return X

class ReplaceMissing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.replace('?', np.nan, inplace=True)
        return X


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            features = [
                "Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies", "Smokes", 
                "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives", "Hormonal Contraceptives (years)", 
                "IUD", "IUD (years)", "STDs", "STDs (number)", "STDs:condylomatosis", "STDs:cervical condylomatosis", 
                "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", 
                "STDs:pelvic inflammatory disease", "STDs:genital herpes", "STDs:molluscum contagiosum", 
                "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV", "STDs: Number of diagnosis", 
                "STDs: Time since first diagnosis", "STDs: Time since last diagnosis", "Dx:CIN", "Dx:HPV", 
                "Dx", "Hinselmann", "Schiller", "Citology", "Biopsy"
            ]

            # Define the pipeline
            pipeline = Pipeline(
                steps=[
                    ("convert_numeric", ConvertToNumeric()),
                    ("replace_missing", ReplaceMissing()),
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info(f"Features: {features}")

            preprocessor = ColumnTransformer(
                [
                    ("pipeline", pipeline, features),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformer_object(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read the training and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column = "Dx:Cancer"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing dataframe")
            try:
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            except AttributeError as e:
                raise CustomException(f"AttributeError in pipeline: {str(e)}", sys)
            except Exception as e:
                raise CustomException(f"An error occurred during pipeline processing: {str(e)}", sys)

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Apply SMOTE
            smote = SMOTE(sampling_strategy='minority')
            input_feature_train_arr, target_feature_train_df = smote.fit_resample(input_feature_train_arr, target_feature_train_df)

            # Apply PCA
            pca = PCA(n_components=15)
            input_feature_train_arr = pca.fit_transform(input_feature_train_arr)
            input_feature_test_arr = pca.transform(input_feature_test_arr)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.processor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.processor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)

