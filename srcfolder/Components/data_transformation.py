import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utlis import save_object
#Step1: To store the model as pickle file
@dataclass
class datatransformationconfig:
    processor_obj_file_path = os.path.join('artifacts', "Processor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = datatransformationconfig()
    def get_data_transformer_object(self):
        try:
            num_features = ["reading_score", "writing_score"]
            cat_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            #Step 02: Now we will be creating two pielines to get our work done in flow and systematically
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")),
                    ("scalar", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean = False)),

                ]
            )
            logging.info(f"Numerical features : {num_features}")
            logging.info(f"Categorical features: {cat_features}")
            #now we will use column Transformer, we will combine pieplines, cat_pipeline and numerical piepeline, and this will trandofrm the data  of diff type in one go
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformer_object(self, train_path,test_path):
        try:
            ##step01: Read the data and call th e get_data_tranformation_object which will  return the preprocessor, means the transformed featuures
            #readint the data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read the training and test data completed")
            logging.info("obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column = "math_score"
            num_columns = ["writing_score","reading_score"]
            input_feature_train_df = train_df.drop(columns = [target_column], axis = 1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns = [target_column],axis =1)
            target_feature_test_df = test_df[target_column]
            logging.info("Applying preprocessing object on training and testing dataframe")
            input_feature_train_Arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_Arr = preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_Arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_Arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object")
            save_object(
                file_path = self.data_transformation_config.processor_obj_file_path,
                obj = preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.processor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)