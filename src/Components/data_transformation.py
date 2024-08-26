import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_object
#Step1: To store the model as pickle file
@dataclass
class datatransformationconfig:
    processor_obj_file_path = os.path.join('artifacts', "Processor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = datatransformationconfig()
    def convert_to_numeric(self, df):
        cols_to_convert = [
            'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
            'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives',
            'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)',
            'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
            'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
            'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B',
            'STDs:HPV', 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'
        ]
        df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
        return df

    def replace_missing(self, df):
        df.replace('?', np.nan, inplace=True)
        return df

    def calculate_total_std(self, df):
        std_cols = [
            'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
            'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
            'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV'
        ]
        df["total_std"] = df[std_cols].sum(axis=1)
        return df

    def calculate_total_tests(self, df):
        test_cols = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
        df["total_tests"] = df[test_cols].sum(axis=1)
        return df

    def convert_columns_to_int(self, df):
        to_int = [
            "total_tests", "total_std", "Smokes", "Biopsy", "Dx:Cancer", "Num of pregnancies",
            "Number of sexual partners", "First sexual intercourse", "Hormonal Contraceptives",
            "IUD", "STDs", "STDs (number)", "STDs: Number of diagnosis", "Dx:CIN", "Dx:HPV",
            "Dx", "Hinselmann", "Schiller", "Biopsy", "Citology"
        ]
        df[to_int] = df[to_int].fillna(0).astype(int)
        return df

    def get_data_transformer_object(self):
        try:
            
            features = ["Age","Number of sexual partners","First sexual intercourse","Num of pregnancies","Smokes","Smokes (years)","Smokes (packs/year)","Hormonal Contraceptives","Hormonal Contraceptives (years)","IUD","IUD (years)","STDs","STDs (number)","STDs:condylomatosis","STDs:cervical condylomatosis","STDs:vaginal condylomatosis","STDs:vulvo-perineal condylomatosis","STDs:syphilis","STDs:pelvic inflammatory disease","STDs:genital herpes","STDs:molluscum contagiosum","STDs:AIDS","STDs:HIV","STDs:Hepatitis B","STDs:HPV","STDs: Number of diagnosis","STDs: Time since first diagnosis","STDs: Time since last diagnosis","Dx:CIN","Dx:HPV","Dx","Hinselmann","Schiller","Citology","Biopsy"]
            
            #Step 02: Now we will be creating two pielines to get our work done in flow and systematically
            pipeline = Pipeline(
                steps=[
                    ("convert_numeric", FunctionTransformer(self.convert_to_numeric)),
                    ("replace_missing", FunctionTransformer(self.replace_missing)),
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("age_category", FunctionTransformer(self.age_category)),
                    ("total_std", FunctionTransformer(self.calculate_total_std)),
                    ("total_tests", FunctionTransformer(self.calculate_total_tests)),
                    ("convert_to_int", FunctionTransformer(self.convert_columns_to_int)),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info(f"Features : {features}")
            
            #now we will use column Transformer, we will combine pieplines, cat_pipeline and numerical piepeline, and this will trandofrm the data  of diff type in one go
            preprocessor = ColumnTransformer(
                [
                    ("pipeline", pipeline, features),
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
            target_column = "Dx:Cancer"
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