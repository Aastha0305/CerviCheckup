#importing the required libraries
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import datatransformationconfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
import pandas as pd

#Step01: first we will create a class data_ingestion_config which will define the paths to store the raw data,train data, and test data
#decorator
@dataclass
class DataIngestionConfig:
    raw_file_path : str = os.path.join('artifacts', "raw.csv")
    train_file_path : str = os.path.join('artifacts', "train.csv")
    test_file_path : str = os.path.join('artifacts', "test.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or componenet")
        try:
            df = pd.read_csv(r'Notebook\data\cervicalcancer.csv')
            logging.info("Read the data set")
            os.makedirs(os.path.dirname(self.ingestion_config.train_file_path),exist_ok = True)
            #Step 02: Converting the data into csv and stroing to the paths 
            df.to_csv(self.ingestion_config.raw_file_path, index = False, header = True)
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            train_set.to_csv(self.ingestion_config.train_file_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_file_path,index = False, header = True)
            logging.info("Data Ingestion completed")
            return(
                #self.ingestion_config.raw_file_path,
               self.ingestion_config.train_file_path,
               self.ingestion_config.test_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
#time to run the file
if __name__ == "__main__":
    #create an object
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformer_object(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr,test_arr))