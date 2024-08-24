import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,

)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object
from src.utlis import evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",  "model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_training(self,train_Array,test_array):
        try:
            logging.info("Splitting training and test input")
            #Step1:First wewill split the transformed train and test data into x_train, y_train, x_test, y_test
            X_train,y_train,X_test,y_test = (
                train_Array[:,:-1],
                train_Array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            #step02: making the dictionary of the models to test upon
            models = {
                "RandomForest" : RandomForestRegressor(),
                "Linear Regressor": LinearRegression(),
                "Catboost" : CatBoostRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "Gradient Boost" : GradientBoostingRegressor(),
                "Decision Tree" :DecisionTreeRegressor(),
                #"KNN Regeressor" : KNeighborsRegressor(),
                "XGBOOST" : XGBRegressor()
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regressor":{},
                "XGBOOST":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Catboost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models = models,params = params)
            #Finding the best model:
            best_model_score = max(sorted(model_report.values()))
            #getting the best model name;
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            #checking if none of  the applied mdel gives the good accuracy
            best_model = models[best_model_name]
            print(best_model)
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"best model found for both test and train {best_model}" )
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            print(r2_square)
        except Exception as e:
            raise CustomException(e,sys)


            