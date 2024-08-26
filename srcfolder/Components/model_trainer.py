import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",  "model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_training(self,train_Array,test_array):
        try:
            logging.info("Splitting training and test input")
            #Step1:First we will split the transformed train and test data into x_train, y_train, x_test, y_test
            X_train,y_train,X_test,y_test = (
                train_Array[:,:-1],
                train_Array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            #step02: making the dictionary of the models to test upon
            models = {
                "RandomForest" : RandomForestRegressor(),
                "Logistic Regressor": LogisticRegression(),
                "Support Vector Machine":SVC(),
                "AdaBoost" : AdaBoostRegressor(),
                "Naive Bayes":GaussianNB,
                "Decision Tree" :DecisionTreeRegressor(),
                "KNN Classifier" : KNeighborsClassifier(),
                
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    
                },
                "RandomForest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regressor":{
                    'penalty': ['l2', 'l1', 'elasticnet', 'none'],
                    'C': [0.1, 1, 10, 100],
                    'max_iter': [100, 200, 300]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                },
                "SVM": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                },
                "Naive Bayes": {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  
                }
    }
                
            
            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models = models,params = params)
            #Finding the best model:
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
        
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with acceptable performance.")
            logging.info(f"best model found:{best_model_name} with score: {best_model_score}" )
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            print("R^2 score of the best model: ",r2_square)
        except Exception as e:
            raise CustomException(e,sys)


            