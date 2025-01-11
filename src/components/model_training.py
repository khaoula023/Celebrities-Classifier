import os
import sys
from dataclasses import dataclass
import cv2
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

# Modelling
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        

    def initiate_model_trainer(self, images, labels):
        try:
            logging.info('Split Training and test input data')    
            X_train,X_test ,y_train,y_test =train_test_split(images, labels, train_size=0.8)
            print(f'train: {X_train.shape}, Test: {X_test.shape}')
            models = {
                "KNN": KNeighborsClassifier(n_neighbors=3),
                "SVM": SVC(kernel='rbf'),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42),
                "Random Forest" : RandomForestClassifier(n_estimators=100, random_state=42)
            }   
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models) 
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score <0.6:
                raise CustomException('No best model found')
            logging.info('Best found model on both training and testing dataet')
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            predicted = best_model.predict(X_test)
            ACC_score = accuracy_score(y_test, predicted)
            return ACC_score    
        except Exception as e:
            raise CustomException(e, sys)