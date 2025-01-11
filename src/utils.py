import sys
import os
import dill
import numpy as np
import cv2
from src.exception import CustomException
from sklearn.metrics import accuracy_score

def resize_image(image_path, img_size):
        try:
            image = cv2.imread(image_path)
            # Resize the image to the given size
            resized_image = cv2.resize(image, img_size)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                
            resized_image = resized_image / 255
            return resized_image
        except Exception as e:
            raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)    
    

