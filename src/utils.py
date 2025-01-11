import sys
import os
import dill
import numpy as np
import cv2
from src.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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

def evaluate_model(true, predicted):
    # Accuracy
    accuracy = accuracy_score(true, predicted)
    
    # Precision (micro, macro, weighted average for multi-class)
    precision = precision_score(true, predicted, average='weighted')
    
    # Recall (micro, macro, weighted average for multi-class)
    recall = recall_score(true, predicted, average='weighted')
    
    # F1-Score (micro, macro, weighted average for multi-class)
    f1 = f1_score(true, predicted, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(true, predicted)
    
    return accuracy, precision, recall, f1, cm
    

