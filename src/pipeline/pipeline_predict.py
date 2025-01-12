import sys
import os
import cv2
import numpy as np

from src.exception import CustomException
from src.utils import load_object, detect_face
from src.components.Data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, image_path):
        try: 
            labels =[]
            results = []
            # load artifacts:
            model_path = 'artifacts\model.pkl'
            classes_path = 'artifacts\Classes.pkl'
            model = load_object(file_path= model_path)
            classes = load_object(file_path= classes_path)
            
            # preprocessing: 
            preprocessor = DataTransformation()
            faces = detect_face(image_path)
            for face in faces:
                image = preprocessor.preprocess_image(face)
                image = image.reshape(1,-1)
                pred = model.predict(image)
                label = [k for k, v in classes.items() if v == pred[0]]
                labels.append(label)
                results.append(pred[0])
            labels = sum(labels, [])  
            print(labels)  
            return labels
        except Exception as e:
            raise CustomException(e, sys)