import os 
import sys
import pandas as pd
from dataclasses import dataclass
import numpy as np
import cv2
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, resize_image
from sklearn.utils import shuffle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self, img_size=(32, 32)):
        self.img_size = img_size
    
    def preprocess_image(self, image_path):
        try:
            resized_image = resize_image(image_path,self.img_size)
            image = resized_image.flatten()
            return image
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_data(self, data_dir):
        try:
            images = []
            labels = []
            class_names = sorted(os.listdir(data_dir))
            logging.info('The Preprocessing is started!')
            for label, class_name in enumerate(class_names):
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, fname)
                    if image_path.lower().endswith(('jpg')):
                        image = self.preprocess_image(image_path)
                        images.append(image)
                        labels.append(label)
                        
            # Convert to numpy arrays and shuffle
            images = np.array(images)
            labels = np.array(labels)
            images, labels = shuffle(images, labels, random_state=42)
            logging.info("The Preprocessing is completed")
            return images, labels
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__ == "__main__":
    obj = DataTransformation()
    images, labels = obj.load_data("notebook\cropped")
    print(f"The dataset has {len(images)} images")
    print('some labels:', labels[:10])