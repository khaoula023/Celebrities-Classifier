import sys
import os
import dill
import numpy as np
import cv2
import mediapipe as mp
from src.exception import CustomException
from sklearn.metrics import accuracy_score

def resize_image(input_image, img_size):
        try:
            if isinstance(input_image, str):
                # Load the image from the file path
                image = cv2.imread(input_image)
                if image is None:
                    raise ValueError(f"Failed to load image from path: {input_image}")
            elif isinstance(input_image, np.ndarray):
                image = input_image
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

# Function to detect faces in a given image:    
def detect_face(image_path):
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return None  # Return None to indicate no faces were detected or an error occurred
        
        # Read the image
        img = cv2.imread(image_path)
        
        # Check if image was loaded correctly
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Convert the image to RGB (MediaPipe uses RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe face detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
        # Perform face detection
        results = face_detection.process(img_rgb)
        
        # List to store cropped faces
        cropped_faces = []
        
        # Check if faces were detected
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x_min = int(bboxC.xmin * iw)
                y_min = int(bboxC.ymin * ih)
                width = int(bboxC.width * iw)
                height = int(bboxC.height * ih)
                
                # Ensure the bounding box stays within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(iw, x_min + width)
                y_max = min(ih, y_min + height)
                
                # Crop the face
                cropped_face = img[y_min:y_max, x_min:x_max]
                cropped_faces.append(cropped_face)
    
        # Return the list of cropped faces
        return cropped_faces if cropped_faces else None
    except Exception as e:
        raise CustomException(e, sys)

# Function to check allowed file extensions
def allowed_file(filename):
    # Allowed extensions
    ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS