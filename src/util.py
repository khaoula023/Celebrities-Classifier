import json
import numpy as np
import base64
import cv2
from tensorflow.keras.models import load_model



# Global variables for class mappings and the model
__class_name_to_number = {}
__class_number_to_name = {}
__model = None

# Classify function
def classify(b64img, path=None):
    global __model
    print('detect face:')
    imgs = detect_face(path, b64img)
    print(imgs)
    result = []
    for img in imgs:
        # Resize the image to the model's input size
        scaled_img = cv2.resize(img, (32, 32))
        scaled_img = scaled_img / 255.0  # Normalize the image
        scaled_img = np.expand_dims(scaled_img, axis=0)  # Add batch dimension
        
        # load the model:
        if __model is None:
            print('loading model:')
            __model = load_model("./artifacts/model_path")
            print(__model.summary())    
        # Predict the class and probabilities
        predictions = __model.predict(scaled_img)
        predicted_class = np.argmax(predictions, axis=1)
        result.append({
            'class': class_number_to_name(predicted_class),
            'class_dictionary': __class_name_to_number
        })
    return result

# Convert class number to name
def class_number_to_name(class_num):
    return __class_number_to_name.get(class_num, "Unknown")

# Load saved artifacts
def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    # Load label mappings
    with open("./artifacts/Labels.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}
    
    print("Loading saved artifacts...done")

# Convert an image to Base64
def img_to_b64():
    with open("./base64/han seo yen.txt", "r") as f:
        return f.read()

# Convert Base64 string to image
def b64_to_img(data):
    print('convert b64 to image:')
    # Extract the Base64 data
    encoded_data = data.split(',')[-1]
    # Decode the Base64 string
    decoded_data = base64.b64decode(encoded_data)
    # Convert binary data to a numpy array
    nparr = np.frombuffer(decoded_data, np.uint8)
    # Decode the numpy array into an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img.shape)
    return img

# Detect faces in an image
def detect_face(image_path=None, image_base64_data=None):
    print('face recognittion :')
    cropped_faces = []
    face_cascade = cv2.CascadeClassifier("C:\\Users\\pc lenovo\\anaconda3\\Lib\\site-packages\\opencv_python-4.10.0.84.dist-info\\haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Read the image
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = b64_to_img(image_base64_data)

    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

# Main execution
if __name__ == '__main__':
    load_saved_artifacts()
    base64_data = img_to_b64()
    print(len(base64_data))
    print('classify the image:')
    results = classify(base64_data, None)
    print("Classification Results:", results)
