import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from src.pipeline.pipeline_predict import PredictPipeline
from src.utils import allowed_file

app = Flask(__name__, static_folder="static")

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def classify_image(image_path):
    # load the model:
    predict_pipeline = PredictPipeline()
    # Make Predictions:
    result = predict_pipeline.predict(image_path)
    return result

@app.route('/')
def home():
    return render_template('website.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']

    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Call your classification function
            classification_result = classify_image(file_path)
            return classification_result[0]  
        except Exception as e:
            app.logger.error(f"Classification error: {e}")
            return "Classification failed", 500

    return "File not allowed", 400


if __name__ =="__main__":
    app.run(debug=True)