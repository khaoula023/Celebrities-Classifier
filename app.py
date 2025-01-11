from src.pipeline.pipeline_predict import PredictPipeline


image_path = "Test images\Image_237.jpg"

predict_pipeline = PredictPipeline()
result = predict_pipeline.predict(image_path)
print(result)

