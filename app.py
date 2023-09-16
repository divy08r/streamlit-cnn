from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from PIL import Image
import io
import keras
from keras.models import load_model
import numpy as np 
import tensorflow as tf
from router import ToolsRoutes
 
app = FastAPI()

MODEL_PATH = 'network2.h5'

model = load_model(MODEL_PATH)
output_class = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic","shoes","trash","white-glass"]

def read_file_as_img(data)->np.ndarray:
    img = np.array(Image.open(io.BytesIO(data)))
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = read_file_as_img(await file.read())
    img = tf.image.resize(img, (256, 256))
    new_image = np.expand_dims(img, 0)

    p1, p2 = model_predict(new_image)
    return {
        "prediction": p1,
        "accuracy": p2
    }


def model_predict(new_image):
  predicted_array = model.predict(new_image)
  predicted_value = output_class[np.argmax(predicted_array)]
  predicted_accuracy = round(np.max(predicted_array) * 100, 2)

  print("Your waste material is ", predicted_value, " with ", predicted_accuracy, " % accuracy")
  return predicted_value, predicted_accuracy
 
@app.get("/")
async def read_random_file():
    return {"hello me"}

app.include_router(ToolsRoutes.router)
