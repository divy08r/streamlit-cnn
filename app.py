import streamlit as st

from tensorflow.keras.models import load_model
import numpy as np
import keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps


st.title("Python or Anaconda Predictor")
st.header("Large Serpent Classifier")
st.text("Upload an Image for of either serpent for  image classification as anaconda or python")
     
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
uploaded_file2 = st.camera_input('Click image')

output_class = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic","shoes","trash","white-glass"]

model = load_model('network2.h5')

def model_predict(new_image):
#   t_img = image.load_img(new_image, target_size = (256,256))
  t_img = ImageOps.fit(new_image, (256, 256), Image.ANTIALIAS)

  t_img = image.img_to_array(t_img)/255
  t_img = np.expand_dims(t_img, axis=0)

  predicted_array = model.predict(t_img)
  predicted_value = output_class[np.argmax(predicted_array)]
  predicted_accuracy = round(np.max(predicted_array) * 100, 2)

  print("Your waste material is ", predicted_value, " with ", predicted_accuracy, " % accuracy")
  return predicted_value

if uploaded_file is not None or uploaded_file2 is not None:
    if uploaded_file2 is not None:
      image1 = Image.open(uploaded_file2)
      st.image(image1, caption='Uploaded Image.', use_column_width=True)
      st.write("Classifying...")

      predicted_value = model_predict(image1)
      st.write("It predicted " + predicted_value)
      uploaded_file2=None
    else:
      image1 = Image.open(uploaded_file)
      st.image(image1, caption='Uploaded Image.', use_column_width=True)
      st.write("Classifying...")

      predicted_value = model_predict(image1)
      st.write("It predicted " + predicted_value)
      uploaded_file=None
