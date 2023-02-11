from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uuid
import numpy as np
from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import time

from pydantic import BaseModel

class Model(BaseModel):
    name : str


showing = False

def load_model(model):

  model_chosen = model

  model_path = "Modelos/" + model_chosen + "/" + "Model.tflite"
  label_path = "Modelos/" + model_chosen + "/" + "Labels.txt"

  interpreter = Interpreter(model_path)

  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  return width, height


def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image
  floating_model = (input_details[0]['dtype'] == np.float32)
  input_mean = 127.5
  input_std = 127.5

def classify_image(interpreter, image, top_k=1):
  if floating_model:
    input_data = (np.float32(image)-input_mean)/input_std
  else:
    input_data = image

  set_input_tensor(interpreter, input_data)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  if floating_model:
    return 645,0.0987
  else:
    return [(i, output[i]) for i in ordered[:top_k]][0]


def procces_image(interpreter,image, label_path):

    # Classify the image.
    time1 = time.time()
    label_id, prob = classify_image(interpreter, image)
    time2 = time.time()
    classification_time = np.round(time2-time1, 3)
    accuracy = np.round(prob*100, 2)
    if showing:

      # Read class labels.
      labels = load_labels(label_path)

      # Return the classification label of the image.
      classification_label = str(labels[label_id])
    else:
    
    
      classification_label = "None"

    return classification_time, classification_label, accuracy 

def run_program(image_raw,interpreter,label_path,width,height):

  # Load an image to be classified.
  image = Image.open(image_raw).convert('RGB').resize((width, height))


  classification_time, classification_label, accuracy =  procces_image(interpreter,image, label_path)

  

  return classification_time, classification_label, accuracy

app = FastAPI()

model_chosen = "Mobilenet_v1"

model_path = "Modelos/" + model_chosen + "/" + "Model.tflite"
label_path = "Modelos/" + model_chosen + "/" + "Labels.txt"

interpreter = Interpreter(model_path)

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = (input_details[0]['dtype'] == np.float32)
showing = not(floating_model)
input_mean = 127.5
input_std = 127.5


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/Models")
def models():
    return {"1.": "mobilenet_v1"}

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    time1 = time.time()
    _, classification_label, accuracy = run_program(file.file,interpreter,label_path,width,height)
    time2 = time.time()
    classification_time = np.round(time2-time1, 3)

    return  {"Classification Time":  classification_time, "Classification Label": classification_label, "Accuracy": accuracy }

@app.post("/model")
async def change_model(model: Model):

  global model_chosen, model_path, label_path, interpreter, height, width, input_details, output_details, floating_model, showing
  model_chosen = model.name

  model_path = "Modelos/" + model_chosen + "/" + "Model.tflite"
  label_path = "Modelos/" + model_chosen + "/" + "Labels.txt"

  interpreter = Interpreter(model_path)

  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  floating_model = (input_details[0]['dtype'] == np.float32)
  showing = not(floating_model)


