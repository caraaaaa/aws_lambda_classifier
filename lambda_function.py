import tflite_runtime.interpreter as tflite
import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(url, targe_size=(150,150)):
    img = download_image(url)
    img = prepare_image(img, targe_size)
    x = np.array(img, dtype='float32')
    x /= 255.
    return np.array([x])

#interpreter = tflite.Interpreter(model_path="dino-vs-dragon-v2.tflite")
interpreter = tflite.Interpreter(model_path="dino_dragon.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ["dino", "dragon"]


def predict(url):
    X = preprocess_input(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()[0]

    return float_predictions


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    if result >= 0.5:
        return {"dragon": result}
    else:
        return {"dino": 1-result}