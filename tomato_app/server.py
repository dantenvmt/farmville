from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()



dataload = tf.keras.models.load_model("../models")

class_name = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
@app.get('/ping')
@app.post('/predict')
def read_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

async def ping():
    return "Hello World"
async def predict(
    file:UploadFile = File(...)
):
    image = read_image(await file.read())
    image_batch=np.expand_dims(image,0)
    prediction = dataload.predict(image_batch)
    return
if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8000)
