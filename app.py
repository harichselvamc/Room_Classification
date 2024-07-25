from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import pickle as pkl
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from io import BytesIO

app = FastAPI()

#### Constants
H, W = 224, 224
INPUT_SHAPE = (H, W, 3)
TARGET_SHAPE = (H, W)
NUM_CLASSES = 6

# Function to get class names
def get_class_names():
    with open('./class_list.pkl', 'rb') as f:
        class_names = pkl.load(f)
    return class_names

# Loading class names list
class_names = get_class_names()

# Function to get model ready
def get_model_ready():
    # Loading pre-trained EfficientNetB0
    efn = EfficientNetB0(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)

    # Making each layer not trainable
    for layer in efn.layers:
        layer.trainable = False

    # Input layer
    inputs = Input(shape=INPUT_SHAPE, name='input_shape')
    # Passing input layer to EfficientNetB0
    x = efn(inputs)
    # Global pooling layer
    x = GlobalAveragePooling2D(name='global_pooling')(x)
    # Dense layer
    x = Dense(1024, activation='relu', name='dense_1')(x)
    # Dropout layer
    x = Dropout(0.5, name='dropout_1')(x)
    # Dense layer
    x = Dense(64, activation='relu', name='dense_2')(x)
    # Dropout layer
    x = Dropout(0.5, name='dropout_2')(x)
    # Output layer
    outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    # Initializing model
    efficientNet_model = Model(inputs=inputs, outputs=outputs, name='efficientNet_based_model')

    return efficientNet_model

# Loading trained model
model = get_model_ready()

# Loading trained weights
model.load_weights('./efficientNetB0_model.h5')

# Function to predict class labels on a single image
def final_fun_2(image):
    # Expanding dimensions to form (1, height, width, channel) format
    image = np.expand_dims(image, axis=0)    
    # Classifying image
    yhats = model.predict(image)
    # Returning class names index
    return yhats

# Function to preprocess and predict a single image
def predict_single_image(image: Image.Image):
    img = image.resize(TARGET_SHAPE)
    img = np.array(img)
    img = preprocess_input(img)
    
    yhats = final_fun_2(img)
    pred = np.argmax(yhats, axis=1)[0]
    conf = np.round_(yhats[0][pred] * 100, 2)
    
    return class_names[pred], conf

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    prediction, confidence = predict_single_image(image)
    return {"prediction": prediction, "confidence": confidence}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
