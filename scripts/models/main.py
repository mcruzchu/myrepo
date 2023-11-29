# -*- coding: utf-8 -*-
"""Proyecto_Modulo6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UOvwtEE0NhjauBOWy9MGC-YASB2gXF-Y
"""

import pandas as pd

!pip install gdown
import gdown
url = 'https://drive.google.com/uc?id=1-3-nVKJfLry8Fi793lV_Uzvl06Sy2Svb'
gdown.download(url, 'DataFlickr8KDataset.zip', quiet=False)
!unzip -oq DataFlickr8KDataset.zip
!ls
!ls DataFlickr8KDataset
image_path = './DataFlickr8KDataset/Images'
data_path = './DataFlickr8KDataset/captions.txt'

data = pd.read_csv(data_path)
data.head()

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from textwrap import wrap

# Función para leer una imagen del sistema de archivos y procesarla
def readImage(path, img_size=224):
    img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.  # Normaliza los valores de los píxeles a [0, 1]
    return img

# Función para mostrar imágenes con sus subtítulos
def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)
        image = readImage(f"{image_path}/{temp_df['image'][i]}")
        plt.imshow(image)
        plt.title("\n".join(wrap(temp_df['caption'][i], 20)))
        plt.axis("off")

# Muestra las imágenes con sus subtítulos
display_images(data.sample(25).reset_index(drop=True))  # Toma una muestra aleatoria de 25 filas del DataFrame

# Preprocesamiento

import re

def text_preprocessing(data):
    data['caption'] = data['caption'].str.lower()
    data['caption'] = data['caption'].apply(lambda x: re.sub("[^a-z\s]+", "", x))
    data['caption'] = data['caption'].apply(lambda x: re.sub("\s+", " ", x))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    data['caption'] = "startseq " + data['caption'] + " endseq"
    return data

def text_preprocessing(data):
    data['caption'] = data['caption'].str.lower()
    data['caption'] = data['caption'].apply(lambda x: re.sub("[^a-z\s]+", "", x))
    data['caption'] = data['caption'].apply(lambda x: re.sub("\s+", " ", x))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    data['caption'] = "startseq " + data['caption'] + " endseq"
    return data

data = text_preprocessing(data)

captions = data['caption'].tolist()
captions[:10]

from tensorflow.keras.preprocessing.text import Tokenizer

# Crea el Tokenizer y ajusta en los subtítulos
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Extrae la lista de imágenes únicas y calcula la cantidad total
images = data['image'].unique().tolist()
nimages = len(images)

# Divide los datos en entrenamiento y validación (85% entrenamiento, 15% validación)
split_index = round(0.85 * nimages)
train_images = images[:split_index]
val_images = images[split_index:]

# Crea DataFrames para entrenamiento y validación
train = data[data['image'].isin(train_images)]
test = data[data['image'].isin(val_images)]

# Restablece los índices de los DataFrames de entrenamiento y validación
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)

# Ejemplo cómo transformar un texto a secuencia de tokens
sequence_example = tokenizer.texts_to_sequences([captions[10]])[0]
sequence_example

from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
from tqdm.notebook import tqdm
import os

# Inicializa el modelo DenseNet201 y crea un nuevo modelo que tome las entradas del modelo original
# pero que entregue como salida las características de la penúltima capa
model = DenseNet201(include_top=False, pooling='avg')
fe = Model(inputs=model.input, outputs=model.output)

img_size = 224
features = {}

# Procesa cada imagen y extrae las características
for image in tqdm(data['image'].unique().tolist()):
    # Carga la imagen y la procesa
    img_path = os.path.join(image_path, image)
    img = load_img(img_path, target_size=(img_size, img_size))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # DenseNet requiere preprocesamiento

    # Predice las características y las guarda en un diccionario
    feature = fe.predict(img, verbose=0)
    features[image] = feature

# Ahora 'features' contendrá un mapeo de los nombres de imagen a sus características extraídas