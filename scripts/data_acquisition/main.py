# IMPORTACIONES Y CONFIGURACIONES INICIALES

# Importar las bibliotecas
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense,
                                     Input, Layer, Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional)
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
import warnings

plt.rcParams['font.size'] = 12
sns.set_style("dark")
warnings.filterwarnings('ignore')

# Inicializo git y dvc (subo antes el archivo credentials.json)

!git init
!git config --global user.email "mcruzchu@gmail.com"
!git config --global user.name "Mariana Cruz"
!pip install dvc-gdrive
!apt install tree
!dvc --version
!dvc init
!dvc remote add -d storage gdrive://'1LQnbv6PF5GncWAufWu6Na_l9Aw0LKgBQ'
import json
with open("credentials.json") as f:
    os.environ["GDRIVE_CREDENTIALS_DATA"] = f.read()

# CARGA DE DATOS

# Instalar y cargar gdown para descargar los datos
!pip install gdown
import gdown

# Descargar y descomprimir el conjunto de datos
url = 'https://drive.google.com/uc?id=1-3-nVKJfLry8Fi793lV_Uzvl06Sy2Svb'
gdown.download(url, 'DataFlickr8KDataset.zip', quiet=False)
!unzip -oq DataFlickr8KDataset.zip
!ls
!ls DataFlickr8KDataset

# Definir las rutas de las imágenes y los datos
image_path = './DataFlickr8KDataset/Images'
data_path = './DataFlickr8KDataset/captions.txt'

# Leer los datos
data = pd.read_csv(data_path)

# Filtrar para obtener solo el 20% de las imágenes únicas
unique_images = data['image'].unique()
selected_images = unique_images[:int(len(unique_images) * 0.20)]
data = data[data['image'].isin(selected_images)]

# Definición de funciones
def readImage(path, img_size=224):
    img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.0
    return img

def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize = (20, 20))
    n = 0
    for i in range(min(15, len(temp_df))):
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace = 0.7, wspace = 0.3)
        image = readImage(image_path + "/" + temp_df.image[i])
        plt.imshow(image)
        plt.title("\n".join(wrap(temp_df.caption[i], 20)))
        plt.axis("off")

# Mostrar las imágenes seleccionadas
display_images(data.sample(min(15, len(data))))

!dvc add DataFlickr8KDataset.zip
!git add DataFlickr8KDataset.zip.dvc .gitignore
!git commit -m "Agregar archivo - DVC"
!dvc push