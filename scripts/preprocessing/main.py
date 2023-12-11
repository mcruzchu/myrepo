# PREPROCESAMIENTO y CREA LOS CONJUNTOS DE ENTRENAMIENTO Y VALIDACION

def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]", ""))
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+", " "))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    data['caption'] = "startseq " + data['caption'] + " endseq"
    return data

# Procesar los textos (captions)
data = text_preprocessing(data)
captions = data['caption'].tolist()
print(captions[:10])

# Crear y ajustar el Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Preparar los datos de las imágenes
images = data['image'].unique().tolist()
nimages = len(images)

# Dividir los datos en conjuntos de entrenamiento y validación
split_index = round(0.85 * nimages)
train_images = images[:split_index]
val_images = images[split_index:]

# Crear los conjuntos de datos de entrenamiento y validación
train = data[data['image'].isin(train_images)]
test = data[data['image'].isin(val_images)]

train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)

# Ejemplo de cómo convertir texto a secuencias
print(tokenizer.texts_to_sequences([captions[1]])[0])

# Image Feature Extraction

# Crear el modelo de extracción de características
model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
fe = Model(inputs=model.input, outputs=model.layers[-2].output)

# Tamaño de las imágenes
img_size = 224

# Diccionario para almacenar las características
features = {}

# Procesar las imágenes y extraer características
# 'data' contiene solo el 20% de las imágenes
for image in tqdm(data['image'].unique().tolist()):
    img = load_img(os.path.join(image_path, image), target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    feature = fe.predict(img, verbose=0)
    features[image] = feature

# En este punto, 'features' contiene las características extraídas de las imágenes reducidas al 20%