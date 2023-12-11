# MODELLING

# Definir la arquitectura del modelo
# Ajusta la forma de entrada de acuerdo con la forma de tus características de imagen 'X1'
input1 = Input(shape=(7, 7, 1920))
input2 = Input(shape=(max_length,))  # 'max_length' debe ser definido en tu entorno

# Procesar características de imagen
img_features = GlobalAveragePooling2D()(input1)  # Reduce la dimensión de las características de la imagen
img_features = Dense(256, activation='relu')(img_features)

# Procesar características del texto
sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
sentence_features = LSTM(256)(sentence_features)
sentence_features = Dropout(0.5)(sentence_features)

# Combinar las características de imagen y texto
merged_features = add([sentence_features, img_features])
merged_features = Dense(128, activation='relu')(merged_features)
merged_features = Dropout(0.5)(merged_features)
output = Dense(vocab_size, activation='softmax')(merged_features)  # 'vocab_size' debe ser definido en tu entorno

# Crear el modelo
caption_model = Model(inputs=[input1, input2], outputs=output)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Ahora tu modelo 'caption_model' está listo para ser entrenado con los datos.

# DATA GENERATION

class CustomDataGenerator(Sequence):

    def __init__(self, df, X_col, y_col, batch_size, directory, tokenizer,
                 vocab_size, max_length, features, shuffle=True):

        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.directory = directory
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        self.shuffle = shuffle
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, index):

        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size, :]
        X1, X2, y = self.__get_data(batch)
        return (X1, X2), y

    def __get_data(self, batch):

        X1, X2, y = list(), list(), list()

        images = batch[self.X_col].tolist()

        for image in images:
            feature = self.features[image][0]

            captions = batch.loc[batch[self.X_col] == image, self.y_col].tolist()
            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)

        X1, X2, y = np.array(X1), np.array(X2), np.array(y)

        return X1, X2, y

# MODIFICACION DEL MODELO

# Visualización del modelo
plot_model(caption_model, to_file='caption_model.png')  # Guarda la visualización del modelo
caption_model.summary()  # Muestra un resumen del modelo

# Crear generadores de datos
train_generator = CustomDataGenerator(df=train, X_col='image', y_col='caption', batch_size=64, directory=image_path,
                                      tokenizer=tokenizer, vocab_size=vocab_size, max_length=max_length, features=features)

validation_generator = CustomDataGenerator(df=test, X_col='image', y_col='caption', batch_size=64, directory=image_path,
                                           tokenizer=tokenizer, vocab_size=vocab_size, max_length=max_length, features=features)

# Configurar los callbacks
model_name = "model.h5"
checkpoint = ModelCheckpoint(model_name,
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=3,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.00000001)

# En este punto, tu modelo 'caption_model' y los generadores de datos están listos para ser utilizados en el entrenamiento.

# Modelo 2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, add, GlobalAveragePooling2D

# Definir la arquitectura del modelo simplificado
input1_simplificado = Input(shape=(7, 7, 1920))
input2_simplificado = Input(shape=(max_length,))

# Procesar características de imagen (más simplificado)
img_features_simplificado = GlobalAveragePooling2D()(input1_simplificado)
img_features_simplificado = Dense(128, activation='relu')(img_features_simplificado)

# Procesar características del texto (más simplificado)
sentence_features_simplificado = Embedding(vocab_size, 128, mask_zero=True)(input2_simplificado)
sentence_features_simplificado = LSTM(128)(sentence_features_simplificado)
sentence_features_simplificado = Dropout(0.5)(sentence_features_simplificado)

# Combinar las características de imagen y texto
merged_features_simplificado = add([sentence_features_simplificado, img_features_simplificado])
merged_features_simplificado = Dense(64, activation='relu')(merged_features_simplificado)
output_simplificado = Dense(vocab_size, activation='softmax')(merged_features_simplificado)

# Crear el modelo simplificado
caption_model_simplificado = Model(inputs=[input1_simplificado, input2_simplificado], outputs=output_simplificado)
caption_model_simplificado.compile(loss='categorical_crossentropy', optimizer='adam')

# Resumen del modelo simplificado
caption_model_simplificado.summary()