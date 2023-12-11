#ENTRENAMIENTO DEL MODELO

history = caption_model.fit(
    train_generator,
    epochs=10,  # número de épocas
    validation_data=validation_generator,
    callbacks=[checkpoint, earlystopping, learning_rate_reduction])

# Entrenamiento del Modelo Simplificado

# Configurar los callbacks para el modelo simplificado
model_name_simplificado = "modelo_simplificado.h5"
checkpoint_simplificado = ModelCheckpoint(model_name_simplificado,
                                          monitor="val_loss",
                                          mode="min",
                                          save_best_only=True,
                                          verbose=1)

earlystopping_simplificado = EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=5,
                                           verbose=1,
                                           restore_best_weights=True)

learning_rate_reduction_simplificado = ReduceLROnPlateau(monitor='val_loss',
                                                         patience=3,
                                                         verbose=1,
                                                         factor=0.2,
                                                         min_lr=0.00000001)

# Entrenar el modelo sencillo
history_simplificado = caption_model_simplificado.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint_simplificado, earlystopping_simplificado, learning_rate_reduction_simplificado])

# Graficar la pérdida del modelo
plt.figure(figsize=(20, 8))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Graficar la pérdida del modelo simplificado
plt.figure(figsize=(20, 8))
plt.plot(history_simplificado.history['loss'], label='train')
plt.plot(history_simplificado.history['val_loss'], label='validation')
plt.title('Model Loss (Simplificado)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()