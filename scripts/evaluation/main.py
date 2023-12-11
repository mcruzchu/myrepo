#Caption Generation

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break

        in_text += " " + word
        if word == 'endseq':
            break

    return in_text

# Uso de la función predict_caption (asegúrate de que 'model', 'tokenizer', 'max_length' y 'features' estén definidos)
# Ejemplo: predicted_caption = predict_caption(model, 'image_name.jpg', tokenizer, max_length, features)

#Muestreo aleatorio para image captioning

# Aplicar el código a ambos modelos

samples = test.sample(15)
samples.reset_index(drop=True, inplace=True)

for index, record in samples.iterrows():
    img = load_img(os.path.join(image_path, record['image']), target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.

    # Generar subtítulo con el primer modelo
    caption = predict_caption(caption_model, record['image'], tokenizer, max_length, features)
    samples.loc[index, 'caption_model'] = caption

    # Generar subtítulo con el modelo simplificado
    caption_simplificado = predict_caption(caption_model_simplificado, record['image'], tokenizer, max_length, features)
    samples.loc[index, 'caption_model_simplificado'] = caption_simplificado

# Función para mostrar una sola imagen por fila con subtítulos
def display_images_with_captions(df):
    for i in range(len(df)):
        plt.figure(figsize=(5, 4))  # tamaño de la figura

        # Mostrar imagen
        img = load_img(os.path.join(image_path, df.iloc[i]['image']), target_size=(224, 224))
        plt.imshow(img)
        plt.axis('off')

        # Mostrar subtítulos
        plt.title(f"Original: {df.iloc[i]['caption']}\nModelo 1: {df.iloc[i]['caption_model']}\nModelo 2: {df.iloc[i]['caption_model_simplificado']}")
        plt.show()

# Mostrar las imágenes con subtítulos
display_images_with_captions(samples)

#Guarda el modelo en mi Google Drive

from google.colab import drive
drive.mount('/content/drive')

# Guarda el modelo en una carpeta específica de Google Drive
model.save('/content/drive/My Drive/ColabData/ProyectoModulo6/mi_modelo_entrenado.h5')

!pip install nltk
import nltk
nltk.download('punkt')

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize

# Función para calcular el BLEU score
def calculate_bleu(data, model, tokenizer, max_length, features):
    actual, predicted = [], []
    for idx in range(len(data)):
        reference = data.iloc[idx]['caption']
        image_id = data.iloc[idx]['image']

        # Generar subtítulo
        yhat = predict_caption(model, image_id, tokenizer, max_length, features)

        # Tokenizar las frases
        reference_tokens = word_tokenize(reference.lower())
        yhat_tokens = word_tokenize(yhat.split('startseq ')[1].split(' endseq')[0].lower())

        actual.append([reference_tokens])
        predicted.append(yhat_tokens)

    # Calcular BLEU score
    score = corpus_bleu(actual, predicted)
    return score

# Calcular BLEU para el primer modelo
bleu_score_model_1 = calculate_bleu(test, caption_model, tokenizer, max_length, features)

# Calcular BLEU para el modelo simplificado
bleu_score_model_2 = calculate_bleu(test, caption_model_simplificado, tokenizer, max_length, features)

print(f"BLEU score para el primer modelo: {bleu_score_model_1}")
print(f"BLEU score para el modelo simplificado: {bleu_score_model_2}")

#Calcular METEOR Score
import nltk
nltk.download('wordnet')

def calculate_meteor(data, model, tokenizer, max_length, features):
    scores = []
    for idx in range(len(data)):
        reference = data.iloc[idx]['caption']
        image_id = data.iloc[idx]['image']
        yhat = predict_caption(model, image_id, tokenizer, max_length, features)

        # Tokenizar las frases
        reference_tokens = word_tokenize(reference.lower())
        yhat_tokens = word_tokenize(yhat.split('startseq ')[1].split(' endseq')[0].lower())

        # Calcular METEOR score
        score = meteor_score([reference_tokens], yhat_tokens)
        scores.append(score)

    # Calcular el promedio de los scores
    return sum(scores) / len(scores)

# Calcular METEOR para ambos modelos
meteor_score_model_1 = calculate_meteor(test, caption_model, tokenizer, max_length, features)
meteor_score_model_2 = calculate_meteor(test, caption_model_simplificado, tokenizer, max_length, features)

print(f"METEOR score para el primer modelo: {meteor_score_model_1}")
print(f"METEOR score para el modelo simplificado: {meteor_score_model_2}")

#Calcular el ROUGE Score

!pip install rouge

from rouge import Rouge

def calculate_rouge(data, model, tokenizer, max_length, features):
    rouge = Rouge()
    actual, predicted = [], []

    for idx in range(len(data)):
        reference = data.iloc[idx]['caption']
        image_id = data.iloc[idx]['image']
        yhat = predict_caption(model, image_id, tokenizer, max_length, features)

        # Extraer subtítulos generados sin los tokens 'startseq' y 'endseq'
        yhat_clean = yhat.split('startseq ')[1].split(' endseq')[0]

        actual.append(reference)
        predicted.append(yhat_clean)

    # Calcular ROUGE score
    scores = rouge.get_scores(predicted, actual, avg=True)
    return scores

# Calcular ROUGE para ambos modelos
rouge_score_model_1 = calculate_rouge(test, caption_model, tokenizer, max_length, features)
rouge_score_model_2 = calculate_rouge(test, caption_model_simplificado, tokenizer, max_length, features)

print("ROUGE score para el primer modelo:", rouge_score_model_1)
print("ROUGE score para el modelo simplificado:", rouge_score_model_2)