# ANÁLISIS EXPLORATORIO DE DATOS

from wordcloud import WordCloud, STOPWORDS

# Análisis de Texto

# Longitud de las descripciones
data['caption_length'] = data['caption'].apply(lambda x: len(x.split()))
plt.figure(figsize=(12, 6))
sns.histplot(data['caption_length'], kde=True)
plt.title('Distribución de la Longitud de las Descripciones')
plt.xlabel('Longitud de la Descripción')
plt.ylabel('Frecuencia')
plt.show()

# Palabras más comunes en las descripciones
from collections import Counter
all_words = ' '.join(data['caption']).split()
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(50)
plt.figure(figsize=(15, 8))
sns.barplot(x=[word[0] for word in most_common_words], y=[word[1] for word in most_common_words])
plt.xticks(rotation=45)
plt.title('50 Palabras más Comunes en las Descripciones')
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.show()

# Análisis de frecuencia de palabras específicas
wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      stopwords = set(STOPWORDS),
                      min_font_size = 10).generate(' '.join(data['caption']))
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# Análisis de Imágenes

# Función para calcular la luminosidad media de una imagen
def get_image_brightness(img):
    # Convertir a escala de grises para simplificar el cálculo
    gray_img = tf.image.rgb_to_grayscale(img)
    brightness = np.mean(gray_img)
    return brightness

# Calcular y graficar la luminosidad media de un conjunto de imágenes
brightness_values = []
for img_name in data['image'].unique()[:50]:
    img = readImage(image_path + '/' + img_name)
    brightness = get_image_brightness(img)
    brightness_values.append(brightness)

plt.figure(figsize=(12, 6))
sns.histplot(brightness_values, kde=True)
plt.title('Distribución de la Luminosidad de las Imágenes')
plt.xlabel('Luminosidad Media')
plt.ylabel('Frecuencia')
plt.show()