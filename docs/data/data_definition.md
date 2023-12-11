# Definición de los datos

## Origen de los datos

El conjunto de datos es Flickr 8k Dataset, disponible en Kaggle

## Especificación de los scripts para la carga de datos


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

unique_images = data['image'].unique()

selected_images = unique_images[:int(len(unique_images) * 0.20)]

data = data[data['image'].isin(selected_images)]

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
        
display_images(data.sample(min(15, len(data))))



## Referencias a rutas o bases de datos origen y destino

https://www.kaggle.com/datasets/adityajn105/flickr8k

### Rutas de origen de datos

https://www.kaggle.com/datasets/adityajn105/flickr8k

Contiene imágenes en el folder Images y sus correspondientes descripciones en el archivo captions.txt

Al cargar los datos, se redimensiona la imagen a 224x224. Cada imagen se divide por 255 para su normalización.
