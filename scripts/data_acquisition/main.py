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