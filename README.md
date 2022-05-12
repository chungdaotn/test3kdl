# MÔ HÌNH PHÂN LOẠI HOA
## YÊU CẦU 
- Python 3.7.13 trở lên
- Sử dụng hệ điều hành Window/MacOS/Linux

## CÀI ĐẶT
### Add thư viện
```
import os
import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
```
### Kết nối driver
```
from google.colab import drive
drive.mount('/content/drive')
```
### Tải model
```
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weight_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

```

### Transfer learning
```
pre_trained_model =  InceptionV3(include_top = False, weights=None, input_shape = (256,256,3))

pre_trained_model.load_weights(local_weight_file)

for layer in pre_trained_model.layers:
  layer.trainable = False 

#get final layer shape 
last_layer = pre_trained_model.get_layer('mixed7')
output_shape = last_layer.output
```
### Xây dựng model
```
from tensorflow.keras.optimizers import Adam, RMSprop

x = layers.Flatten()(output_shape)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512,activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128,activation='relu')(x)


x = layers.Dense(6,activation='softmax')(x) #cái dòng này k động vào

model = Model(pre_trained_model.input,x)
model.summary()
```
### DATA
Data là 1 file Train.rar. Sau khi giải nén thu được 6 folder, mỗi folder khoảng 700-900 ảnh về các loài hoa.
```
pip install rarfile
```

```
import zipfile
import os
path_rar = '/content/drive/MyDrive/Kho và khai phá'
for i in os.listdir(path_rar):
  path = path_rar +'/'+ i
  if path =='/content/drive/MyDrive/Kho và khai phá/archive.zip': 
    print(True)
    files = zipfile.ZipFile(path)
    files.extractall(path_rar)
    files.close()
```

### Đọc ghi file
Add thư viện
```
import cv2
import numpy as np
sample = cv2.imread('/content/drive/MyDrive/Kho và khai phá/Train/astilbe/1033455028_f0c6518ec9_c.jpg')
sample.shape
```
Đọc ghi 
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_type = ImageDataGenerator(
    rescale = 1./255.,
    rotation_range = 45,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip= True,
    validation_split = 0.2
    # vertical_flip = True
)


train_data = train_type.flow_from_directory(
    "/content/drive/MyDrive/Kho và khai phá/Train",
    target_size=(256,256),
    subset = "training",
    class_mode = 'categorical',
    batch_size=20
)

val_data = train_type.flow_from_directory(
    "/content/drive/MyDrive/Kho và khai phá/Train",
    target_size=(256,256),
    subset = "validation",
    class_mode = 'categorical',
    batch_size=20
    
)
```
Lọc nhãn dữ liệu
```
labels = os.listdir("/content/drive/MyDrive/Kho và khai phá/Train")
labels.sort()
labels
```
### Hàm Callback để dừng chương trình khi kết quả trên 95%
```
import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True
```

```
callbacks = myCallback()
history = model.fit(train_data, validation_data= val_data, steps_per_epoch = 100,
            epochs = 50,
            validation_steps = 50,
            verbose = 2, callbacks = [callbacks])
```

