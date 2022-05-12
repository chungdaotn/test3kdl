# DỰ Án
dkjajfd dfjakjfk dsa fjkdsa

## Hình ảnh

[dog](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.baamboozle.com%2Fstudy%2F148572&psig=AOvVaw3aMzqKDuGB5Kxg_uYQstDt&ust=1652416928740000&source=images&cd=vfe&ved=0CAwQjRxqFwoTCLCx8fKS2fcCFQAAAAAdAAAAABAD)

### code


```!wget --no-check-certificate \https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \-O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weight_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model =  InceptionV3(include_top = False, weights=None, input_shape = (256,256,3))

pre_trained_model.load_weights(local_weight_file)

for layer in pre_trained_model.layers:
  layer.trainable = False 

#get final layer shape 
last_layer = pre_trained_model.get_layer('mixed7')
output_shape = last_layer.output```

#### list
- dsfsd
- dsfd
- fdsf
    
