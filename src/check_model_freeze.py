import os
import json
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

from tf_hub import TFHub
from image_net_util import *

filepath = './models/ResNet50V2_0.1/bestmodel\ResNet50V2_0.1_1.h5'
model = tf.keras.models.load_model(
            filepath, 
            compile=True,
            custom_objects={'KerasLayer':hub.KerasLayer},
        )
model.layers[0].trainable=True
model_yaml = model.to_yaml()
print(model_yaml)


print(str(model.layers[0].__dir__()))
print(str(model.layers[0].trainable))
print(str(model.layers[0].trainable_weights))
print(str(model.layers[0].non_trainable_weights))
# for i, layer in enumerate(model.layers):
#     if layer.name == 'keras_layer':
#         for j, keras_layers in layer:
#             print(j,keras_layers.name)

#     print(i, layer.name)