# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import json

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

from get_config import get_config
from tf_hub import TFHub

config_file = 'ResNet50V2.yaml'
# config_path = os.path.join(os.getcwd(), "models/MobileNetV2.yaml")
config_path = os.path.join(*[os.getcwd(), "models",config_file])
config = get_config(config_path)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **config["data_preprocessing"]["base_datagen"],
    # **config["data_preprocessing"]["train_datagen"],
)

development_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **config["data_preprocessing"]["base_datagen"],
    # **config['data_preprocessing']['development_datagen'],
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **config["data_preprocessing"]["base_datagen"],
    # **config['data_preprocessing']['test_datagen'],
)

train_generator = train_datagen.flow_from_directory(
    config["data_preprocessing"]["train_data_dir"],
    **config["data_preprocessing"]["train_dataflow"],
    **config["data_preprocessing"]["base_dataflow"],
)

development_generator = development_datagen.flow_from_directory(
    config["data_preprocessing"]["train_data_dir"],
    **config["data_preprocessing"]["development_dataflow"],
    **config["data_preprocessing"]["base_dataflow"],
)

test_generator = test_datagen.flow_from_directory(
    config["data_preprocessing"]["test_data_dir"],
    **config["data_preprocessing"]["test_dataflow"],
    **config["data_preprocessing"]["base_dataflow"],
)

config["train_generator"] = train_generator
config["development_generator"] = development_generator
config["test_generator"] = test_generator


tf_hub_kwargs = config
del tf_hub_kwargs["data_preprocessing"]
im_cls = TFHub(tf_hub_kwargs)
im_cls.setup_model()
hist=im_cls.fit()
im_cls.save(name=config_file.split('.')[0], version="0.2")

print(hist)
# Get the dictionary containing each metric and the loss for each epoch
history_dict = hist.history
# Save it under the form of a json file
json.dump(history_dict, open('./models/' + config_file.split('.')[0] + 'hist.json', 'w'))

# test_steps = test_generator.samples // test_generator.batch_size
# evaluation=im_cls.evaluate(test_generator,test_steps)

# im_cls.save(name='im_cls', version='0.1')
# im_cls.load(name='im_cls', version='0.1')
# evaluation=im_cls.evaluate()

# print(evaluation)
