# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import yaml
import tensorflow as tf

from get_config import get_config
from tf_hub import TFHub

np.random.seed(1)

def get_config(path_to_yaml):
    """read yaml configuration file
    safe_load disallows objects but not sure what happens if it finds one
    """
    if not os.path.exists(path_to_yaml):
        print("file not found")
        return -1
    else:
        print("loading yaml from " + path_to_yaml)
        with open(path_to_yaml, "r") as yaml_file:
            d = yaml.load(yaml_file, Loader=yaml.FullLoader)
            # print(d)
        return d

def get_image_generator(config):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **config["data_preprocessing"]["train_base_datagen"],
        **config["data_preprocessing"]["train_datagen"],
    )

    development_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **config["data_preprocessing"]["train_base_datagen"],
        **config['data_preprocessing']['development_datagen'],
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **config['data_preprocessing']['test_datagen'],
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
    return train_generator, development_generator, test_generator

def create_new_model(config, train_generator, development_generator):
    config["train_generator"] = train_generator
    config["development_generator"] = development_generator
    tf_hub_kwargs = config
    del tf_hub_kwargs["data_preprocessing"]
    im_cls = TFHub(tf_hub_kwargs)
    im_cls.setup_model()
    return im_cls, tf_hub_kwargs

def make_predictions(config,best_model_path):
    # get image generators
    train_generator, development_generator, test_generator = get_image_generator(config)
    # setup new instance
    im_cls2, tf_hub_kwargs = create_new_model(config, train_generator, development_generator)

    # load best model
    # best_model_path = 'models/ResNet50V2_0.2/bestmodel/ResNet50V2_0.2_1.h5'
    im_cls2.load(best_model_path)

    steps = test_generator.samples // test_generator.batch_size
    # evaluation=im_cls2.evaluate(test_generator,steps)
    pred = im_cls2.predict(test_generator,steps)
    # print(pred)

    df = pd.DataFrame(pred, columns = ['probs_kermit','probs_no_kermit'])
    df['true'] = test_generator.labels
    df['pred'] = df['probs_kermit'].apply(lambda x: 0 if x>=0.5 else 1)

    df.to_csv('data/image_predictions_labelled.csv',index=False)
    return df
    