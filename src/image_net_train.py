# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

from tf_hub import TFHub
from image_net_util import *

np.random.seed(1)

config_file = 'ResNet50V2_0.2.yaml'
config_path = os.path.join(*[os.getcwd(), "models",config_file])
config = get_config(config_path)

# get image generators
train_generator, development_generator, test_generator = get_image_generator(config)
# setup new instance
im_cls, tf_hub_kwargs = create_new_model(config, train_generator, development_generator)

# First Training
# im_cls.load('models/ResNet50V2_0.2/bestmodel','ResNet50V2','0.2_1')
hist=im_cls.fit()
hist_df = pd.DataFrame.from_dict(hist,orient='columns')
hist_df.to_csv('{}{}/hist_{}.csv'.format(config['model_path'],config['model_name'],config['model_name']),
               index=False)

