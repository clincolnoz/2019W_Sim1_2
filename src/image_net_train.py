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
config_file = 'ResNet50V2_0.1.yaml'
best_model_file = setup_model_and_train(config_file)

config_file = 'ResNet50V2_0.2.yaml'
best_model_file = load_model_and_train(config_file,best_model_file)
