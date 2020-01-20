# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import yaml.dumper
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

from tf_hub import TFHub
from image_net_util import *

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

np.random.seed(1)

config_file = 'ResNet50V2_0.2.yaml'
# config_file = 'MobileNetV2_0.2.yaml'

# get best model path
best_model_file = get_best_model_file(config_file)
df = make_predictions(config_file,best_model_file)

print(confusion_matrix(df['true'], df['pred']))

fpr, tpr, _ = roc_curve(df['true'], df['probs_no_kermit'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,6))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of kermit classifier ({})'.format(config_file.split('.yaml')[0]))
plt.legend(loc="lower right")
plt.savefig('./reports/image_ROC.png')
plt.show()
