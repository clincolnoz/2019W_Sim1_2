import os
import json
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

from audio_features_model import AudioFeatures

audio_features = pd.read_csv('data/audio_features/audio_features_labelled_test.csv')
print(audio_features)
X_train = audio_features.loc[audio_features.test==False].drop(['label','test','file','index'],axis=1).values
y_train = audio_features.loc[audio_features.test==False, 'label']
y_train_dummies = pd.get_dummies(y_train).values
print(y_train_dummies)


X_test = audio_features.loc[audio_features.test==True].drop(['label','test','file','index'],axis=1).values
y_test = audio_features.loc[audio_features.test==True, 'label']
y_test_dummies = pd.get_dummies(y_test).values

le = sklearn.preprocessing.LabelEncoder()
le.fit(y_test)
true=le.transform(y_test)
print([X_train.shape,y_train.shape,X_test.shape,y_test.shape])

n_labels=y_train.nunique()
print(n_labels)

audio_cls = AudioFeatures()
audio_cls.setup_model(n_features=X_train.shape[1],n_labels=n_labels)
hist=audio_cls.fit(X_train=X_train,y_train=y_train_dummies)
print(hist)
audio_cls.evaluate(X_test=X_test,y_test=y_test_dummies)
pred = audio_cls.predict(X_test=X_test)

df = pd.DataFrame(pred, columns = ['probs_kermit','probs_no_kermit'])
df['true'] = true
df['pred'] = df['probs_kermit'].apply(lambda x: 0 if x>=0.5 else 1)

df.to_csv('data/audio_predictions_labelled.csv',index=False)

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
plt.title('Receiver operating characteristic of kermit classifier (audio)')
plt.legend(loc="lower right")
plt.savefig('./reports/audio_ROC.png')
plt.show()