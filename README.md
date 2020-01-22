# Kermit Detection - Similiarty Modelling 1/2 Project

![Kermit Wanted Header](https://vignette.wikia.nocookie.net/muppet/images/0/05/Iflorist_1.jpg)

This repository contains code for dececting Kermit the frog from 
`The Muppets TV Show` using visual and accoustic features made as part
of the `188.498 Similarity Modeling 2 - Computational Seeing and Hearing`
lecture at the Technical University of Vienna.

## Students

Craig Lincoln (11828331) & Gent Rexha (11832486)

## Getting started

Install the needed packaged via pip, by running:

```bash
pip install -r requirements.txt
```
To have everything running execute the commands below inside of the root of the project:

```bash
python src\\extract_video.py
python src\\extractAudioFeatures.py
python src\\label_data.py
python src\\train_test_split.py
python src\\image_net_train.py
python src\\audio_net_train.py
python src\\combined_model.py
```

## Timesheet

| Date       	| Time          	| Description                                                        	| Person responsible 	|
|------------	|-----------------	|--------------------------------------------------------------------	|--------------------	|
| 08/10/2019 	| 09:00-12:00     	| Attended Lecture on SIM 1                                           	| Craig & Gent         	|
| 11/10/2019 	| 09:00-12:00     	| Attended Lecture on SIM 1                                          	| Craig & Gent         	|
| 19/10/2019 	| 16:00-19:00     	| Initialized repository and initial research.                       	| Craig              	|
| 23/10/2019 	| 14:00-15:00     	| Watched and got Kermit times from episode 02-01-01.                	| Craig              	|
| 04/11/2019 	| 20:00-22:00     	| Set project structure and moved some stuff around for the meeting. 	| Gent               	|
| 07/11/2019 	| 20:00-24:00     	| Started working on some dataset preparation.                       	| Gent               	|
| 08/11/2019 	| 10:00-11:00     	| Sync Meeting.                                                      	| Craig & Gent       	|
| 17/11/2019 	| 14:00-18:00     	| Look into Librosa and audio features.                              	| Craig              	|
| 18/11/2019 	| 14:00-17:00     	| Finished labeling GUI application.                                 	| Gent               	|
| 24/11/2019	| 12:00-17:00		| First look at implementing the ImageAI Neural Network.				| Gent					|
| 06/12/2019    | 12:00-15:00       | Fixed the model.                                                      | Gent                  |
| 16/01/2020    | 08:00-22:00       | Developed image extraction/labelling and implemented tensorflow hub   | Craig                 |
| 17/01/2020    | 15:00-22:00       | Further developed tensorflow hub                                      | Craig                 |
| 18/01/2020    | 15:00-16:00       | Alignment meeting                                                     | Craig & Gent          |
| 18/01/2020    | 16:00-18:00       | Finalized audio feature extraction and labelling                      | Craig                 |
| 18/01/2020    | 20:30-22:30       | Created test set by hand and made sure audio and video were consistent| Craig                 |
| 18/01/2020    | 22:30-23:30       | Train model                                                           | Craig                 |
| 19/01/2020    | 22:30-23:30       | Train model                                                           | Craig                 |
| 19/01/2020    | 20:00-22:00       | GPU Tensorflow environment setup                                      | Gent                  |
| 20/01/2020    | 11:00-24:00       | Model re-training, last changes and write report.                     | Gent                  |
| 20/01/2020    | 09:00-24:00       | Model training, messing with audio clf, checking report               | Craig                 |

## Computer Vision

### Extraction

Frames were extracted for every second using [PyAV](https://github.com/mikeboers/PyAV), which is a wrapper for ffmpeg and seems to be the simpler to use and install than opencv (see extract_video.py). Still there were conflicts on install with librosa as they had different version ffmpeg by default. I hope somebody fixes that one day.
Librosa was used to extract and process the audio chucks per second. Essentially, the video was loaded
and the hop_length variable of mfcc and chroma_stft was set to the sampling rate. In both cases, n_fft was set to 1024, which corresponds to approximately 46 ms window (see extractAudioFeatures.py). This resulted in 32 features, 20 mfcc_deltas’s and 12 chroma_stft
spectra.

### Train-test splitting the data

The training and test split was done by hand to ensure that whole segments (scenes) where he images were very similar were not shared across both splits. Moreover, to challenge the models, the scenes were kermit is with maid Marion and many of his jolly men (Muppets-03-04-03 1360-1386 seconds) and four no kermit scenes of squiggly arm guy (Muppets-02-01-01 277-367 seconds), the dancing scene were Kermit is also present but the frames were the other green guy is there (Muppets-02-04-02 746-777 seconds), Kermit’s cousin with the monster (Muppets-02-04-04 963-1021 seconds) and last and probably the worst Robin’s men again but not with Kermit (Mupets-03-04-03 124-1461 seconds). The hats and kermit style neck thing really made it challenging and while not shown, the image classifier struggled most with these images.

In addition, there was a significant class imbalance of around 2:1 in training and 3:1 in test. The reason for the small test set and its class imbalance is the relatively small dataset size for NN’s and leaving as many samples in training as possible. The smallness is made worse by the significant repetition of very similar images, effectively reducing the ‘real’ size of unique samples.

### Training the model(s)

All networks were developed using tensorflow with Keras. Image classification was made using [ResNet50](https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4) feature vector from tensorflow hub. The weights were frozen and a dense layer was added with 2 outputs and softmax. Optimization was done using Adam with default learning rate and loss was BinaryCrossentropy. 

The training images were split into train and validation(development) datasets which enabled best model saving per epoch. The initial fit was performed using 20 epochs on the images. The best model from this fit was re-loaded and further fit for 10 epochs with augmented images and best model saving implemented. 

Augmentation was randomized and included rotation, horizontal flip, width_shift, height_shift,
shear, zoom and brightness.

The audio features were fitted using a simple network of four dense layers of 40 units and relu activation and a dense output layer with two outputs. Again Adam and BinaryCrossentropy was used. The fit was run for 50 epochs with no binning.

### Model Evaluation

Below we’ve evaluated the ROC curves of all our three models.

This is by far the best performing model. This is somewhat to be expected considering the groundtruth is very stable and there’s a lot of data.

![Image ROC Curve](https://github.com/clincolnoz/2019W_Sim1_2/raw/master/reports/image_ROC.png "Image ROC Curve")

Comparing the two ROC curves of the image and audio models below we can clearly see that the image model outperforms the audio one. This we can set on the fact the image data provides a lot better base for classification than the audio data, due to the fact that in an image it is clear if kermit can be seen or not, but in audio data he could be talking while music is playing or someone else is talking which creates a lot of noise.

![Audio ROC Curve](https://github.com/clincolnoz/2019W_Sim1_2/raw/master/reports/audio_ROC.png "Audio ROC Curve")

A simple convolutional network trained in both of the features extracted, achieves a validation accuracy of ~90% over a single epoch, so training on that training set was not very representative of actual performance. Which can be seen when comparing performance metrics with the actual combined confusion matrix below.

Because of this we wanted to compare how the model compares to more traditional approaches, if we use the same combined data as input. After running a KNN, SVC and RandomForest classifier through a gridsearch we found that the RandomForest outperformed all other models.

![Combined ROC Curve](https://github.com/clincolnoz/2019W_Sim1_2/raw/master/reports/image_ROC_combined.png "Combined ROC Curve")

#### Classification Report

| Type of Model | Accuracy | Precision | Recall | Loss   |
|---------------|----------|-----------|--------|--------|
| Audio         | 0.6695   | 0.6712    | 0.6973 | 1.1384 |
| Image         | 0.7947   | 0.7380    | 0.7348 | 0.0322 |
| Combined      | 0.9489   | 0.9489    | 1.0    | 0.5178 |

#### Confusion Matrices

Audio Confusion Matrix

|           | Predicted kermit | Predicted no-kermit |
|-----------|------------------|---------------------|
| kermit    | 20               | 60                  |
| no-kermit | 58               | 198                 |


Image Confusion Matrix

|           | Predicted kermit | Predicted no-kermit |
|-----------|------------------|---------------------|
| kermit    | 71               | 9                   |
| no-kermit | 102              | 154                 |

Combined Confusion Matrix

|           | Predicted kermit | Predicted no-kermit |
|-----------|------------------|---------------------|
| kermit    | 0                | 43                  |
| no-kermit | 0                | 798                 |


### Discussion

The biggest problem that our project faced was probably the sheer size of the data and the missing of effective GPU’s on our machines. It severely limited what we could actually produce, and slowed down the training and testing massively. Given more time, it might have been possible to mitigate this problem by uploading our code to Google Colab, in reality we had a lot of ideas that would have gained from this method effectively.

Distribution of classes in the dataset was very unequal, where one class was almost twice as much in dataset as the other, which is not very good, therefore in hindsight balancing the dataset artificially might’ve been a good idea.

Ultimately, this was a very interesting challenge, with the biggest problem being data volumes, the time needed to do something effective with the data and building your own simple neural network. We've learned how to train a model for such large amounts of data effectively, how to correctly implement feature engineering and combining multiple features into a supervised classification problem.
