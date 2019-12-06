# Kermit Detection - Similiarty Modelling 1/2 Project

![Kermit Wanted Header](https://vignette.wikia.nocookie.net/muppet/images/0/05/Iflorist_1.jpg)

This repository contains code for dececting Kermit the frog from `The Muppets TV Show` using visual and accoustic features.

## Requirements

Have `ffmpeg` installed.

Have `imaegai` installed. [Further information.](https://imageai.readthedocs.io/en/latest/#installing-imageai)

## Getting started

Install the needed packaged via pip, by running:

```bash
pip install -r requirements.txt
```

To create the images from the video:
```python
python data/make_dataset.py "../data/raw/YOUR_VIDEO_HERE.mp4" "../data/interim/"
```

To classify images from the extracted ones:
```python
python data/classify_dataset.py "../data/interim/" "../data/processed/"
```

## Train-test splitting the data TODO(Consider dataset imbalance)

We use the split-folders pip package for train-test splitting. (Running from `/src`)

```bash
python data/traintest_split.py
```

Output:

```bash
├── data
│	└── images
│		├── train
│		│	├── kermit
│		│	└── no_kermit
│		└── test
│			├── kermit
│			└── no_kermit

```

## Training the model

Since ImageAI library was used, we have trained the CustomImagePrediction model (which uses Resnets as 
network model type).

To train the model, run the following:

```bash
python models/build_model.py
```

This may take different running times, depending on the number of training images, epochs, batch sizes etc.

After model is trained, the corresponding trained model is stored in `h5` format under 
 the `data/images/models/model_name.h5`

## Troubleshooting

### Path errors

All paths are relative **Windows** paths. Adjust according to your OS.

### Tried to convert 'y' to a tensor and failed. Error: None values not supported.

I had to change one of the imports of the `ModelTraining` in the `init.py` file. [Reference](https://github.com/tensorflow/tensorflow/issues/32646)

Replace:

```python
from tensorflow.python.keras.optimizers import Adam
```
With:

```python
from tensorflow.keras.optimizers import Adam
```

### KeyError: `val_acc`

Replace `val_acc` with `val_accuracy` in the `__init__.py` file of `imageai/Prediction/Custom`