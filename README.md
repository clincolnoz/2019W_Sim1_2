# Kermit Detection - Similiarty Modelling 1/2 Project

![Kermit Wanted Header](https://vignette.wikia.nocookie.net/muppet/images/0/05/Iflorist_1.jpg)

This repository contains code for dececting Kermit the frog from `The Muppets TV Show` using visual and accoustic features.

## Requirements

Have `ffmpeg` installed.

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

## Research

### Similar/Interesting Projects

* https://github.com/ilirosmanaj/detect_kermit