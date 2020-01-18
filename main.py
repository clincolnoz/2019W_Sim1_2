#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.extractAudioFeatures import *

# Extract Audio Features and save to csv
files = ['data/external/Muppets-02-01-01.avi','data/external/Muppets-02-04-04.avi','data/external/Muppets-03-04-03.avi']
save_audio_features_to_csv(files)