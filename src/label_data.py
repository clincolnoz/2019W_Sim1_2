import datetime
import time
import os
import pandas as pd

raw_frame_path = './data/interim/'
raw_audio_path = 'data/audio_features/'
label_csvs_path = './data/label_csvs/'
kermit_path = './data/frames/kermit'
no_kermit_path = './data/frames/no_kermit'

if not os.path.exists(kermit_path):
    os.mkdir(kermit_path)
if not os.path.exists(no_kermit_path):
    os.mkdir(no_kermit_path)


def get_time_seconds(time_str):
    time_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], time_str.split(":")))
    return time_sec


kermit_time_files = os.listdir(label_csvs_path)

kermit_file_times = {}
for file in kermit_time_files:
    kermit_times = []
    with open((label_csvs_path + file), "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            start, stop = [get_time_seconds(time_str) for time_str in line.split(",")]
            kermit_times.extend(range(start, stop + 1))
    kermit_file_times[file.split(".")[0]] = kermit_times

# print(kermit_file_times)

image_files = os.listdir(raw_frame_path)
audio_label = []
for image_file in image_files:
    if not image_file in ["kermit", "no_kermit", ".gitkeep"]:
        filename, time = image_file.split("_")
        if int(time.split(".")[0]) in kermit_file_times[filename]:
            path_out = kermit_path
            audio_label.append([int(time.split('.')[0]),filename,'kermit'])
        else:
            path_out = no_kermit_path
            audio_label.append([int(time.split('.')[0]),filename,'no_kermit'])
        os.rename(os.path.join(raw_frame_path,image_file), os.path.join(path_out,image_file))

audio_labels_df = pd.DataFrame(audio_label, columns=['index','file','label'])

audio_labels_df.to_csv(os.path.join(raw_audio_path,'audio_features_labels.csv'),index=False)
audio_features = pd.read_csv(os.path.join(raw_audio_path,'audio_features.csv'))
audio_features_labelled=audio_features.set_index(['index','file']).join(audio_labels_df.set_index(['index','file']))
audio_features_labelled.dropna(axis=0,inplace=True)
audio_features_labelled.to_csv(os.path.join(raw_audio_path,'audio_features_labelled.csv'),index=True)

