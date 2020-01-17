import datetime
import time
import os

raw_frame_path = './data/interim/'
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
    with open((label_csvs_path + file), 'r') as f:
        for i,line in enumerate(f):
            if i == 0:
                continue
            start, stop = [get_time_seconds(time_str) for time_str in line.split(',')]
            kermit_times.extend(range(start,stop+1))
    kermit_file_times[file.split('.')[0]] = kermit_times

# print(kermit_file_times)

image_files = os.listdir(raw_frame_path)
for image_file in image_files:
    if not image_file in ['kermit','no_kermit','.gitkeep']:
        filename, time = image_file.split('_')
        if int(time.split('.')[0]) in kermit_file_times[filename]:
            path_out = kermit_path
        else:
            path_out = no_kermit_path
        os.rename(os.path.join(raw_frame_path,image_file), os.path.join(path_out,image_file))
