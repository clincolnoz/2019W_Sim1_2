import av
import os
import re

data_path = './data/external/'
cache_path = './data/raw/'
fps = 25

input_dir = './data/external'
if not os.path.exists(input_dir):
    print('Creating ' + input_dir)
    os.mkdir(input_dir)
output_dir = './data/internal'
if not os.path.exists(output_dir):
    print('Creating ' + output_dir)
    os.mkdir(output_dir)

video_files = os.listdir(input_dir)
print(video_files)

for file in video_files:
    if re.search('.avi',file):
        video_input = os.path.join(input_dir,file)
        print('Processing file: ' + video_input)
        container = av.open(video_input)
        stream = container.streams.video[0]
        for frame_i,frame in enumerate(container.decode(stream)):
            if frame.index % fps == 0:
                frame.to_image().save(os.path.join(output_dir,file.split('.avi')[0] + ('-%04d.jpg' % (frame.index/fps))))
