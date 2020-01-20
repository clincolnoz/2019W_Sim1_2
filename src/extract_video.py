import av  # https://docs.mikeboers.com/pyav/develop/api/video.html#module-av.video.frame
import os
import re

input_dir = "./data/external"
output_dir = "./data/interim"
fps = 25

if not os.path.exists(input_dir):
    print("Creating " + input_dir)
    os.mkdir(input_dir)
if not os.path.exists(output_dir):
    print("Creating " + output_dir)
    os.mkdir(output_dir)

video_files = os.listdir(input_dir)
print(video_files)

for file in video_files:
    if re.search(".avi", file):
        video_input = os.path.join(input_dir, file)
        print("Processing file: " + video_input)
        container = av.open(video_input)
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if (frame.index % fps == 0) and not (frame.index == 0):
                # formater = av.video.reformatter.VideoReformatter(frame, width=224, height=224, interpolation='BILINEAR')
                frame = frame.reformat(width=224, height=224)
                frame.to_image().save(
                    os.path.join(
                        output_dir,
                        file.split(".avi")[0] + ("_%04d.jpg" % (frame.index / fps)),
                    )
                )
