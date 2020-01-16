import subprocess
import os
import re
import datetime

root_dir = os.getcwd()#os.path.split(os.getcwd())[0]
input_dir = os.path.join(root_dir,'data/external')
if not os.path.exists(input_dir):
    print('Creating ' + input_dir)
    os.mkdir(input_dir)
output_dir = os.path.join(root_dir,'data/internal')
if not os.path.exists(output_dir):
    print('Creating ' + output_dir)
    os.mkdir(output_dir)

video_files = os.listdir(input_dir)
print(video_files)

for file in video_files:
    if re.search('.avi',file):
        video_input = os.path.join(input_dir,file)
        print('Processing file: ' + video_input)
        try:
            shell_output = subprocess.run('ffmpeg -i ' + video_input,capture_output=True,timeout=10)
        except subprocess.TimeoutExpired:
            print('Failed to get duration')
            continue
        for line in shell_output.stderr.decode("utf-8").splitlines():
            if re.search('Duration',line):
                h,m,s = [int(round(float(x))) for x in line.split(',',1)[0].split('Duration: ')[1].split(':')]
                duration_s = h*60*60+m*60+s
                print(duration_s)
        video_output_prefix = os.path.join(output_dir,file.split('.')[0])
        print(video_output_prefix)
        
        for t_s in range(duration_s+1):
            start = datetime.datetime.now()
            video_output = video_output_prefix + '_' + str(t_s) + '.jpg'
            if not os.path.exists(video_output):
                print('Writing to: ' + video_output)
                t_offset = str(datetime.timedelta(seconds=t_s))
                try:
                    shell_output = subprocess.run('ffmpeg -i ' +  video_input + ' -ss ' + t_offset + ' -t 00:00:1 -s 224x224 -r 1 -f singlejpeg ' + video_output,capture_output=True,timeout=60)
                except subprocess.TimeoutExpired:
                    print('Processed timed out... hopefully means filename exists')
                    continue
                # print(datetime.datetime.now()-start)
            
            


