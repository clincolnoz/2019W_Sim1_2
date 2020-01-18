import os
import pandas as pd

test_file_csv_path = './data/testdata_csvs/'
test_file_csv_files = os.listdir(test_file_csv_path)

kermit_train_path = './data/frames/kermit'
no_kermit_train_path = './data/frames/no_kermit'

kermit_test_path = './data/frames/test/kermit'
no_kermit_test_path = './data/frames/test/no_kermit'

raw_audio_path = 'data/audio_features/'

if not os.path.exists(kermit_test_path):
    os.mkdir(os.path.split(kermit_test_path)[0])
    os.mkdir(kermit_test_path)
if not os.path.exists(no_kermit_test_path):
    os.mkdir(no_kermit_test_path)

test_list = []
def move_files(pathin,pathout,file_list,test_list):
    for file_name in file_list:
        file_stub, t_sec = file_name.split('_')
        t_sec = int(t_sec.split('.jpg')[0])
        test_list.append([t_sec,file_stub,True])
        os.rename(os.path.join(pathin,file_name), os.path.join(pathout,file_name))
    return test_list
    
csv_file = test_file_csv_files[0]
file_list = [line.rstrip('\n') for line in open(os.path.join(test_file_csv_path,csv_file))]
move_files(kermit_train_path,kermit_test_path,file_list,test_list)

csv_file = test_file_csv_files[1]
file_list = [line.rstrip('\n') for line in open(os.path.join(test_file_csv_path,csv_file))]
move_files(no_kermit_train_path,no_kermit_test_path,file_list,test_list)


test_labels = pd.DataFrame(test_list, columns=['index','file','test'])
test_labels.set_index(['index','file'],inplace=True)
audio_features_labelled = pd.read_csv(os.path.join(raw_audio_path,'audio_features_labelled.csv'),index_col=['index','file'])
audio_features_labelled_test = audio_features_labelled.join(test_labels)
audio_features_labelled_test.fillna(False,axis=0,inplace=True)
audio_features_labelled_test.to_csv(os.path.join(raw_audio_path,'audio_features_labelled_test.csv'),index=True)
