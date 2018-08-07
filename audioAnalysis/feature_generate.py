#! /usr/bin/env python
import os
import sys
import json
import feature_extraction
from glob import glob
import random
import librosa


AUDIO_PATH = "/opt/gxxj_robot/upload/voice/"
# AUDIO_PATH = "/home/whj/gitrepo/audio_process/audio_file/neg_0419/"

# get feature file list in /feature folder
def get_feature_list():
    abs_dir = os.path.join(os.path.dirname(feature_extraction.__file__), 'feature_data')
    filename_list = os.listdir(abs_dir)
    featurefile_list = []
    for filename in filename_list:
        if filename.split('.')[-1] == 'log':
            featurefile_list.append(os.path.join(abs_dir, filename))
    return featurefile_list


# calc feature according to the date  20180419
def calc_feature(date_str):
    audio_list = glob(AUDIO_PATH + date_str + '*.mp3')
    feature_file = os.path.join(os.path.dirname(feature_extraction.__file__), 'feature_data/neg_' + date_str + '.log')
    print "saving features to {} ...".format(feature_file)
    feat = {}
    if len(audio_list) > 1:
        # sample the audio_list
        if len(audio_list) > 1500:
            audio_list = random.sample(audio_list, 1500)

        with open(feature_file, 'w') as f_file:
            for f in audio_list:
                print("processing: {}".format(f))
                try:
                    y, sr = librosa.load(f)
                    feat['feature'] = list(feature_extraction.feature(y, sr=sr))
                except Exception as e:
                    print(e)
                    continue
                feat_json = json.dumps(feat)
                f_file.write(feat_json+'\n')


'''
extract feature array from audio file,
save them to file
arg 1: audio folder
arg 2: feature filename
'''
if __name__ == "__main__":
    ff = []
    try:
        if os.path.isdir(sys.argv[1]):
            # ff = [os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1]) if f.split('.')[1]=='mp3']
            ff = [os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1])]
        feature_file = 'feature_data/' + sys.argv[2]
    except Exception as e:
        print(e)

    feat = {}
    # sample the audio-list
    if len(ff) > 1500:
        ff = random.sample(ff, 1500)

    with open(feature_file, 'w') as f_file:
        for f in ff:
            print('processing: {}'.format(f))
            y, sr = librosa.load(f)
            try:
                feat['feature'] = list(feature_extraction.feature(y, sr=sr))
            except Exception as e:
                print(e)
                continue
            feat_json = json.dumps(feat)
            f_file.write(feat_json+'\n')
    print('saved feature array to ' + feature_file)
