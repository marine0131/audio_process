#!/usr/bin/env python

import sys
sys.path.append("..")
import os
import numpy as np
import cPickle as pickle
import librosa
from audioAnalysis import feature_extraction

from urllib2 import urlopen
from bottle import route, run, request


def detect(model, feat):
    '''
    detect a set of data
    '''
    # load random forest model
    with open(model, 'rb') as f:
        itree = pickle.load(f)

    return itree.predict(feat)


def realtime_detect(af, af_url, model):
    if not af:  # no local file, download from url
        # y, sr = sf.read(io.BytesIO(urlopen(af_url).read()))
        af = 'audio.mp3'
        with open(af, 'wb') as audio:
            audio.write(urlopen(af_url).read())
    try:
        y, sr = librosa.load(af)
    except Exception as e:
        return e
    raw_feat = feature_extraction.feature(y, sr=sr)
    feat = np.r_[raw_feat[1], raw_feat[3:6], raw_feat[12:14], raw_feat[16:18],
                 raw_feat[24:26], raw_feat[27:30]]
    # reshape array for single sample
    feat = feat.reshape(1,-1)
    result = detect(model, feat)
    return result
    # start = 0
    # step = int(sr * 1)
    # pos = 0
    # total = 0
    # while start+step < len(y):
    #     # print('slice from: {}'.format(start))
    #     slice_y = y[start: start+step]
    #     feat = [feature_extraction.feature(slice_y, sr)]
    #     resultLabel = detect(rf, feat)
    #     pos = pos+1 if resultLabel == 'positive' else pos
    #     total += 1
    #     start = start+step

    # if float(pos)/float(total) > 0.7:
    #     return 'positive'
    # return 'negative'


'''
arg1: random froest model
arg2: audio file
'''
# if __name__ == '__main__':
#     rf = os.path.join(os.path.abspath('..'), 'model/trainedForest_norm')
#     ff = sys.argv[1]
#
#     y, sr = librosa.load(ff)
#     start = 0
#     step = int(sr * 1)
#     feat = []
#     # print(y.shape)
#     point = []
#     while start+step < len(y):
#         # print('slice from: {}'.format(start))
#         slice_y = y[start: start+step]
#         feat = [feature_extraction.feature(slice_y, sr)]
#         resultLabel = detect(rf, feat)
#         # plot
#         result = 0.5 if resultLabel == 'positive' else 0.05
#         point.append([start/sr, result])
#         point.append([(start+step)/sr, result])
#
#         start = start+step
#
#     plot(y, sr, point)


'''
file: audio file absolute path in string
url: audio file download url
'''
@route('/')
def index():
    itree = os.path.join(os.path.abspath('./model'), 'new_model')
    f = request.query.file
    f_url = request.query.url
    # ff = urllib2.urlopen(url)
    result = realtime_detect(f, f_url, itree)
    return 'negative' if result[0] > 0 else 'positive'

run(host='192.168.10.12', port=2444)


# self test
# itree = os.path.join(os.path.abspath('./model'), 'model_20180329')
# f = "/home/whj/gitrepo/audioAnalysis/audio_file/neg_0313/20180313_111442.mp3"
# f_url = None
# print('negative' if realtime_detect(f, f_url, itree)[0] > 0 else 'positive')
# af = '/home/whj/gitrepo/randomForest/audio_file/0319_neg'
# for f in os.listdir(af):
#     ff = os.path.join(af, f)
#     print(f, realtime_detect(ff, af_url, rf))
