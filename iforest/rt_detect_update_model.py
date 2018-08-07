#!/usr/bin/env python

import sys
sys.path.append("/opt/audio_process")
import os
import numpy as np
import cPickle as pickle
import librosa
from audioAnalysis import feature_extraction
import iForest
import datetime

from urllib2 import urlopen
from bottle import route, run, request
import thread
import svm
from svmutil import *

def detect(model, feat):
    '''
    detect a set of data
    '''
    # load random forest model
    with open(model, 'rb') as f:
        itree = pickle.load(f)

    return itree.predict(feat)

def svm_detect(model, feat):
    model = svm_load_model(model)
    pre, accuracy, value = svm_predict([-1 for i in range(len(feat))], feat, model)
    return value


def realtime_detect(af, af_url, model):
    if not af:  # no local file, download from url
        af = '/opt/audio_process/audio.mp3'
	audio_f = urlopen(af_url, timeout=5)
	if not audio_f:
	    return "url request timeout"
        with open(af, 'wb') as audio:
            audio.write(audio_f.read())
    y, sr = librosa.load(af)
    raw_feat = feature_extraction.feature(y, sr=sr)
    # feat = raw_feat.tolist()
    # result = svm_detect(model, feats)
    feat = np.r_[raw_feat[1], raw_feat[3:6], raw_feat[12:14], raw_feat[16:18],
                 raw_feat[24:26], raw_feat[27:30]]
    # reshape array for single sample
    feat = feat.reshape(1, -1)
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


def update_trainning(curr_time):
    date_str = str(curr_time.year)+str(curr_time.month).zfill(2)+str(curr_time.day-1).zfill(2)

    try:
        thread.start_new_thread(iForest.train_new, (date_str,))
    except Exception:
        print("Error: unable to start thread")


'''
arg1: random froest model
arg2: audio file
'''
# if __name__ == '__main__':
#     curr_time = datetime.datetime.now()
#     update_trainning(curr_time)
#     while True:
#         pass
#     svm_model = os.path.join('/opt/audio_process/iforest/model', 'svm_model.model') 
#     f = sys.argv[1]
#     result = realtime_detect(f, None, svm_model)
#     print "audio result: ", result


def audio_detect(url):
    model = os.path.join('/opt/audio_process/iforest/model', 'new_model')
    # model = os.path.join('/opt/audio_process/iforest/model', 'svm_model.model') 
    result = realtime_detect(None, url, model)
    print "audio result: ", result
    if isinstance(result, str):
	return result
    return 'negative' if result[0] > 0 else 'positive'

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

    # check if need to trainning new model
    curr_time = datetime.datetime.now()
    if curr_time.hour == 0 and curr_time.minute == 0 and curr_time.second <= 10:
        update_trainning(curr_time)

    return 'negative' if result[0] > 0 else 'positive'


# run(host='192.168.0.252', port=2444)


# self test
# itree = os.path.join(os.path.abspath('./model'), 'model_20180329')
# f = "/home/whj/gitrepo/audioAnalysis/audio_file/neg_0313/20180313_111442.mp3"
# f_url = None
# print('negative' if realtime_detect(f, f_url, itree)[0] > 0 else 'positive')
# af = '/home/whj/gitrepo/randomForest/audio_file/0319_neg'
# for f in os.listdir(af):
#     ff = os.path.join(af, f)
#     print(f, realtime_detect(ff, af_url, rf))
