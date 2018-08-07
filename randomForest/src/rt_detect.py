#!/usr/bin/env python

from Data import sample
import os
import io
import sys
import numpy as np
import cPickle
import librosa
import feature_extraction

import librosa.display
import matplotlib.pyplot as plt
from urllib2 import urlopen
from bottle import route, run, request


def generateDetectData(feat):
    '''
    This creates a list of data samples to be tested.
    Theses samples are from participants 2 and 3 and
    the calculated angles are used as features.
    '''
    detectSamples = []

    for line in feat:
        features = np.array(line)
        detectSamples.append(sample(features))

    return detectSamples


def detect(rf, feat):
    '''
    detect a set of data
    '''
    # load random forest model
    with open(rf, 'rb') as f:
        testForest = cPickle.load(f)

    # generate feature array
    detectList = generateDetectData(feat)

    # classify sample or samples
    for samp in detectList:
        resultLabel = testForest.classify(samp)
        # print(resultLabel)

    return resultLabel


def plot(y, sr, point):
    plt.figure()
    plt.subplot()
    librosa.display.waveplot(y=y, sr=sr)
    plt.ylim(-1, 1)
    for i, p in enumerate(point):
        if i < len(point)-1:
            plt.plot([p[0], point[i+1][0]], [p[1], point[i+1][1]], 'r-')
    plt.show()
    plt.waitforbuttonpress()
    plt.close


def realtime_detect(af, af_url, rf):
    if not af:  # no local file, download from url
        # y, sr = sf.read(io.BytesIO(urlopen(af_url).read()))
        af = '/opt/audio_process/audio.mp3'
        audio_f = urlopen(af_url, timeout=5)                                    
        if not audio_f:                                                         
            return "url request timeout"                                        
        with open(af, 'wb') as audio:                                           
            audio.write(audio_f.read())                                         
    y, sr = librosa.load(af) 
    raw_feat = feature_extraction.feature(y, sr=sr)                             
    feat = raw_feat                                                  
    feat = feat.reshape(1, -1)                                                  
    result = detect(rf, feat)                                                
    return result  
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

def audio_detect(url):
    model = os.path.join('/opt/audio_process/randomForest/model', 'new_model') 
    result = realtime_detect(None, url, model)                                  
    print "audio result: ", result                                              
    return result
    # if isinstance(result, str):                                                 
    #     return result                                                           
    # return 'negative' if result[0] > 0 else 'positive' 



if __name__ == '__main__':
    model = os.path.join('/opt/audio_process/randomForest/model', 'new_model') 
    ff = '/opt/audio_process/audio/abnormal'
    for f in os.listdir(ff):
        f = os.path.join(ff, f)
        print realtime_detect(f, None, model)                                  
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
# @route('/')
# def index():
#     rf = os.path.join(os.path.abspath('..'), 'model/new_model')
#     af = request.query.file
#     af_url = request.query.url
#     return realtime_detect(af, af_url, rf)

# run(host='192.168.0.252', port=2444)
