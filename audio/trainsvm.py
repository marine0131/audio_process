import librosa
import svm 
from svmutil import *
import os
import feature_extraction

data = []
labels = []
for file_name in os.listdir('normal/'):
    y, sr = librosa.load('normal/'+file_name)
    feat = feature_extraction.feature(y, sr=sr)
    feat = feat.tolist()
    data.append(feat)
    labels.append(1)
print(len(labels))

for file_name in os.listdir('abnormal/'):
    y, sr = librosa.load('abnormal/'+file_name) 
    feat = feature_extraction.feature(y, sr=sr)
    feat = feat.tolist()
    data.append(feat)
    labels.append(-1)
print(len(labels))

bestcv = 0
for log2c in range(-5,5):
    for log2g in range(-5,5):
        param = '-v 9 -c '+str(2**log2c)+' -g '+str(2**log2g)
        cv = svm_train(labels, data, param)
        if cv >= bestcv:
            bestcv = cv
            bestc = 2**log2c
            bestg = 2**log2g
print('bestc:',bestc, ' bestg: ',bestg, ' rate: ', bestcv)
print(len(labels))
param = '-c '+str(bestc)+' -g '+str(bestg)
model = svm_train(labels, data, param)
svm_save_model('audio.model',model)
