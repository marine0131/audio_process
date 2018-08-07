import os
import librosa
import feature_extraction
import svm
from svmutil import *

features = []
labels = []
for file_name in os.listdir('test/'):
    y, sr = librosa.load('test/'+file_name)
    feat = feature_extraction.feature(y, sr=sr)
    feat = feat.tolist()
    features.append(feat)
    labels.append(-1)

model = svm_load_model('audio.model')
pre, acc, val = svm_predict(labels, features, model)
print('accuracy: ', acc)
print(pre)
