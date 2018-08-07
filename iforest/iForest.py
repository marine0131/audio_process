#! /usr/bin/env python

import sys
sys.path.append("..")
import numpy as np
from sklearn.ensemble import IsolationForest
import json
import cPickle as pickle
import os
from audioAnalysis import feature_generate

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    x = []
    for l in lines:
        x.append(json.loads(l)['feature'])

    y = np.array([])
    if len(x) > 0:
        x = np.array(x)
        y = np.c_[x[:,1], x[:,3:6], x[:,12:14], x[:,16:18], x[:,24:26], x[:, 27:30]]
    return y


# train_set is  samples(n)xfeatures(m) numpy array
def train(train_set, contam):
    clf = IsolationForest(max_samples="auto", contamination=contam, verbose=1)
    clf.fit(train_set)

    return clf


# auto train regulary
def train_new(date_str):
    # calc features from audio file
    print "start processing date: {}'s audios".format(date_str)
    feature_generate.calc_feature(date_str)
    x = np.array([])
    for feature_file in feature_generate.get_feature_list():
        # print "process {}".format(feature_file)
        if len(x) == 0:
            x = load_data(feature_file)
        else:
            x = np.r_[x, load_data(feature_file)]
    contamination = 0.03
    itree = train(x, contamination)
    with open("./model/new_model", "wb") as f:
        pickle.dump(itree, f)
    print "successfuly update model"

# if __name__ == "__main__":
#     train_new("20180419")
if __name__ == "__main__":
    # read dataset
    X = []
    for feature_file in feature_generate.get_feature_list():
        print "process {}".format(feature_file)
        if len(X) == 0:
            X = load_data(feature_file)
        else:
            X = np.r_[X, load_data(feature_file)]
    contamination = 0.01  # float(len(C1))/float(len(X))

    # train
    itree = train(X, contamination)
    with open("./model/new_model", "wb") as f:
        pickle.dump(itree, f)

    # with open("./model", "r") as f:
    #     itree =  pickle.load(f)

    # x_z = itree.decision_function(X)
    x_predict = itree.predict(X)
    print('train error: {}'.format(x_predict[x_predict == -1].size/float(x_predict.size)))

    # test
    t1 = load_data("/home/whj/gitrepo/audio_process/audioAnalysis/feature_data/neg_0313.log")
    t2 = load_data("/home/whj/gitrepo/audio_process/audioAnalysis/feature_data/neg_0316.log")
    t = np.r_[t1, t2]
    # t_z = itree.decision_function(t)
    t_predict = itree.predict(t)
    print('test error: {}'.format(t_predict[t_predict == -1].size/float(t_predict.size)))

    ab1 = load_data("/home/whj/gitrepo/audio_process/audioAnalysis/feature_data/pos/pos_0313.log")
    ab2 = load_data("/home/whj/gitrepo/audio_process/audioAnalysis/feature_data/pos/pos_0319.log")
    ab = np.r_[ab1, ab2]
    # ab_z= itree.decision_function(ab)
    ab_predict = itree.predict(ab)
    print('abnormal error: {}'.format(ab_predict[ab_predict == 1].size/float(ab_predict.size)))

    # with open("./result.txt", 'wb') as f:
    #     tt = np.c_[x_z, x_predict]
    #     tt1 = np.c_[t_z, t_predict]
    #     tt2 = np.c_[ab_z, ab_predict]
    #     # np.savetxt(f, tt)
    #     # f.write('\n\n')
    #     np.savetxt(f, tt1)
    #     f.write('\n\n')
    #     np.savetxt(f, tt2)
