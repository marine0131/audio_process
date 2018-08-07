#! /usr/bin/env python

import numpy as np
from sklearn.ensemble import IsolationForest
import json
import cPickle as pickle


def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    x = []
    for l in lines:
        x.append(json.loads(l)['feature'])

    x = np.array(x)
    y = np.c_[x[:,1], x[:,3:6], x[:,12:14], x[:,16:18], x[:,24:26], x[:, 27:30]]
    # y = x
    return y

# train_set is  samples(n)xfeatures(m) numpy array
def train(train_set, contam):
    clf = IsolationForest(max_samples="auto", contamination=contam, verbose=1)
    clf.fit(train_set)

    return clf


if __name__ == "__main__":
    # read dataset
    X1 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0328_part1.log")
    X2 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0328_part2.log")
    X3 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0328_part3.log")
    X4 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0329.log")
    C1 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/pos_0319.log")
    X = np.r_[X1, X2, X3, X4]
    contamination = 0.05  # float(len(C1))/float(len(X))
    print("process {} samples with contamination: {}".format(len(X), contamination))

    # train
    itree = train(X, contamination)
    with open("./model/new_model", "wb") as f:
        pickle.dump(itree, f)

    # with open("./model", "r") as f:
    #     itree =  pickle.load(f)

    x_z = itree.decision_function(X)
    x_predict = itree.predict(X)

    # test
    t1 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0313.log")
    t2 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0316.log")
    t = np.r_[t1, t2]
    t_z = itree.decision_function(t)
    t_predict = itree.predict(t)

    ab1= load_data("/home/whj/gitrepo/audioAnalysis/feature_data/pos_0313.log")
    ab2= load_data("/home/whj/gitrepo/audioAnalysis/feature_data/pos_0319.log")
    ab = np.r_[ab1, ab2]
    ab_z= itree.decision_function(ab)
    ab_predict = itree.predict(ab)

    with open("./result.txt", 'wb') as f:
        tt = np.c_[x_z, x_predict]
        tt1 = np.c_[t_z, t_predict]
        tt2 = np.c_[ab_z, ab_predict]
        # np.savetxt(f, tt)
        # f.write('\n\n')
        np.savetxt(f, tt1)
        f.write('\n\n')
        np.savetxt(f, tt2)
