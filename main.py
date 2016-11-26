# coding=utf-8
from time import time
from features import loadFeatures
from numpy import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
def main():
    starttime = time()
    featureMat = loadFeatures()
    print("features loading over...")
    random.shuffle(featureMat)
    print(featureMat.shape)


    train_data  = featureMat[:1000, :-1]
    train_label = featureMat[:1000,  -1]
    test_data   = featureMat[1000:6000, :-1]
    test_label  = featureMat[1000:6000,  -1]

    clf = AdaBoostClassifier(n_estimators=200)
    clf.fit(train_data, train_label)

    print("training over...")

    pred = clf.predict(test_data)

    for i in range(len(pred)):
        print((pred[i], test_label[i]))
    print(accuracy_score(test_label, pred))

    endtime = time()

    print(endtime-starttime)





    pass

if __name__ == "__main__":
    main()