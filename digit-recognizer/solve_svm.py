import numpy as np
import pandas as pd
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from time import time

raw = pd.read_csv('data/train.csv')
train = raw.drop('label', 1)
train_y = raw['label']
print('data loaded')
best_n = 35
best_score = 0
for n in range(30, 40):
    t1 = time()
    pca = PCA(n_components=n, whiten=True)
    pca.fit(train)
    train_data = pca.transform(train)
    # score = cross_val_score(SVC(), train_data, train_y, cv=KFold(n_splits=10, shuffle=True))
    svc = SVC()
    score = 0.0
    for train_k, test_k in KFold(n_splits=10).split(train_data):
        svc.fit(train_data[train_k], train_y.iloc[train_k])
        pred = svc.predict(train_data[test_k])
        score += sum(map(lambda x: x[0] == x[1], zip(pred, train_y.iloc[test_k]))) / (len(pred) * 10.0)
    if best_score < score:
        best_n = n
        best_score = score
    t2 = time()
    print('the number of component: %d, accuracy: %.6f (+/- %.6f), time: %.3f' % \
        (n, score.mean(), score.std() * 2, t2 - t1))
print('found best number of component: %d' % best_n)
pca = PCA(n_components=best_n, whiten=True)
pca.fit(train)
train_data = pca.transform(train)
svc = SVC()
svc.fit(train_data, train_y)

test = pca.transform(pd.read_csv('data/test.csv'))
sub = pd.read_csv('data/sample_submission.csv')
sub['Label'] = svc.predict(test)
sub.to_csv('data/submission_svm.csv', index=False)
