#!/usr/bin/env python3
import pandas as pd
import numpy as np

# before the 36 round of vgg
# I made a single round test in data/vgg_results/ubmission_vgg.csv


base = pd.read_csv('data/vgg_results/submission_vgg.csv')['Label']
data = np.empty(shape=(37, base.shape[0]))
data[36] = base

for i in range(36):
    data[i] = pd.read_csv('data/vgg_results/submission_vgg_%d.csv' % i)['Label']

data = data.transpose()

for i in range(data.shape[0]):
    base[i] = np.argmax(np.bincount(data[i].astype(int)))
sub = pd.read_csv('data/sample_submission.csv')
sub['Label'] = base
sub.to_csv('data/submission_vgg_ensemble.csv', index=False)
