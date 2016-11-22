import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow

lnt_res = pd.read_csv('data/submission_lenet5.csv')['Label']
svm_res = pd.read_csv('data/submission_svm.csv')['Label']
vgg_res = pd.read_csv('data/submission_vgg.csv')['Label']
test = pd.read_csv('data/test.csv')

cnt = 0

for i in range(28000):
    if lnt_res.iloc[i] != svm_res.iloc[i] and lnt_res.iloc[i] != vgg_res.iloc[i] and vgg_res.iloc[i] != svm_res.iloc[i]:
        print(i)
        cnt += 1
print(cnt)
