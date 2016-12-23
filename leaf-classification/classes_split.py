#!/usr/bin/env python3

import os
import pandas as pd


data = pd.read_csv('data/train.csv')
for i in range(data.shape[0]):
    idx = data.loc[i, 'id']
    spe = data.loc[i, 'species']
    try:
        os.mkdir('data/classes/%s' % spe)
    except:
        pass
    os.system('cp data/images/%d.jpg data/classes/%s/' % (idx, spe))
