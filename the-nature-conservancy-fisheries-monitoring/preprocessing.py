#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
from PIL import Image


CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
COL = 160
ROW = 90

try:
    os.mkdir('data/train_resized')
except:
    pass
for fd in CLASSES:
    files = glob.glob('data/train/%s/*.jpg' % fd)
    try:
        os.mkdir('data/train_resized/%s' % fd)
    except:
        pass
    for fl in files:
        img = Image.open(fl)
        img = img.resize((COL, ROW), Image.HAMMING)
        fname = os.path.basename(fl)
        img.save('data/train_resized/%s/%s' % (fd, fname))

try:
    os.mkdir('data/test_resized')
except:
    pass
for fl in glob.glob('data/test_stg1/*.jpg'):
    img = Image.open(fl)
    img = img.resize((COL, ROW), Image.HAMMING)
    fname = os.path.basename(fl)
    img.save('data/test_stg1_resized/%s' % fname)
