#!/usr/bin/env python3

from PIL import Image
import numpy as np

SIZE = 64

for i in range(1, 1585):
    img = Image.open('data/images/%d.jpg' % i)
    data = np.array(img.getdata()).reshape(img.height, img.width)
    print(data.shape)
    diff = abs(img.height - img.width)
    mx = max(img.height, img.width)
    if img.height > img.width:
        data = np.concatenate((
            np.zeros((mx, diff // 2), dtype=int),
            data,
            np.zeros((mx, diff - diff // 2), dtype=int)
        ), axis=1)
    elif img.height < img.width:
        data = np.concatenate((
            np.zeros((diff // 2, mx), dtype=int),
            data,
            np.zeros((diff - diff // 2, mx), dtype=int)
        ), axis=0)

    resized = Image.new('L', (mx, mx))
    resized.putdata(data.flatten())
    resized = resized.resize((SIZE, SIZE), Image.HAMMING)

    resized.save('data/augmented/%d_0.jpg' % i)
    resized.rotate(90).save('data/augmented/%d_1.jpg' % i)
    resized.rotate(180).save('data/augmented/%d_2.jpg' % i)
    resized.rotate(270).save('data/augmented/%d_3.jpg' % i)
