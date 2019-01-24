import argparse, os
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
from functools import partial
import pickle
import math
import numpy
import PIL
from PIL import Image
import cv2


def HSI2RGB(h_i, b_hsi):  # H,S 성분은 im_b_hsi와 동일하다.
    # 즉 변경된 i 성분을 가지고 super resolution을 진행하는 것!
    h = b_hsi[:, :, 1];
    s = b_hsi[:, :, 2];
    i = h_i;

    hs = len(h)
    hs1 = len(h[0])

    R = numpy.zeros([hs, hs1])
    G = numpy.zeros([hs, hs1])
    B = numpy.zeros([hs, hs1])

    flag = 0  # 0 : color 1:흑백

    for k in range(0, hs):
        for j in range(0, hs1):

            flag = 0

            if h[k][j] <= 1.2e-16 or (numpy.isnan(h[k][j]) == 1) :
                if  s[k][j] <= 1.2e-16 or (numpy.isnan(s[k][j])==1):

                    flag = 1
                    R[k][j] = i[k][j]
                    G[k][j] = i[k][j]
                    B[k][j] = i[k][j]


            if flag == 0 :
                if h[k][j] >= 0 and h[k][j] < pi*2/3:
                    B[k][j] = i[k][j] * (1 - s[k][j]);
                    R[k][j] = i[k][j] * (1 + ((s[k][j] * math.cos(h[k][j])) / (math.cos(pi/3 - h[k][j]))));
                    G[k][j] = 3 * i[k][j] - (R[k][j] + B[k][j]);
                elif h[k][j] >= pi*2/3 and h[k][j] < pi*4/3:
                    h[k][j] = h[k][j] - pi*2/3;
                    R[k][j] = i[k][j] * (1 - s[k][j]);
                    G[k][j] = i[k][j] * (1 + ((s[k][j] * math.cos(h[k][j])) / (math.cos(pi/3 - h[k][j]))));
                    B[k][j] = 3 * i[k][j] - (R[k][j] + G[k][j])
                elif h[k][j] >= pi*4/3 and h[k][j] <= pi*2:
                    h[k][j] = h[k][j] - pi*4/3;
                    G[k][j] = i[k][j] * (1 - s[k][j]);
                    B[k][j] = i[k][j] * (1 + ((s[k][j] * math.cos(h[k][j])) / (math.cos(pi/3 - h[k][j]))));
                    R[k][j] = 3 * i[k][j] - (G[k][j] + B[k][j]);


            if R[k][j]>255:
                R[k][j]=255
            if R[k][j]<0:
                R[k][j]=0
            if G[k][j]>255:
                G[k][j]=255
            if G[k][j]<0:
                G[k][j]=0
            if B[k][j]>255:
                B[k][j]=255
            if B[k][j]<0:
                B[k][j]=0

                        #     img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)


    RGB = numpy.zeros([len(R), len(R[0]), 3],np.uint8)
    RGB[:, :, 0] = R;
    RGB[:, :, 1] = G;
    RGB[:, :, 2] = B;

    return RGB
