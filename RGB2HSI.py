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

def RGB2HSI(rgb):
    rgb = np.array(rgb.astype(float));
    r = rgb[:, :, 0];
    g = rgb[:, :, 1];
    b = rgb[:, :, 2];
    
    rs = len(r)
    # gs = len(g)
    # bs = len(b)
    
    rs1 = len(r[0])
    # gs1 = len(g[0])
    # bs1 = len(b[0])
    
    H = numpy.zeros([len(r), len(r[0])])
    S = numpy.zeros([len(r), len(r[0])])
    I = numpy.zeros([len(r), len(r[0])])
    p = numpy.zeros([len(r), len(r[0])])
    q = numpy.zeros([len(r), len(r[0])])
    w = numpy.zeros([len(r), len(r[0])])
    h = numpy.zeros([len(r), len(r[0])])
    
    for i in range(0, rs):
        for j in range(0, rs1):
            p[i][j] = ((r[i][j] - g[i][j]) + (r[i][j] - b[i][j])) * (0.5)
            q[i][j] = ((r[i][j] - g[i][j]) ** 2 + (r[i][j] - b[i][j]) * (g[i][j] - b[i][j])) ** (0.5)
            w[i][j] = (((r[i][j] - g[i][j]) + (r[i][j] - b[i][j])) * (0.5)) / (((r[i][j] - g[i][j]) ** 2 + (r[i][j] - b[i][j]) * (g[i][j] - b[i][j])) ** (0.5))
                
            h[i][j] = numpy.arccos(w[i][j])
            I[i][j] = (r[i][j] + g[i][j] + b[i][j]) / 3
            S[i][j] = 1 - 3 / (r[i][j] + g[i][j] + b[i][j]) * min(r[i][j], g[i][j], b[i][j])
                                                                               
            if b[i][j] <= g[i][j]:
                H[i][j] = h[i][j]
            else:
                H[i][j] = 2*pi - h[i][j]

HSI = numpy.zeros([len(H), len(H[0]), 4])
HSI[:, :, 1] = H;
HSI[:, :, 2] = S;
HSI[:, :, 3] = I;

return HSI
