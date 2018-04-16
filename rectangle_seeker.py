# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:24:21 2018

@author: Rebecca Napolitano
"""

import numpy as np
import cv2


#load a color image 

fullImageFileName = 'G:\My Drive\Documents\Research\mikehess\paper1_baptistery\computervision\wall.jpg'
img = cv2.imread(fullImageFileName)
cv2.imshow('image', img)