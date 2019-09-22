## Algorithm to detect the amount of insects in an image using Template Matching ##
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
# https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html
#https://stackoverflow.com/questions/48732991/search-for-all-templates-using-scikit-image
import numpy as np
import cv2
import os
import tkinter
import re
from tkinter import filedialog
from matplotlib import pyplot as plt
import statistics
import math
import sklearn
import skimage
import itertools
from scipy.stats.stats import pearsonr
from itertools import groupby, product, islice
from PIL import Image

template1 = cv2.imread("roitest1tm.png")
template1gray = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template1gray = cv2.resize(template1gray, (300, 300))

template2 = cv2.imread("roitest2tm.png")
template2gray = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
template2gray = cv2.resize(template2gray, (300, 300))

template3 = cv2.imread("roitestnoise.png")
template3gray = cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)
template3gray = cv2.resize(template3gray, (300, 300))

template4 = cv2.imread("roitestnoise2.png")

hist1,bins1 = np.histogram(template1,256,[0,256])
hist2,bins2 = np.histogram(template2,256,[0,256])
hist3,bins3 = np.histogram(template3,256,[0,256])
hist4,bins4 = np.histogram(template4,256,[0,256])

s1 = pearsonr(hist1, hist2)
print("Pearson CC between two insects: " + str(s1[0]))
s2 = pearsonr(hist1, hist3)
print("Pearson CC between insect 1 and noise 1: " + str(s2[0]))
s3 = pearsonr(hist2, hist3) 
print("Pearson CC between insect 2 and noise 1: " + str(s3[0]))
s4 = pearsonr(hist1, hist4)
print("Pearson CC between insect 1 and noise 2: " + str(s4[0]))
s5 = pearsonr(hist2, hist4)
print("Pearson CC between insect 2 and noise 2: " + str(s5[0]))

'''
fig = plt.figure()
plt.subplot(2, 4, 1)
plt.title('Insect 1')
plt.plot(hist1)
plt.subplot(2, 4, 2)
plt.title('Insect 2')
plt.plot(hist2)
plt.subplot(2, 4, 3)
plt.title('Noise 1')
plt.plot(hist3)
plt.subplot(2, 4, 4)
plt.title('Noise 2')
plt.plot(hist4)
plt.subplot(2, 4, 5)
plt.imshow(template1)
plt.subplot(2, 4, 6)
plt.imshow(template2)
plt.subplot(2, 4, 7)
plt.imshow(template3)
plt.subplot(2, 4, 8)
plt.imshow(template4)
plt.show()
'''

_, im1bin = cv2.threshold(template1gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_, im2bin = cv2.threshold(template2gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_, im3bin = cv2.threshold(template3gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('bin1', im1bin)
cv2.imshow('bin2', im2bin)
cv2.imshow('bin3', im3bin)

im1 = Image.fromarray(im1bin)
im2 = Image.fromarray(im2bin)
im3 = Image.fromarray(im3bin)

im1 = np.array(im1)
im2 = np.array(im2)
im3 = np.array(im3)

ssim1 = skimage.measure.compare_ssim(im1, im2)
ssim2 = skimage.measure.compare_ssim(im1, im3)
ssim3 = skimage.measure.compare_ssim(im2, im3)

print("SSIM between two insects: " + str(ssim1))
print("SSIM between insect 1 and noise 1: " + str(ssim2))
print("SSIM between insect 2 and noise 1: " + str(ssim3))
