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

oriimg = cv2.imread("41861.jpg")
oriimg = cv2.resize(oriimg, (4000, 4000))
oriimgcopy = oriimg.copy()

smoothedimgmed = cv2.medianBlur(oriimgcopy, 121)
grayscaleimg = cv2.cvtColor(smoothedimgmed, cv2.COLOR_BGR2GRAY)
        
v_edges = cv2.Sobel(grayscaleimg, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
h_edges = cv2.Sobel(grayscaleimg, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(v_edges)
abs_grad_y = cv2.convertScaleAbs(h_edges)

grad = cv2.addWeighted(abs_grad_x, 0.8, abs_grad_y, 0.8, 0)
grad = cv2.bitwise_not(grad)

ret2,th2 = cv2.threshold(grad,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
masked_out = cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR)

maskedimg = cv2.subtract(masked_out, oriimgcopy)
maskedimg = cv2.cvtColor(maskedimg, cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(maskedimg,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

rects = []
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key = cv2.contourArea, reverse = True)

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)

    # Remove the overlapping big box
    if (h > 0.25*thresh.shape[0]) | (w > 0.25*thresh.shape[1]):
        pass
    elif (h > 5*w) | (w > 5*h):
        pass
    elif (h*w < thresh.shape[0]*thresh.shape[1]/52000):
        pass
    ## Set the threshold to allow only rectangles of certain size to be appended onto the image ##
    elif h >= 250:
        rect = (x, y, w, h)
        rects.append(rect)

presimrectsremove = oriimgcopy.copy()

rectcount = 0
for rect in rects:
    (x, y, w, h) = rect
    cv2.rectangle(presimrectsremove, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 8);
    cv2.putText(presimrectsremove, str(rectcount), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,255), 16)
    rectcount += 1

outputimagedisp = cv2.resize(presimrectsremove, (800,800))
cv2.imshow('Initial segmentation', outputimagedisp)

def get_overlap(rects):
    numofrects = []
    rectsTL = []
    rectsBR = []
    alliou = []
    fulloverlaplistpairs = []
    rectanglestokeep = []
    overlaplist = []
    overlappedrects = []
    rectsTL = []
    rectsBR = []
    rectanglestocompare = []
    rectangleareas = []
    sortedoverlaplist = []
    uniqueoverlaplist = []
    rectanglestoremove = []
    resultslist = []
        
    rectcount = 0
    print("Pre-remove overlaps: " + str(len(rects)))

    for rect in rects:
        numofrects.append(rectcount)
        rectcount += 1
            
    for i in range(len(rects)):
        rectsTL.append((rects[i][0],rects[i][1]))
        rectsBR.append((rects[i][0]+rects[i][2],rects[i][1]+rects[i][3]))

    for rectone in range(len(rects)):
        for recttwo in range(len(rects)):
            if numofrects[rectone] == numofrects[recttwo]:
                pass
            else:
                width = min(rectsBR[rectone][0], rectsBR[recttwo][0]) - max(rectsTL[rectone][0], rectsTL[recttwo][0])
                height = min(rectsBR[rectone][1], rectsBR[recttwo][1]) - max(rectsTL[rectone][1], rectsTL[recttwo][1])
                if width <= 0 or height <= 0:
                    alliou.append(0)
                else:
                    Area = width * height
                    rectoneArea = (rectsTL[rectone][0]-rectsBR[rectone][0])*(rectsTL[rectone][1]-rectsBR[rectone][1])
                    recttwoArea = (rectsTL[recttwo][0]-rectsBR[recttwo][0])*(rectsTL[recttwo][1]-rectsBR[recttwo][1])
                    # If the overlapping area is more than 50% #
                    if (Area >= rectoneArea * (50/100) ) | (Area >= recttwoArea * (50/100)):
                        overlaplist.append([numofrects[rectone], numofrects[recttwo]])
                        #print(str(numofrects[rectone]) + " and " +  str(numofrects[recttwo]) + " overlaps. The area of overlap is " + str(Area) + " px.")

    for i in range(len(overlaplist)):
        sortedoverlaplist.append(sorted(overlaplist[i]))

    sortedoverlaplist.sort()
    uniqueoverlaplist = (list(sortedoverlaplist for sortedoverlaplist,_ in itertools.groupby(sortedoverlaplist)))

    for i in range(len(uniqueoverlaplist)):
        rectanglestoremove.append(uniqueoverlaplist[i][1])

    rectanglestoremove = list(set(rectanglestoremove))
    rectanglestoremove = sorted(rectanglestoremove, reverse=True)
            
    for i in rectanglestoremove:
        del rects[i]

    print("Post-remove overlaps: " + str(len(rects)))

    return rects

def similarity_test(rects):
    from scipy.stats.stats import pearsonr

    print("Pre-ratio test: " + str(len(rects)))

    hwratio = []
    hwratioindex = []
    for rect in rects:
        hwratio.append(rect[2]/rect[3]) 

    meanhwratio = (statistics.mean(hwratio))
    hwthresholdupper = meanhwratio + (meanhwratio * 0.5)
    hwthresholdlower = meanhwratio - (meanhwratio * 0.5)

    for i in hwratio:
        if (i > hwthresholdupper) | (i < hwthresholdlower):
            hwratioindex.append(hwratio.index(i))

    hwratioindex = list(set(hwratioindex))
    hwratioindex = sorted(hwratioindex, reverse=True)

    for i in hwratioindex:
        del rects[i]
        
    similaritylist = []
    similaritylistpairs = []
    simrect1 = rects[:]
    simrect2 = rects[:]

    print("Pre-similarity test: " + str(len(rects)))

    numofrects = []
    rectcount = 0
    for i in range(len(rects)):
        numofrects.append(rectcount)
        rectcount+=1
            
    for rect1 in simrect1:
        for rect2 in simrect2:
            if rect1 == rect2:
                pass
            else:
                imroi1 = oriimgcopy[int(rect1[1]):int(rect1[1])+int(rect1[3]), int(rect1[0]):int(rect1[0])+int(rect1[2])]
                imroi2 = oriimgcopy[int(rect2[1]):int(rect2[1])+int(rect2[3]), int(rect2[0]):int(rect2[0])+int(rect2[2])]

                hist1,bins1 = np.histogram(imroi1,256,[0,256])
                hist2,bins2 = np.histogram(imroi2,256,[0,256])

                s = pearsonr(hist1, hist2)
                s = s[0]
                
                if s >= 0.95:
                    similaritylist.append((numofrects[simrect1.index(rect1)]))
                    similaritylistpairs.append((numofrects[simrect1.index(rect1)], numofrects[simrect2.index(rect2)], s))
                    
    print("Finding similar rectangle pairs: " + str(similaritylistpairs))
    similaritylist = list(set(similaritylist))
    similarity_diff = list(set(numofrects) - set(similaritylist))
    similarity_diff = sorted(similarity_diff, reverse = True)

    ## Rectangles without a similar pair are removed ## 
    print("Dissimilar rectangles to be removed: " + str(similarity_diff))

    if len(similarity_diff) != len(rects):
        for i in similarity_diff:
            del rects[i]

    print("Post-similarity: " + str(len(rects)))

    postremovsim1img = oriimg.copy()
    rectcountpostrem = 0
    for rect in rects:
        (x, y, w, h) = rect
        cv2.rectangle(postremovsim1img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 8);
        cv2.putText(postremovsim1img, str(rectcountpostrem), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,255), 16)
        rectcountpostrem += 1

    postremovsim1imgr = cv2.resize(postremovsim1img, (800,800))
    cv2.imshow('Post-similarity 1', postremovsim1imgr)

    similaritylist2 = []
    ssimrect1 = rects[:]
    ssimrect2 = rects[:]

    numofrectsnew = []
    rectcountnew = 0
    for rect in range(len(rects)):
        numofrectsnew.append(rectcountnew)
        rectcountnew += 1

    gray = cv2.cvtColor(oriimgcopy, cv2.COLOR_BGR2GRAY)
    for rect1 in ssimrect1:
        for rect2 in ssimrect2:
            if rect1 == rect2:
                pass
            else:
                imgray1 = gray[int(rect1[1]):int(rect1[1])+int(rect1[3]), int(rect1[0]):int(rect1[0])+int(rect1[2])]
                imgray2 = gray[int(rect2[1]):int(rect2[1])+int(rect2[3]), int(rect2[0]):int(rect2[0])+int(rect2[2])]
                _, im1bin = cv2.threshold(imgray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                _, im2bin = cv2.threshold(imgray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                im1 = Image.fromarray(im1bin)
                im2 = Image.fromarray(im2bin)

                if im1.width<im2.width:
                    im2=im2.resize((im1.width,im1.height))
                else:
                    im1=im1.resize((im2.width,im2.height))

                im1 = np.array(im1)
                im2 = np.array(im2)
                s = skimage.measure.compare_ssim(im1, im2)
                similaritylist2.append((numofrectsnew[ssimrect1.index(rect1)], s))

    similaritydic = {x:0 for x, _ in similaritylist2}

    for name, num in similaritylist2: similaritydic[name] += num

    mostsimilar = list(map(tuple, similaritydic.items()))
    mostsimilar = sorted(mostsimilar, key = lambda x: x[1], reverse=True)

    mostsimilarlist = []

    for i in range(len(mostsimilar)):
        mostsimilarlist.append(mostsimilar[i][0])

removeoverlappedrects = get_overlap(rects)

postrectsremoved = oriimgcopy.copy()

rectcount2 = 0
for rect in removeoverlappedrects:
    (x, y, w, h) = rect
    cv2.rectangle(postrectsremoved, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 8);
    cv2.putText(postrectsremoved, str(rectcount2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0,0,255), 35)
    rectcount2 += 1

outputimageremov = cv2.resize(postrectsremoved, (800,800))
cv2.imshow('After overlap removal', outputimageremov)


similarity_test(removeoverlappedrects)
