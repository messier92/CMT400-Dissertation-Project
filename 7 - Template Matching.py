'''
RUN ON PYTHON 3.6
Template Matching code from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html, date unknown
Accessed 20-8-2019
'''

import numpy as np
import cv2
import os, os.path
import tkinter
import re
from tkinter import filedialog
from matplotlib import pyplot as plt
from PIL import Image
import xml.etree.ElementTree as xml
from collections import Counter
import statistics
import itertools
from itertools import accumulate, islice, product, groupby
from operator import itemgetter
import math
import sklearn
import pandas as pd
import skimage
from skimage import data, measure
from skimage.feature import peak_local_max, match_template

def whitened_borders(image):
    ### WHITENED BORDERS ###
    ### Theory: For each row and column in the image, the standard deviation of the pixels in the border
    ### is expected to be minimal compared to when the borders are separated by the background of the tray and the insects

    ## Remove the border before detecting the insects ##
    img_borders = image.copy()
    #orrimg_borders_resized = cv2.resize(oriimg_borders, (4000,4000))

    ## Extract the red channel ##
    img_borders_redchannel = img_borders[:, :, 2]
    rows, cols = img_borders_redchannel.shape

    ## Convert to array ##
    image_data = np.asarray(img_borders_redchannel)

    ## Get the values of the image ##
    rpxvalue = []
    for i in range(rows):
        for j in range(cols):
            rpxvalue.append((image_data[i, j]))

    rimage_aslist = list(rpxvalue[i:i+cols] for i in range(0, len(rpxvalue), cols))
    rimage_asarray = np.asarray(rimage_aslist)

    # Get vertical standard deviation
    rgbsta0 = np.std(rimage_asarray, axis = 0)
    # Get horizontal standard deviation
    rgbsta1 = np.std(rimage_asarray, axis = 1)

    axis0limit = []
    axis0limitindexnp = []
    axis0limitindexflat = []
    axis1limit = []
    axis1limitindexnp = []
    axis1limitindexflat = []
    toplistlimit = []
    bottomlistlimit = []
    rightlistlimit = []
    leftlistlimit = []

    ## Set sd threshold here ##
    sdthreshold = 20

    ## If the pixel value is below the threshold, keep it ##
    for i in range(len(rgbsta0)):
        if rgbsta0[i] < sdthreshold:
            axis0limit.append(rgbsta0[i])
            axis0limitindexnp.append(np.where(rgbsta0 == rgbsta0[i]))

    for i in range(len(axis0limitindexnp)):
        axis0limitindexflat.append(axis0limitindexnp[i][0][0])

    ## If the pixel value is below the threshold, keep it ##
    for i in range(len(rgbsta1)):
        if rgbsta1[i] < sdthreshold:
            axis1limit.append(rgbsta1[i])
            axis1limitindexnp.append(np.where(rgbsta1 == rgbsta1[i]))

    for i in range(len(axis1limitindexnp)):
        axis1limitindexflat.append(axis1limitindexnp[i][0][0])

    splitlisthresholdrows = rows/4
    splitlisthresholdcols = cols/4

    for i in axis0limitindexflat:
        if (i < splitlisthresholdrows) & (i <= 0.05*rows):
            leftlistlimit.append(i)
        elif (i > splitlisthresholdrows) & (i >= 0.95*rows):
            rightlistlimit.append(i)

    for i in axis1limitindexflat:
        if (i < splitlisthresholdcols) & (i <= 0.05*cols) :
            toplistlimit.append(i)
        elif (i > splitlisthresholdcols) & (i >= 0.95*cols):
            bottomlistlimit.append(i)

    try:
        toplimit = max(toplistlimit)
    except ValueError:
        toplimit = int(0.02*rows)

    try:
        bottomlimit = min(bottomlistlimit)
    except ValueError:
        bottomlimit = int(0.98*rows)

    try:
        leftlimit = max(leftlistlimit)
    except ValueError:
        leftlimit = int(0.02*cols)

    try:
        rightlimit = min(rightlistlimit)
    except ValueError:
        rightlimit = int(0.98*cols)

    ### WHITENED BORDERS: PART 2 ###
    img_borders = image.copy()
    mediansmoothedimg_borders = cv2.medianBlur(img_borders, 3)
    grayimg_borders = cv2.cvtColor(mediansmoothedimg_borders, cv2.COLOR_BGR2GRAY)

    v_edges = cv2.Sobel(grayimg_borders, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    h_edges = cv2.Sobel(grayimg_borders, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(v_edges)
    abs_grad_y = cv2.convertScaleAbs(h_edges)

    mask = np.zeros(grayimg_borders.shape, dtype=np.uint8)
    linewidth = ((grayimg_borders.shape[0] + grayimg_borders.shape[1])) / 50

    ## Detect the vertical edges ##
    magv = np.abs(v_edges)
    magv2 = (255*magv/np.max(magv)).astype(np.uint8)
    _, mag2 = cv2.threshold(magv2, 15, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mag2.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if h > grayimg_borders.shape[0] / 4 and w < linewidth:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    ## Detect the horizontal edges ##
    magh = np.abs(h_edges)
    magh2 = (255*magh/np.max(magh)).astype(np.uint8)
    _, mag2 = cv2.threshold(magh2, 15, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mag2.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if w > grayimg_borders.shape[1] / 4 and h < linewidth:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    kerneldilate = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kerneldilate, 10)

    dst = cv2.cornerHarris(dilation,2,3,0.001)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    #dstdisplay = cv2.resize(dilation, (800,800))
    #cv2.imshow('d', dstdisplay)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(dst,np.float32(centroids),(5,5),(-1,-1),criteria)

    cornerstoplist = []
    cornerstoplistx = []
    cornerstoplisty = []
    cornersbottomlist = []
    cornersbottomlistx = []
    cornersbottomlisty = []
    cornersleftlist = []
    cornersleftlistx = []
    cornersleftlisty = []
    cornersrightlist = []
    cornersrightlistx = []
    cornersrightlisty = []

    upperrowlimit = int(rows * (5/100))
    lowerrowlimit = int(rows * (95/100))
    leftcollimit = int(cols * (5/100))
    rightcollimit = int(cols * (95/100))
    heightpadding = int(rows) - 100
    widthpadding = int(cols) - 100

    for i in range(len(corners)):
        ## if the y coordinate is less than 80px ad more than 10px
        if ((corners[i][1] <= upperrowlimit) & (corners[i][1] >= 10)):
            cornerstoplist.append((int(corners[i][0]), int(corners[i][1])))
            cornerstoplistx.append((int(corners[i][0])))
            cornerstoplisty.append((int(corners[i][1])))

    try:
        maxcornerstoplist = int(max(cornerstoplisty))
    except statistics.StatisticsError:
        maxcornerstoplist = 0.02*rows
    except ValueError:
        maxcornerstoplist = 0.02*rows
        
    for i in range(len(corners)):
        ## If the y coordinate is more than 720px and less than 790px
        if ((corners[i][1] >= lowerrowlimit) & (corners[i][1] <= heightpadding)):
            cornersbottomlist.append((int(corners[i][0]), int(corners[i][1])))
            cornersbottomlistx.append((int(corners[i][0])))
            cornersbottomlisty.append((int(corners[i][1])))

    try:
        mincornersbottomlist = int(min(cornersbottomlisty))
    except statistics.StatisticsError:
        mincornersbottomlist = 0.98*rows
    except ValueError:
        mincornersbottomlist = 0.98*rows
        

    for i in range(len(corners)):
        ## If the x coordinate is less than 40px and more than 10px ##
        if ((corners[i][0] <= leftcollimit) & (corners[i][0] >= 10)):
            cornersleftlist.append((int(corners[i][0]), int(corners[i][1])))
            cornersleftlistx.append((int(corners[i][0])))
            cornersleftlisty.append((int(corners[i][1])))

    try:
        meancornersleftlist = int(statistics.mean(cornersleftlistx))
    except statistics.StatisticsError:
        meancornersleftlist = 0.02*cols
    except ValueError:
        meancornersleftlist = 0.98*rows
        

    for i in range(len(corners)):
        # If the x coordinate is more than 720px and less than 790px ##
        if ((corners[i][0] >= rightcollimit) & (corners[i][0] <= widthpadding)):
            cornersrightlist.append((int(corners[i][0]), int(corners[i][1])))
            cornersrightlistx.append((int(corners[i][0])))
            cornersrightlisty.append((int(corners[i][1])))
    try:
        meancornersrightlist = int(statistics.mean(cornersrightlistx))
    except statistics.StatisticsError:
        meancornersrightlist = 0.98*cols
    except ValueError:
        meancornersrightlist = 0.98*rows

    topborder = int(max(toplimit,maxcornerstoplist))
    bottomborder = int(min(bottomlimit, mincornersbottomlist))
    leftborder = int(max(leftlimit, meancornersleftlist))
    rightborder = int(min(rightlimit, meancornersrightlist))

    whitenedbordersimg = image.copy()
    topblank = 255 * np.ones(shape=[topborder, cols, 3], dtype = np.uint8)
    bottomblank = 255 * np.ones(shape=[rows-bottomborder, cols, 3], dtype = np.uint8)
    leftblank = 255 * np.ones(shape=[rows, leftborder, 3] , dtype = np.uint8)
    rightblank = 255 * np.ones(shape=[rows, cols-rightborder, 3] , dtype = np.uint8)

    whitenedbordersimg[0:topborder, 0:cols] = topblank
    whitenedbordersimg[bottomborder:rows] = bottomblank
    whitenedbordersimg[0:cols, 0:leftborder] = leftblank
    whitenedbordersimg[0:rows, rightborder:cols] = rightblank

    whitenedbordersimg = cv2.cvtColor(whitenedbordersimg, cv2.COLOR_BGR2GRAY)
    whitenedbordersimgsub = cv2.subtract(whitenedbordersimg, dilation)
    whitenedbordersimgsub = cv2.cvtColor(whitenedbordersimgsub,cv2.COLOR_GRAY2BGR)

    #whitenedbordersimgsubr = cv2.resize(whitenedbordersimgsub, (800,800))
    #cv2.imshow('dst',whitenedbordersimgsubr)

    return whitenedbordersimgsub

def segment_edges(image, threshold=12, 
                  variance_threshold=150, resize=True):

    original_height, original_width = image.shape[:2]
    if resize:
        image = cv2.resize(image, resize)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 3)
    display = gray.copy()

    # Thresholding
    v_edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, None, 1)
    h_edges = cv2.Sobel(gray, cv2.CV_32F, 0, 1, None, 1)
    mag = np.sqrt(v_edges ** 2 + h_edges ** 2)
    mag2 = (255*mag/np.max(mag)).astype(np.uint8)
    _, mag2 = cv2.threshold(mag2, threshold, 255, cv2.THRESH_BINARY)

    # Remove lines
    magv = np.abs(v_edges)
    mask_removelines = np.zeros(gray.shape, dtype = np.uint8)
    linewidth = ((image.shape[0] + image.shape[1]) /2) / 20

    mag2_removelines = (255*magv/np.max(magv)).astype(np.uint8)
    _, mag2_removelines = cv2.threshold(mag2_removelines, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mag2_removelines.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if h > image.shape[0] / 4 and w < linewidth:
            cv2.drawContours(mask_removelines, [contour], -1, 255, -1)

    magh = np.abs(h_edges)
    mag2_removelines = (255*magh/np.max(magh)).astype(np.uint8)
    _, mag2_removelines = cv2.threshold(mag2_removelines, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mag2_removelines.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if w > image.shape[1] / 4 and h < linewidth:
            cv2.drawContours(mask_removelines, [contour], -1, 255, -1)

    # Remove lines
    mag2[mask_removelines == 255] = 0

    display = np.dstack((mag2, mag2, mag2))
    
    contours, hierarchy = cv2.findContours(mag2.copy(),
                                         cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    container_filter=True
    size_filter=True
    index = 0
    while index >= 0:
        next, previous, child, parent = hierarchy[0][index]
        image_size = image.shape
        x, y, w, h = cv2.boundingRect(contours[index])
        area = image_size[0] * image_size[1]
        if w > h:
            ratio = float(w) / h
        else:
            ratio = float(h) / w

        fill_ratio = cv2.contourArea(contours[index]) / (w * h)
        is_right_shape = ratio < 5 and w * h > area / 3000
        is_container = (not 0.1 < fill_ratio < 0.8 and (w > image_size[1] * 0.35 or h > image_size[0] * 0.35))
        is_too_large = (w > image_size[1] * 0.4 and h > image_size[0] * 0.4)

        if is_right_shape and not (container_filter and is_container) and not (size_filter and is_too_large):
            rect = cv2.boundingRect(contours[index])
            rect += (contours[index],)
            rects.append(rect)
        else:
            pass
            
        index = next

    new_rects = []
    for rect in rects:
        im = gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        if np.var(im) > variance_threshold:
            new_rects.append(rect)
    
    rects = new_rects

    display = cv2.resize(display, (original_width, original_height))
    new_rects_display = []

    for rect in rects:
        fx = float(original_width) / resize[0]
        fy = float(original_height) / resize[1]
        new_rect = (rect[0] * fx, rect[1] * fy, rect[2] * fx, rect[3] * fy)
        new_rects_display.append(new_rect)
    rects = new_rects_display

    arealistremov = []
    arealistremov1index = []
    arealist = []
    
    for rect in rects:
        arealistremov.append(rect[2]*rect[3])

    meanarea = statistics.mean(arealistremov)
    #print(max(arealistremov))

    #print(arealist1)
    #print(meanarea)
    for i in arealistremov:
        if i > meanarea*5.5:
            arealistremov1index.append(arealistremov.index(i))

    arealistremov1index = sorted(arealistremov1index, reverse=True)
    for i in arealistremov1index:
        del rects[i]

    for rect in rects:
        arealist.append(rect[2]*rect[3])

    return rects, arealist, image

def determine_insect_size(rects, arealist, display):
    from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
    from sklearn.tree.export import export_text
    import pandas as pd

    numofrects = []
    rectcount = 0
    for i in range(len(arealist)):
        numofrects.append(rectcount)
        rectcount+=1

    ### Load csv ###
    col_names = ['No. of Rects', 'Max Area', 'Area Range', 'Standard Deviation', 'CLASSIFICATION']
    insectcsv = pd.read_csv("EdgeDetection_InsectSize_DT_Train.csv", header=None, names=col_names)
    feature_cols = ['No. of Rects', 'Max Area', 'Area Range', 'Standard Deviation']
    X = insectcsv[feature_cols]
    y = insectcsv.CLASSIFICATION
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) #split into 0.9 train and 0.3 test
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    imagearea = display.shape[0]*display.shape[1]

    arealist = sorted(arealist)
        
    maxarealistrange = int(len(arealist)*0.9)
    sixtyarealistrange = int(len(arealist)*0.6)
    fourtyarealistrange = int(len(arealist)*0.3)
    minarealistrange = int(len(arealist)*0.1)
        
    numofrects = len(rects)
        
    maxarea = int(statistics.mean(arealist[maxarealistrange:]))
    midarea = int(statistics.mean(arealist[fourtyarealistrange:sixtyarealistrange]))
    minarea = int(statistics.mean(arealist[:minarealistrange]))
    arearange = maxarea-minarea
    stdarea = int(statistics.stdev(arealist))

    testing_data = []
    testing_data.append((numofrects, maxarea, arearange, stdarea))

    print(testing_data)

    #Predict the response for test dataset
    y_pred = clf.predict(testing_data)
    if y_pred == 1:
        print("DT Prediction: Big Insects")
    else:
        print("DT Prediction: Small Insects")

    r = export_text(clf, feature_names=feature_cols)
    print(r)

    prediction = []
    if maxarea >= 56000:
        prediction.append(1)
    else:
        prediction.append(0)

    if stdarea >= 20000:
        prediction.append(1)
    else:
        prediction.append(0)

    if arearange >= 51000:
        prediction.append(1)
    else:
        prediction.append(0)

    if numofrects <= 350:
        prediction.append(1)
    else:
        prediction.append(0)

    print(prediction)
    
    prediction_final = sum(prediction)
    
    if (prediction_final >= 3):
        print("Processing Big Insects")
        removedoverlappedrects_big = get_overlap(rects)
        bigrects_analysed = analyse_areas_biginsects(removedoverlappedrects_big)
        templaterect = similarity_test(bigrects_analysed)
        template_matching(templaterect, image)
    else:
        print("Processing Small Insects")
        bigrectsremoved = analyse_areas_smallinsects(rects)
        mergedrects = merge_rectangles(bigrectsremoved)
        removedoverlappedrects_small = get_overlap(mergedrects)
        templaterect = similarity_test(removedoverlappedrects_small)
        template_matching(templaterect, image)
        #save_coordinates_to_xml(filename,numofrects_si,rects_si)

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
        rectanglestoremove.append(uniqueoverlaplist[i][0])

    rectanglestoremove = list(set(rectanglestoremove))
    rectanglestoremove = sorted(rectanglestoremove, reverse=True)
            
    for i in rectanglestoremove:
        del rects[i]

    print("Post-remove overlaps: " + str(len(rects)))

    return rects    

def analyse_areas_biginsects(rects):
    arealist = []
    smallrectstoremove = []
    rectnpvar = []
    rectnpindex = []

    print("Pre-analysed rects: " + str(len(rects)))

    for rect in rects:
        imroi = whitenedbordersimgsub[int(rect[1]):int(rect[1])+int(rect[3]), int(rect[0]):int(rect[0])+int(rect[2])]
        if np.var(imroi) < 100:
            rectnpvar.append(rect)

    for i in rectnpvar:
        rectnpindex.append(rects.index(i))

    rectnpindex = sorted(rectnpindex, reverse=True)
    for i in rectnpindex:
        del rects[i]

    for i in range(len(rects)):
        arealist.append(rects[i][2]*rects[i][3])

    for i in arealist:
        if i < statistics.mean(arealist)*0.20:
            smallrectstoremove.append(arealist.index(i))

    smallrectstoremove = sorted(smallrectstoremove, reverse=True)

    for i in smallrectstoremove:
        del rects[i]

    print("Post-analysed rects: " + str(len(rects)))

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
                imroi1 = whitenedbordersimgsub[int(rect1[1]):int(rect1[1])+int(rect1[3]), int(rect1[0]):int(rect1[0])+int(rect1[2])]
                imroi2 = whitenedbordersimgsub[int(rect2[1]):int(rect2[1])+int(rect2[3]), int(rect2[0]):int(rect2[0])+int(rect2[2])]

                hist1,bins1 = np.histogram(imroi1,256,[0,256])
                hist2,bins2 = np.histogram(imroi2,256,[0,256])

                s = pearsonr(hist1, hist2)
                s = s[0]
                
                if s >= 0.95:
                    similaritylist.append((numofrects[simrect1.index(rect1)]))

    similaritylist = list(set(similaritylist))
    similarity_diff = list(set(numofrects) - set(similaritylist))
    similarity_diff = sorted(similarity_diff, reverse = True)

    if len(similarity_diff) != len(rects):
        for i in similarity_diff:
            del rects[i]

    print("Post-similarity: " + str(len(rects)))

    similaritylist2 = []
    ssimrect1 = rects[:]
    ssimrect2 = rects[:]

    numofrectsnew = []
    rectcountnew = 0
    for rect in range(len(rects)):
        numofrectsnew.append(rectcountnew)
        rectcountnew += 1

    gray = cv2.cvtColor(whitenedbordersimgsub, cv2.COLOR_BGR2GRAY)
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

    mostsimilarrect = mostsimilarlist[0]

    templaterect = rects[mostsimilarrect]

    outputimagetem = whitenedbordersimgsub.copy()
    cv2.rectangle(outputimagetem, (int(templaterect[0]), int(templaterect[1])), (int(templaterect[0]+templaterect[2]), int(templaterect[1]+templaterect[3])), (0,255,0), 5);

    outputimagetem = cv2.resize(outputimagetem, (600,600))
    cv2.imshow('Template',outputimagetem)

    return templaterect

def draw_rectangles_biginsects(rects, image):
    global rects_bi, numofrects_bi

    outputimage = image.copy()
    
    numofrects_bi = []
    rectcountbi = 0
    rects_bi = rects[:]
    
    for rect in rects_bi:
        numofrects_bi.append(rectcountbi)
        (x, y, w, h) = rect
        cv2.rectangle(outputimage, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 8);
        #cv2.putText(outputimage, str(rectcountbi), (int(x), int((y)-10)), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0,0,255), 32)
        rectcountbi += 1

    outputimager = cv2.resize(outputimage, (600,600))
    cv2.imshow('Final Output',outputimager)

    return rects_bi, numofrects_bi

def draw_rectangles_smallinsects(rects, image):
    global rects_si, numofrects_si
    
    outputimage = image.copy()
    
    numofrects_si = []
    rectcountsi = 0
    rects_si = rects[:]
    
    for rect in rects_si:
        numofrects_si.append(rectcountsi)
        (x, y, w, h) = rect
        cv2.rectangle(outputimage, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 10);
        #cv2.putText(outputimage, str(rectcountbi), (int(x), int((y)-10)), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0,0,255), 32)
        rectcountsi += 1

    outputimager = cv2.resize(outputimage, (600,600))
    cv2.imshow('Final output', outputimager)

    return rects_si, numofrects_si

def analyse_areas_smallinsects(rects):
    arealist = []
    bigrectstoremove = []

    rectnpvar = []
    rectnpindex = []

    for rect in rects:
        imroi = whitenedbordersimgsub[int(rect[1]):int(rect[1])+int(rect[3]), int(rect[0]):int(rect[0])+int(rect[2])]
        if np.var(imroi) < 100:
            rectnpvar.append(rect)

    for i in rectnpvar:
        rectnpindex.append(rects.index(i))

    rectnpindex = sorted(rectnpindex, reverse=True)
    for i in rectnpindex:
        del rects[i]
 
    for i in range(len(rects)):
        arealist.append(rects[i][2]*rects[i][3])
            
    for i in arealist:
        if i > statistics.mean(arealist)*5:
            bigrectstoremove.append(arealist.index(i))

    bigrectstoremove = sorted(bigrectstoremove, reverse=True)

    for i in bigrectstoremove:
        del rects[i]

    return rects

def merge_rectangles(rects):
    print("Number of rects (pre-merged): " + str(len(rects)))
        
    centroidlistx = []
    centroidlisty = []
        
    numofrects = []
    rectstomerge_withduplicates = []
    sortedrectstomerge = []
    uniquerectsungrouped = []
    arealist = []
    rectcount = 0
    for rect in rects:
        numofrects.append(rectcount)
        rectcount += 1
        arealist.append(rect[2]*rect[3])

    ## Get the centroids of all the rectangles by adding half of the width and height to the x and y coordinates respectively ##
    for (x,y,w,h) in rects:
        centroidlistx.append(int(x+0.5*w))
        centroidlisty.append(int(y+0.5*h))

    mergethreshold = 10000/len(rects)

    print(statistics.mean(arealist)/100)
    print("Merging threshold: " +str(mergethreshold))

    ## Get all the rectangles within the merging threshold ##
    for i in range(len(centroidlistx)):
        for x in range(len(centroidlistx)):
            if numofrects[x] == numofrects[i]:
                pass
            # CHANGE THE MERGING THRESHOLD HERE
            elif (centroidlistx[x]-mergethreshold <= centroidlistx[i] <= centroidlistx[x]+mergethreshold) & (centroidlisty[x]-mergethreshold <= centroidlisty[i] <= centroidlisty[x]+mergethreshold):
                rectstomerge_withduplicates.append((numofrects[x], numofrects[i]))

    ## Group the rectangles to merge - remove the duplicates ie (27, 89) and (89, 27) - keep only one set ##
    for i in range(len(rectstomerge_withduplicates)):
        sortedrectstomerge.append(tuple(sorted(rectstomerge_withduplicates[i])))

    uniquerectstomerge = sorted(list(set(sortedrectstomerge)))
    uniquerectsgrouped = [(k, list(list(zip(*g))[1])) for k, g in groupby(uniquerectstomerge, itemgetter(0))]

    for i in range(len(uniquerectsgrouped)):
        for x in range(len(uniquerectsgrouped[i])):
            uniquerectsungrouped.append((uniquerectsgrouped[i][x]))

    ## Un-group it for connectivity ##
    rectsungrouped1 = uniquerectsungrouped[::2]
    rectsungrouped2 = uniquerectsungrouped[1::2]

    for i in range(len(rectsungrouped1)):
        rectsungrouped2[i].append(rectsungrouped1[i])

    ## Apply connectivity - if 27, [89, 102, 104] and 89, [25, 37] are connected, then we merge all these into a single list ##
    ## Final result will be [27, 89, 102, 104, 25, 37] ##
    rectsungrouped2=[set(x) for x in rectsungrouped2]
    for a,b in product(rectsungrouped2, rectsungrouped2):
        if a.intersection(b):
            a.update(b)
            b.update(a)
    rectsungrouped2 = sorted( [sorted(list(x)) for x in rectsungrouped2])
    cluster1 = list(rectsungrouped2 for rectsungrouped2,_ in groupby(rectsungrouped2))

    splitlist = []
    for i in range(len(cluster1)):
        splitlist.append(len(cluster1[i]))

    ## Remove ALL rectangles that needs to be merged ##
    cluster1toremove = []
    for i in range(len(cluster1)):
        for x in cluster1[i]:
            cluster1toremove.append(x)

    ## Translate the rectangle number into rectangle coordinates ##
    mergingcoords = []
    for i in range(len(cluster1)):
        for x in cluster1[i]:
            mergingcoords.append((rects[x]))

    ## Group the rectangle coordinates by their cluster ## 
    it = iter(mergingcoords)
    allslicedcoords =[list(islice(it, 0, i)) for i in splitlist]

    allslicedx = []
    allslicedy = []
    allslicedw = []
    allslicedh = []
    allsliceda = []

    mergedacoords_pre = []
    mergedacoords_pre_index = []
    mergedxcoords = []
    mergedycoords = []
    mergedwcoords = []
    mergedhcoords = []
    mergedrectscoords = []
    for i in range(len(allslicedcoords)):
        for x in range(len(allslicedcoords[i])):
            allslicedx.append(allslicedcoords[i][x][0])
            allslicedy.append((allslicedcoords[i][x][1]))
            allslicedw.append((allslicedcoords[i][x][2]))
            allslicedh.append((allslicedcoords[i][x][3]))
            allsliceda.append((allslicedcoords[i][x][2]*allslicedcoords[i][x][3]))
                
    itx = iter(allslicedx)
    ity = iter(allslicedy)
    itw = iter(allslicedw)
    ith = iter(allslicedh)
    ita = iter(allsliceda)
    slicedcoordsx =[list(islice(itx, 0, i)) for i in splitlist]
    slicedcoordsy =[list(islice(ity, 0, i)) for i in splitlist]
    slicedcoordsw =[list(islice(itw, 0, i)) for i in splitlist]
    slicedcoordsh =[list(islice(ith, 0, i)) for i in splitlist]
    slicedcoordsa =[list(islice(ita, 0, i)) for i in splitlist]

    ## Keep only the rectangles with the highest area in their cluster ##
    for i in range(len(slicedcoordsa)):
            mergedacoords_pre.append((max(slicedcoordsa[i])))
            mergedacoords_pre_index.append((slicedcoordsa[i].index(max(slicedcoordsa[i]))))

    flattenedalist = [item for sublist in slicedcoordsa for item in sublist]

    indexlist = []

    def duplicates(lst, item):
        return [i for i, x in enumerate(lst) if x == item]

    for i in range(len(mergedacoords_pre)):
        indexlist.append(max(duplicates(flattenedalist, mergedacoords_pre[i])))

    for i in indexlist:
        mergedxcoords.append(allslicedx[i])
        mergedycoords.append(allslicedy[i])
        mergedwcoords.append(allslicedw[i])
        mergedhcoords.append(allslicedh[i])

    mergedrectscoords = []
    for i in range(len(mergedxcoords)):
        mergedrectscoords.append((mergedxcoords[i],mergedycoords[i],mergedwcoords[i],mergedhcoords[i]))

    cluster1toremove = sorted(cluster1toremove, reverse = True)

    for i in cluster1toremove:
        del rects[i]

    rects.extend(mergedrectscoords)
    print("Number of rects (post-merge): " + str(len(rects)))
        
    return rects

def template_matching(templaterect, image):

    imagecopy = image.copy()
    gray = cv2.cvtColor(imagecopy, cv2.COLOR_BGR2GRAY)
    imgtemplate = gray[int(templaterect[1]):int(templaterect[1])+int(templaterect[3]), int(templaterect[0]):int(templaterect[0])+int(templaterect[2])]

    imgtemplateblur = cv2.GaussianBlur(imgtemplate,(3,3),3)
    ht, wt = imgtemplate.shape
    
    result = match_template(gray, imgtemplate)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    peaks = peak_local_max(result,min_distance=15,threshold_rel=0.3)

    templatedrects = []
    numofrects = []
    rectcount = 0
    for circle in peaks:
        y,x = circle
        templatedrects.append((x,y,wt,ht))
        numofrects.append(rectcount)
        rectcount+=1

    templatedrects_removedoverlap = get_overlap(templatedrects)

    ## IMPORTANT STEP - noise removal for those at the border##
    postvarrects = []
    for rect in templatedrects_removedoverlap:
        im = gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        if np.var(im) > 100:
            postvarrects.append(rect)

    rects_template, numofrects_template = draw_rectangles_biginsects(postvarrects, image)

    save_coordinates_to_xml(filename,numofrects_template,rects_template)
    

def save_coordinates_to_xml(filename,numofrects,rectscoords):
    '''
    Save the coordinates in an .xml file
    The .xml file can be found in the same directory where the image is selected
    '''

    xcoords = []
    ycoords = []
    widthcoords = []
    heightcoords = []

    for i in range(len(rectscoords)):
            xcoords.append(int((rectscoords[i][0]/4000)*800))
            ycoords.append(int((rectscoords[i][1]/4000)*800))
            widthcoords.append(int(((rectscoords[i][2])/4000)*800))
            heightcoords.append(int(((rectscoords[i][3])/4000)*800))

    xmlfile = re.sub(".jpg","Inselect_TemplateMatching.xml",filename)
    root = xml.Element("Information")
    imagenameelement = xml.Element("ImageName")
    imagenamesubelement = xml.SubElement(imagenameelement, "imagenamesubelement")
    imagenamesubelement.text = str(filename)
    root.append(imagenameelement)

    for i in range(len(xcoords)):
        rectangleinformation = xml.Element("RectangleInformation")
        root.append(rectangleinformation)

        rectnumber = xml.SubElement(rectangleinformation, "rectnumber")
        rectnumber.text = str(numofrects[i])

        xcoordsubelement = xml.SubElement(rectangleinformation, "X-coord")
        xcoordsubelement.text = str(xcoords[i])

        ycoordsubelement = xml.SubElement(rectangleinformation, "Y-coord")
        ycoordsubelement.text = str(ycoords[i])

        widthsubelement = xml.SubElement(rectangleinformation, "Width")
        widthsubelement.text = str(widthcoords[i])

        heightsubelement = xml.SubElement(rectangleinformation, "Height")
        heightsubelement.text = str(heightcoords[i])

    tree = xml.ElementTree(root)
    with open(xmlfile, "wb") as fh:
        tree.write(fh)
        
# MAIN FUNCTION
if __name__ == "__main__":
    # Read the image
    roottk = tkinter.Tk()
    roottk.withdraw()
    filename =  filedialog.askopenfilename(initialdir = "/C:", title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

    image = cv2.imread(filename)
    image = cv2.resize(image, (4000,4000))
    whitenedbordersimgsub = whitened_borders(image)

    rects, arealist, display = segment_edges(whitenedbordersimgsub,
                                   resize=(4000, 4000),
                                   variance_threshold=150)

    #outputimagex = image.copy()
    #rectcount = 0
    #numofrects = []
    #for rect in rects:
    #    (x, y, w, h) = rect
    #    cv2.rectangle(outputimagex, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 5);
    #    cv2.putText(outputimagex, str(rectcount), (int(x), int((y)-10)), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0,0,255), 32)
    #    rectcount+=1
        

    #outputimager = cv2.resize(outputimagex, (800,800))
    #cv2.imshow('Overlapped',outputimager)

    determine_insect_size(rects, arealist, image)

