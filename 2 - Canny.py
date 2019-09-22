'''
RUN ON PYTHON 3.6
Algorithm to detect the amount of insects in an image
'''
import numpy as np
import cv2
import os
import tkinter
import re
from tkinter import filedialog
from PIL import Image, ImageChops
import xml.etree.ElementTree as xml
from collections import Counter
import statistics
import math
import time
import itertools
from itertools import groupby, product, islice
from operator import itemgetter

## Open the directory to get the file ##
def load_inselect_file():
    global filename
    roottk = tkinter.Tk()
    roottk.withdraw()
    filename =  filedialog.askopenfilename(initialdir = "/C:", title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    return filename

class algorithm_edge:
    def __init__(self, filename):
        self.filename = filename

    ## Load the image and re-scale it ##
    def rescale_image(filename, width, height):
        oriimg = cv2.imread(filename)
        scaledimg = cv2.resize(oriimg,(width,height))
        return scaledimg

    ## Perform border whitenening ##
    def white_out_borders(scaledimg):
        ## PART 1 ##
        ## Declare empty lists ##
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
        
        scaledimg_borders = scaledimg.copy()
        ## Get the red channel as we have a priori knowledge that the border will be reddish-brown wood ##
        scaledimg_borders_redchannel = scaledimg_borders[:, :, 2]
        rows, cols = scaledimg_borders_redchannel.shape
        ## Convert the image to an array ##
        image_data = np.asarray(scaledimg_borders_redchannel)
        ## Save each pixel value ##
        rpxvalue = []
        for i in range(rows):
            for j in range(cols):
                rpxvalue.append((image_data[i, j]))

        rimage_aslist = list(rpxvalue[i:i+cols] for i in range(0, len(rpxvalue), cols))
        rimage_asarray = np.asarray(rimage_aslist)

        ## Get vertical standard deviation ##
        rgbsta0 = np.std(rimage_asarray, axis = 0)
        ## Get horizontal standard deviation ##
        rgbsta1 = np.std(rimage_asarray, axis = 1)

        ## Set the threshold of 20 ##
        sdthreshold = 20

        ## Get the pixel index where the value is below the threshold value - this gives us the row and column at which the border ends ## 
        for i in range(len(rgbsta0)):
            if rgbsta0[i] < sdthreshold:
                axis0limit.append(rgbsta0[i])
                axis0limitindexnp.append(np.where(rgbsta0 == rgbsta0[i]))

        for i in range(len(axis0limitindexnp)):
            axis0limitindexflat.append(axis0limitindexnp[i][0][0])

        for i in range(len(rgbsta1)):
            if rgbsta1[i] < sdthreshold:
                axis1limit.append(rgbsta1[i])
                axis1limitindexnp.append(np.where(rgbsta1 == rgbsta1[i]))

        for i in range(len(axis1limitindexnp)):
            axis1limitindexflat.append(axis1limitindexnp[i][0][0])

        ## Disqualify any points in the center of the image as we are only interested in the points at the extreme ends ##
        splitlisthresholdrows = rows/4
        splitlisthresholdcols = cols/4

        ## Get the top, bottom, left and right coordinates where the border ends ##
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

        ### PART 2 ###
        ## Declare empty lists ##
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
        
        scaledimg_borders_2 = scaledimg.copy()
        mediansmoothedimg_borders = cv2.medianBlur(scaledimg_borders_2, 3)
        grayimg_borders = cv2.cvtColor(mediansmoothedimg_borders, cv2.COLOR_BGR2GRAY)

        ## Apply the Sobel filter to get the border edges ##
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

        ## Perform dilation to strengthten the lines ##
        kerneldilate = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(mask, kerneldilate, 10)
        

        #dstr = cv2.resize(dilation, (800,800))
        #cv2.imshow('dst',dstr)

        ## It is difficult to extract the precise line coordinates from the Sobel filter, so we get it by pixel level instead ## 
        ## Use Corner Harris detection to get the pixel-level coordinates of the border ##
        dst = cv2.cornerHarris(dilation,2,3,0.001)
        ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(dst,np.float32(centroids),(5,5),(-1,-1),criteria)

        ## Set the limits to prevent overfitting and cutting out important data ##
        upperrowlimit = int(rows * (5/100))
        lowerrowlimit = int(rows * (95/100))
        leftcollimit = int(cols * (5/100))
        rightcollimit = int(cols * (95/100))
        heightpadding = int(rows) - 100
        widthpadding = int(cols) - 100

        ## Get all the pixels that fall within the upper and lower limit and get the furthest point ##
        for i in range(len(corners)):
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

         ## Get all the pixels that fall within the upper and lower limit and get the lowest point ##
        for i in range(len(corners)):
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

        ## Get all the pixels that fall within the upper and lower limit and get the average ##
        for i in range(len(corners)):
            if ((corners[i][0] <= leftcollimit) & (corners[i][0] >= 10)):
                cornersleftlist.append((int(corners[i][0]), int(corners[i][1])))
                cornersleftlistx.append((int(corners[i][0])))
                cornersleftlisty.append((int(corners[i][1])))
        try:
            meancornersleftlist = int(statistics.mean(cornersleftlistx))
        except statistics.StatisticsError:
            meancornersleftlist = 0.02*cols
        except ValueError:
            meancornersleftlist = 0.02*cols

        ## Get all the pixels that fall within the upper and lower limit and get the average ##
        for i in range(len(corners)):
            if ((corners[i][0] >= rightcollimit) & (corners[i][0] <= widthpadding)):
                cornersrightlist.append((int(corners[i][0]), int(corners[i][1])))
                cornersrightlistx.append((int(corners[i][0])))
                cornersrightlisty.append((int(corners[i][1])))
        try:
            meancornersrightlist = int(statistics.mean(cornersrightlistx))
        except statistics.StatisticsError:
            meancornersrightlist = 0.98*cols
        except ValueError:
            meancornersrightlist = 0.98*cols

        ## Get the coordinates that has the least error, or that ensures that it does not intersect with the tray area ##
        topborder = int(max(toplimit,maxcornerstoplist))
        bottomborder = int(min(bottomlimit, mincornersbottomlist))
        leftborder = int(max(leftlimit, meancornersleftlist))
        rightborder = int(min(rightlimit, meancornersrightlist))

        ## Remove the border by setting all the pixels to white and that no bounding boxes will be generated at that location ## 
        whitenedbordersimg = scaledimg.copy()
        topblank = 255 * np.ones(shape=[topborder, cols, 3], dtype = np.uint8)
        bottomblank = 255 * np.ones(shape=[rows-bottomborder, cols, 3], dtype = np.uint8)
        leftblank = 255 * np.ones(shape=[rows, leftborder, 3] , dtype = np.uint8)
        rightblank = 255 * np.ones(shape=[rows, cols-rightborder, 3] , dtype = np.uint8)

        whitenedbordersimg[0:topborder, 0:cols] = topblank
        whitenedbordersimg[bottomborder:rows] = bottomblank
        whitenedbordersimg[0:cols, 0:leftborder] = leftblank
        whitenedbordersimg[0:rows, rightborder:cols] = rightblank

        whitenedbordersimgsub = cv2.cvtColor(whitenedbordersimg, cv2.COLOR_BGR2GRAY)
        whitenedbordersimgsub = cv2.subtract(whitenedbordersimgsub, dilation)
        whitenedbordersimgsub = cv2.cvtColor(whitenedbordersimgsub,cv2.COLOR_GRAY2BGR)

        #whitenedbordersimgsubr = cv2.resize(whitenedbordersimgsub, (800,800))
        #cv2.imshow('dst',whitenedbordersimgsubr)
        
        return whitenedbordersimg, whitenedbordersimgsub

    ## Feature detection by Canny operator ##
    def canny_filter(whitenedbordersimg):

        ## Set the threshold too low and it becomes extremely noisy for big insects##
        ## Set the threshold too high and you get a poor detection ##
        ## dst = cv2.Canny(smoothedimgbilat, 10, 100) ##
        ## dst = cv2.Canny(smoothedimgbilat, 200, 255) ##
        
        gray = cv2.cvtColor(whitenedbordersimg,cv2.COLOR_BGR2GRAY)
        smoothedimgbilat = cv2.bilateralFilter(gray,9,75,75)
        
        dst = cv2.Canny(smoothedimgbilat, 50, 150)
        
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(dst,kernel,iterations = 3)

        #fig10 = cv2.resize(dilation, (800,800))
        #cv2.imshow('Grad Canny', fig10)

        grad = cv2.bitwise_not(dilation)

        ret2,th2 = cv2.threshold(grad,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        masked_out = cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR)
        masked_canny = cv2.resize(th2, (800,800))
        cv2.imshow('Masked Canny', masked_canny)

        return th2

    def find_contours(thresh):
        '''
        Find the contours in an image, returns only returns the bounding rectangles that meets the size limit
        '''
        heightslist = []
        widthslist = []
        rects = []
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key = cv2.contourArea, reverse = True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            heightslist.append(h)
            widthslist.append(w)
            # Remove the overlapping big box
            if (h > 0.25*thresh.shape[0]) | (w > 0.25*thresh.shape[1]):
                pass
            elif (h > 5*w) | (w > 5*h):
                pass
            elif (h*w < thresh.shape[0]*thresh.shape[1]/52000):
                pass
            ## Set the threshold to allow only rectangles of certain size to be appended onto the image ##
            elif h >= 3:
                rect = (x, y, w, h)
                rects.append(rect)

        return rects

    def determine_insect_size(rects, display):
        from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
        from sklearn.model_selection import train_test_split # Import train_test_split function
        from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
        from sklearn.tree.export import export_text
        import pandas as pd

        numofrects = []
        rectcount = 0
        for i in range(len(rects)):
            numofrects.append(rectcount)
            rectcount+=1

        arealist = []
        for rect in rects:
            arealist.append(rect[2]*rect[3])

        ### Load csv ###
        col_names = ['No. of Rects', 'Max Area', 'Area Range', 'Standard Deviation', 'CLASSIFICATION']
        insectcsv = pd.read_csv("EdgeDetection_InsectSize_DT_Train.csv", header=None, names=col_names)
        feature_cols = ['No. of Rects', 'Max Area', 'Area Range', 'Standard Deviation']
        X = insectcsv[feature_cols]
        y = insectcsv.CLASSIFICATION
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=1) #split into 0.9 train and 0.3 test
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
        #y_pred = clf.predict(testing_data)
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


        #if y_pred == 1:
        #    print("DT Prediction: Big Insects")
        #else:
        #    print("DT Prediction: Small Insects")

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
            removedoverlappedrects_big = algorithm_edge.get_overlap(rects)
            bigrects_analysed = algorithm_edge.analyse_areas_biginsects(removedoverlappedrects_big)
            rects_bi, numofrects_bi = algorithm_edge.draw_rectangles_biginsects(bigrects_analysed, scaledimg)
            algorithm_edge.save_coordinates_to_xml(filename,numofrects_bi,rects_bi, display.shape[0])
        else:
            print("Processing Small Insects")
            bigrectsremoved = algorithm_edge.analyse_areas_smallinsects(rects)
            mergedrects = algorithm_edge.merge_rectangles(bigrectsremoved)
            removedoverlappedrects_small = algorithm_edge.get_overlap(mergedrects)
            rects_si, numofrects_si = algorithm_edge.draw_rectangles_smallinsects(removedoverlappedrects_small, scaledimg)
            algorithm_edge.save_coordinates_to_xml(filename,numofrects_si,rects_si, display.shape[1])

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
                        # If the overlapping area is more than 20% #
                        if (Area >= rectoneArea * (20/100) ) | (Area >= recttwoArea * (20/100)):
                            overlaplist.append([numofrects[rectone], numofrects[recttwo]])
                            #print(str(numofrects[rectone]) + " and " +  str(numofrects[recttwo]) + " overlaps. The area of overlap is " + str(Area) + " px.")

        # Get all the overlapped rectangles - optional #
        for i in range(len(overlaplist)):
            overlappedrects.append(overlaplist[i][1])

        overlappedrects = list(set(overlappedrects))

        # Sort the list internally #
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

    def analyse_areas_biginsects(rects):
        arealist = []
        smallrectstoremove = []
        rectnpvar = []
        rectnpindex = []

        for rect in rects:
            imroi = whitenedbordersimg[int(rect[1]):int(rect[1])+int(rect[3]), int(rect[0]):int(rect[0])+int(rect[2])]
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

        return rects

    def analyse_areas_smallinsects(rects):
        arealist = []
        bigrectstoremove = []
        rectnpvar = []
        rectnpindex = []

        print("Pre-analyse small insects: " + str(len(rects)))

        for rect in rects:
            imroi = whitenedbordersimg[int(rect[1]):int(rect[1])+int(rect[3]), int(rect[0]):int(rect[0])+int(rect[2])]
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

        print("Post-analyse small insects: " + str(len(rects)))

        return rects

    def draw_rectangles_biginsects(mostsimilar, scaledimg):
        global rects_bi, numofrects_bi
    
        numofrects_bi = []
        rectcountbi = 0
        rects_bi = rects[:]

        outputimage = scaledimg.copy()
        for rect in rects_bi:
            numofrects_bi.append(rectcountbi)
            (x, y, w, h) = rect
            cv2.rectangle(outputimage, ((x), (y)), ((x+w), (y+h)), (0,255,0), 5);
            #cv2.putText(outputimage, str(rectcountbi), ((x), (y)-10), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0,0,255), 32)
            rectcountbi += 1

        outputimager = cv2.resize(outputimage, (800,800))
        cv2.imshow('Final Output',outputimager)

        return rects_bi, numofrects_bi

    def draw_rectangles_smallinsects(rects, scaledimg):
        global rects_si, numofrects_si

        numofrects_si = []
        rectcountsi = 0
        rects_si = rects[:]

        outputimage = scaledimg.copy()

        for rect in rects_si:
            numofrects_si.append(rectcountsi)
            (x, y, w, h) = rect
            cv2.rectangle(outputimage, (x, y), (x+w, y+h), (0,255,0), 5);
            #cv2.putText(outputimage, str(rectcountsi), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            rectcountsi += 1

        outputimager = cv2.resize(outputimage, (800,800))
        cv2.imshow('Final output', outputimager)
        
        return rects_si, numofrects_si

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

    def save_coordinates_to_xml(filename,numofrects,rectscoords, dimensions):
        '''
        Save the coordinates in an .xml file
        The .xml file can be found in the same directory where the image is selected
        '''

        xcoords = []
        ycoords = []
        widthcoords = []
        heightcoords = []

        for i in range(len(rectscoords)):
            xcoords.append(int((rectscoords[i][0]/dimensions)*800))
            ycoords.append(int((rectscoords[i][1]/dimensions)*800))
            widthcoords.append(int(((rectscoords[i][2])/dimensions)*800))
            heightcoords.append(int(((rectscoords[i][3])/dimensions)*800))

        xmlfile = re.sub(".jpg","Canny_4000x4000.xml",filename)
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

if __name__ == '__main__':
    load_inselect_file()
    algorithm_edge(filename)
    scaledimg = algorithm_edge.rescale_image(filename,4000,4000)
    print("Processing image at: " + "Image width " + str(scaledimg.shape[0]) + " Image height " + str(scaledimg.shape[1]))
    whitenedbordersimg, whitenedbordersimgsub = algorithm_edge.white_out_borders(scaledimg)
    dstcanny = algorithm_edge.canny_filter(whitenedbordersimg)
    
    rects = algorithm_edge.find_contours(dstcanny)
    #algorithm_edge.draw_rectangles(scaledimg, rects)
    algorithm_edge.determine_insect_size(rects, dstcanny)









