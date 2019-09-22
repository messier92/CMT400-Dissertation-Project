'''
RUN ON PYTHON 3.6 
Algorithm to detect the amount of insects in an image using FAST 
'''
import numpy as np
import cv2
import os
import tkinter
import re
from tkinter import filedialog
from PIL import Image, ImageChops, ImageFilter
from matplotlib import pyplot as plt
import xml.etree.ElementTree as xml
from collections import Counter
import statistics
import itertools
import math
import sklearn
import pandas as pd
from itertools import accumulate, islice, product, groupby
from itertools import cycle
from operator import itemgetter
import skimage
from skimage import measure

## Declare empty lists ##
numofrects2 = []
alliou = []
overlaplist = []
rects = []
heightslist = []
widthslist = []
numofrects = []
rectsTL = []
rectsBR = []
fulloverlaplistpairs = []
rectanglestokeep = []

rectanglestocompare = []
rectangleareas = []
sortedoverlaplist = []
uniqueoverlaplist = []
rectanglenumberstoremove = []
remainingrects = []
        
xcoordsremove = []
ycoordsremove = []
widthcoordsremove = []
heightcoordsremove = []
rectanglecoordsremove = []

def load_inselect_file():
    global filename
    roottk = tkinter.Tk()
    roottk.withdraw()
    filename =  filedialog.askopenfilename(initialdir = "/C:", title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    return filename

class algorithm_corner:
    def __init__(self, filename):
        self.filename = filename

    def rescale_image(filename, width, height):
        oriimg = cv2.imread(filename)
        scaledimg = cv2.resize(oriimg, (width,height))
        return scaledimg

    def white_out_borders(scaledimg):
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
        scaledimg_borders_redchannel = scaledimg_borders[:, :, 2]

        rows, cols = scaledimg_borders_redchannel.shape
        image_data = np.asarray(scaledimg_borders_redchannel)

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

        sdthreshold = 20
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

        ### PART 2 ###
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
        
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(dst,np.float32(centroids),(5,5),(-1,-1),criteria)

        upperrowlimit = int(rows * (5/100))
        lowerrowlimit = int(rows * (95/100))
        leftcollimit = int(cols * (5/100))
        rightcollimit = int(cols * (95/100))
        heightpadding = int(rows) - 100
        widthpadding = int(cols) - 100

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
            mincornersbottomlist = 0.02*rows

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

        topborder = int(max(toplimit,maxcornerstoplist))
        bottomborder = int(min(bottomlimit, mincornersbottomlist))
        leftborder = int(max(leftlimit, meancornersleftlist))
        rightborder = int(min(rightlimit, meancornersrightlist))
        whitenedbordersimg = scaledimg.copy()

        topblank = 255 * np.ones(shape=[topborder, cols, 3], dtype = np.uint8)
        bottomblank = 255 * np.ones(shape=[rows-bottomborder, cols, 3], dtype = np.uint8)
        leftblank = 255 * np.ones(shape=[rows, leftborder, 3] , dtype = np.uint8)
        rightblank = 255 * np.ones(shape=[rows, cols-rightborder, 3] , dtype = np.uint8)

        whitenedbordersimg[0:topborder, 0:cols] = topblank
        whitenedbordersimg[bottomborder:rows] = bottomblank
        whitenedbordersimg[0:cols, 0:leftborder] = leftblank
        whitenedbordersimg[0:rows, rightborder:cols] = rightblank
        
        #whitenedbordersimgr = cv2.resize(whitenedbordersimg, (800,800))
        #cv2.imshow('dst',whitenedbordersimgr)

        return whitenedbordersimg

    def FAST_detector(whitenedbordersimg):
        rects = []
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(whitenedbordersimg,None)

        FASTImage = cv2.drawKeypoints(whitenedbordersimg, kp, None, color=(255,255,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #FASTImager = cv2.resize(FASTImage, (800,800))
        #cv2.imshow('FAST kp', FASTImager)
        FASTImage_gray = cv2.cvtColor(FASTImage, cv2.COLOR_BGR2GRAY)

        pts = [p.pt for p in kp]
        pts_list = [list(elem) for elem in pts]

        #whitenedbordersimg_gray = cv2.cvtColor(whitenedbordersimg, cv2.COLOR_BGR2GRAY)

        blankimg = np.zeros((4000,4000,3))
        
        for i in range(len(pts_list)):
            blankimg[int(pts_list[i][1]), int(pts_list[i][0])] = (255,255,255)

        #blankimgr = cv2.resize(blankimg, (800,800))
        #cv2.imshow('Initial', blankimgr)

        kernel = np.ones((5,5),np.uint8)
        kernel_opening = np.ones((30, 30), np.uint8)
        kernel_erode = np.ones((2,2), np.uint8)
        dilation = cv2.dilate(blankimg,kernel,iterations = 3)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_opening)
        erosion = cv2.erode(opening,kernel_erode,iterations = 2)
        erosion = cv2.cvtColor(erosion.astype('uint8'),cv2.COLOR_BGR2GRAY)

        erosionr = cv2.resize(erosion, (600,600))
        cv2.imshow('Final mask', erosionr)
        
        contours, _ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if (h > 0.25*FASTImage_gray.shape[0]) | (w > 0.25*FASTImage_gray.shape[1]):
                pass
            elif (h > 6*w) | (w > 6*h):
                pass
            elif h >= 5:
                rect = (x,y,w,h)
                rects.append(rect)

        return rects, FASTImage_gray

    def Harris_Corner_detector(whitenedbordersimg):
        rects = []
        gray = cv2.cvtColor(whitenedbordersimg,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,17,17,0.02)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        whitenedbordersimg[dst>0.01*dst.max()]=[0,0,255]
        whitenedbordersimg_gray = cv2.cvtColor(whitenedbordersimg, cv2.COLOR_BGR2GRAY)

        blankimg = np.zeros((4000,4000,3))
        for i in range(len(corners)):
            blankimg[int(corners[i][1]), int(corners[i][0])] = (255,255,255)

        #predilate = cv2.resize(blankimg, (800,800))
        #cv2.imshow('pre-dilate', predilate)
        
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(blankimg,kernel,iterations = 15)

        dilation = cv2.cvtColor(dilation.astype('uint8'),cv2.COLOR_BGR2GRAY)

        maskedimg = cv2.subtract(dilation, whitenedbordersimg_gray)

        masked = cv2.resize(dilation, (600,600))
        cv2.imshow('Masked Image - HARRIS', masked)
        
        contours, _ = cv2.findContours(maskedimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if (h > 0.25*whitenedbordersimg.shape[0]) | (w > 0.25*whitenedbordersimg.shape[1]):
                pass
            elif (h > 6*w) | (w > 6*h):
                pass
            elif h >= 5:
                rect = (x,y,w,h)
                rects.append(rect)

        return rects, whitenedbordersimg_gray

    def draw_rectangles(scaledimg, rects):
        numofrects = []
        arealist = []
        
        outputimage = scaledimg.copy()
        rectcount = 0
        for rect in rects:
            numofrects.append(rectcount)
            (x, y, w, h) = rect
            outputimage = cv2.rectangle(outputimage, (x, y), (x+w, y+h), (0,255,0), 5);
            rectcount += 1
            arealist.append((w*h))

        ptsimage = cv2.resize(outputimage, (600,600))
        cv2.imshow('Final output', ptsimage)

        return outputimage, numofrects, arealist

    def get_overlap(rects):
        import itertools
        global overlaplist

        numofrects = []
        
        rectcount = 0
        for rect in rects:
            numofrects.append(rectcount)
            rectcount += 1
            
        rectsTL = []
        rectsBR = []
        alliou = []
        fulloverlaplistpairs = []
        rectanglestokeep = []
        overlaplist = []
        overlappedrects = []
        rectanglestocompare = []
        rectangleareas = []
        sortedoverlaplist = []
        uniqueoverlaplist = []
        rectanglestoremove = []
        remainingrects = []
        resultslist = []
        newrectscoords = []
        newrects = rects[:]
        
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

        for i in range(len(overlaplist)):
            overlappedrects.append(overlaplist[i][1])

        overlappedrects = list(set(overlappedrects))

        for i in range(len(overlaplist)):
            rectanglestocompare.append(overlaplist[i][0])

        rectanglestocompareset = list(set(rectanglestocompare))

        for i in rectanglestocompareset:
            rectangleareas.append(rects[i][2]*rects[i][3])

        for i in range(len(overlaplist)):
            sortedoverlaplist.append(sorted(overlaplist[i]))

        sortedoverlaplist.sort()
        uniqueoverlaplist = (list(sortedoverlaplist for sortedoverlaplist,_ in itertools.groupby(sortedoverlaplist)))

        for i in range(len(uniqueoverlaplist)):
            rectanglestoremove.append(uniqueoverlaplist[i][1])

        rectanglestoremove = list(set(rectanglestoremove))
        rectanglestoremove = sorted(rectanglestoremove, reverse = True)
            
        for i in rectanglestoremove:
            del newrects[i]

        return overlaplist, overlappedrects, newrects

    def merge_rectangles(rects):
        import itertools
        from itertools import groupby, product, islice
        from operator import itemgetter

        #algorithm_edge.draw_rectangles(whitenedbordersimg, rects, 'Pre-Merge')
        print("Number of rects (pre-merged): " + str(len(rects)))
        
        centroidlistx = []
        centroidlisty = []
        arealist = []

        numofrects = []
        rectstomerge_withduplicates = []
        sortedrectstomerge = []
        uniquerectsungrouped = []
        rectcount = 0
        for rect in rects:
            numofrects.append(rectcount)
            rectcount += 1

        ## Get the centroids of all the rectangles by adding half of the width and height to the x and y coordinates respectively ##
        for (x,y,w,h) in rects:
            centroidlistx.append(int(x+0.5*w))
            centroidlisty.append(int(y+0.5*h))
            arealist.append(w*h)

        areaten = int(len(arealist)*0.1)
        areafourty = int(len(arealist)*0.4)

        minarealist = statistics.mean(arealist[areaten:areafourty])
        mergethresholdalt = int(minarealist/300)
        print("Merging threshold (alternate): " + str(mergethresholdalt))
        #mergethreshold = mergethresholdalt
        mergethreshold = 10000/len(numofrects)
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
        rectsungrouped2 = sorted([sorted(list(x)) for x in rectsungrouped2])
        
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

        #print(mergingcoords)

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

        #algorithm_edge.draw_rectangles(whitenedbordersimg, rects, 'Post-Merge')
        
        return rects

    def analyse_areas(rects, whitenedbordersimg):

        rectnpvar = []
        rectnpindex = []
        print(len(rects))
        for rect in rects:
            imroi = whitenedbordersimg[int(rect[1]):int(rect[1])+int(rect[3]), int(rect[0]):int(rect[0])+int(rect[2])]
            if np.var(imroi) < 100:
                rectnpvar.append(rect)

        for i in rectnpvar:
            rectnpindex.append(rects.index(i))

        rectnpindex = sorted(rectnpindex, reverse=True)
        for i in rectnpindex:
            del rects[i]

        return rects

    def get_overlap(rects):
        import itertools
        numofrects = []
        
        rectcount = 0
        for rect in rects:
            numofrects.append(rectcount)
            rectcount += 1
            
        rectsTL = []
        rectsBR = []
        alliou = []
        fulloverlaplistpairs = []
        rectanglestokeep = []
        overlaplist = []
        overlappedrects = []
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

        for i in range(len(overlaplist)):
            overlappedrects.append(overlaplist[i][1])

        overlappedrects = list(set(overlappedrects))

        return overlaplist, overlappedrects

    def remove_overlapping_rectangles(overlaplist, rects):

        rectsTL = []
        rectsBR = []
        rectanglestocompare = []
        rectangleareas = []
        sortedoverlaplist = []
        uniqueoverlaplist = []
        rectanglestoremove = []
        resultslist = []

        for i in range(len(rects)):
            rectsTL.append((rects[i][0],rects[i][1]))
            rectsBR.append((rects[i][0]+rects[i][2],rects[i][1]+rects[i][3]))

        for i in range(len(overlaplist)):
            rectanglestocompare.append(overlaplist[i][0])

        rectanglestocompareset = list(set(rectanglestocompare))

        for i in rectanglestocompareset:
            rectangleareas.append(rects[i][2]*rects[i][3])

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

        return rects

    def save_coordinates_to_xml(filename, rectscoords):
        '''
        Save the coordinates in an .xml file
        The .xml file can be found in the same directory where the image is selected
        '''

        xcoords = []
        ycoords = []
        widthcoords = []
        heightcoords = []
        numofrects = []

        rectcount = 0
        for rect in rectscoords:
            numofrects.append(rectcount)
            rectcount+=1
            

        for i in range(len(rectscoords)):
            xcoords.append(int((rectscoords[i][0]/4000)*800))
            ycoords.append(int((rectscoords[i][1]/4000)*800))
            widthcoords.append(int(((rectscoords[i][2])/4000)*800))
            heightcoords.append(int(((rectscoords[i][3])/4000)*800))

        xmlfile = re.sub(".jpg","corner_FASTalgorithm.xml",filename)
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
    algorithm_corner(filename)
    scaledimg = algorithm_corner.rescale_image(filename,4000,4000)
    whitenedbordersimg = algorithm_corner.white_out_borders(scaledimg)
    FAST_rects, FASTImage_gray = algorithm_corner.FAST_detector(whitenedbordersimg)
    #Harris_rects, whitenedbordersimg_gray = algorithm_corner.Harris_Corner_detector(whitenedbordersimg)
    merged_rects = algorithm_corner.merge_rectangles(FAST_rects)
    overlaplist, overlappedrects = algorithm_corner.get_overlap(FAST_rects)
    finalrects = algorithm_corner.remove_overlapping_rectangles(overlaplist, FAST_rects)
    algorithm_corner.draw_rectangles(scaledimg, finalrects)
    #algorithm_corner.save_coordinates_to_xml(filename,finalrects)

