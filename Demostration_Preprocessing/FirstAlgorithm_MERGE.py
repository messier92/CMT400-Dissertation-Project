import numpy as np
import cv2
import os
import tkinter
import re
from tkinter import filedialog
from PIL import Image, ImageChops
from matplotlib import pyplot as plt
import xml.etree.ElementTree as xml
from collections import Counter
import statistics
import itertools
from itertools import accumulate, islice, product, groupby
from operator import itemgetter

## Declare empty lists ##

cornersleftlist = []
cornersleftlistx = []
cornersleftlisty = []

cornersrightlist = []
cornersrightlistx = []
cornersrightlisty = []

## Open the directory to get the image ##
def load_image_file():
    global filename
    roottk = tkinter.Tk()
    roottk.withdraw()
    filename =  filedialog.askopenfilename(initialdir = "/C:", title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    return filename

def rescale_image(filename, width, height):
    global oriimg, scaledimg, scaledimginsects
    oriimg = cv2.imread(filename)
    scaledimg = cv2.resize(oriimg,(width,height))
    scaledimginsects = cv2.resize(oriimg,(width,height))
    return oriimg, scaledimg, scaledimginsects

def median_blur(scaledimg, pixelsize):
    global mediansmoothedimg
    mediansmoothedimg = cv2.medianBlur(scaledimg, pixelsize)
    return mediansmoothedimg

def convert_to_grayscale(mediansmoothedimg):
    global grayscaleimg
    grayscaleimg= cv2.cvtColor(mediansmoothedimg, cv2.COLOR_BGR2GRAY)
    return grayscaleimg

def threshold_image_binary(grayscaleimg, minVal, maxVal):
    global ret, thresh
    ret,thresh = cv2.threshold(grayscaleimg,minVal,maxVal,cv2.THRESH_BINARY)
    return ret, thresh

def apply_sobel_filter_borders(grayscaleimg, lw, threshold):
    global mask
    v_edges = cv2.Sobel(grayscaleimg, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    h_edges = cv2.Sobel(grayscaleimg, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(v_edges)
    abs_grad_y = cv2.convertScaleAbs(h_edges)
    mask = np.zeros(grayscaleimg.shape, dtype=np.uint8)
    linewidth = ((grayscaleimg.shape[0] + grayscaleimg.shape[1])) / lw
    ## LOWER THIS THRESHOLD TO INCREASE THE NUMBER OF POINTS - default is 20## 

    ## Detect the vertical edges ##
    magv = np.abs(v_edges)
    magv2 = (255*magv/np.max(magv)).astype(np.uint8)
    _, mag2 = cv2.threshold(magv2, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mag2.copy(),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if h > grayscaleimg.shape[0] / 4 and w < linewidth:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    ## Detect the horizontal edges ##
    magh = np.abs(h_edges)
    magh2 = (255*magh/np.max(magh)).astype(np.uint8)
    _, mag2 = cv2.threshold(magh2, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mag2.copy(),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if w > grayscaleimg.shape[1] / 4 and h < linewidth:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    return mask     

def apply_canny_filter(mask, apertureSize):
    global edges
    edges = cv2.Canny(mask,0,255,apertureSize)
    return edges

def detect_corners_harris(edges, ksize, iterations):
    global addedimg, dst
    kernel = np.ones((ksize, ksize), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations)
    addedimg = cv2.add(dilation, grayscaleimg)
    ret, thresh = cv2.threshold(addedimg, 240, 255, cv2.THRESH_BINARY)
    addedimg[thresh == 255] = 0

    dst = cv2.cornerHarris(edges,2,3,0.001)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    #cv2.imshow('Outline', dst)
    return addedimg, dst

def get_corner_coordinates(addedimg, dst):
    global corners
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(addedimg,np.float32(centroids),(5,5),(-1,-1),criteria)
    return corners

def find_top_limit(scaledimg, corners):
    global meancornerstoplisty
    
    cornerstoplist = []
    cornerstoplistx = []
    cornerstoplisty = []

    upperrowlimit = int(scaledimg.shape[0] * (10/100))
    for i in range(len(corners)):
        ## if the y coordinate is less than 80px ad more than 10px
        if ((corners[i][1] <= upperrowlimit) & (corners[i][1] >= 10)):
            cornerstoplist.append((int(corners[i][0]), int(corners[i][1])))
            cornerstoplistx.append((int(corners[i][0])))
            cornerstoplisty.append((int(corners[i][1])))

    meancornerstoplisty = int(statistics.mean(cornerstoplisty))
    #cv2.line(scaledimg,(scaledimg.shape[1],meancornerstoplisty),(0,meancornerstoplisty),(255,0,0),5)
    return meancornerstoplisty

def find_bottom_limit(scaledimg, corners):
    global meancornersbottomlisty

    cornersbottomlist = []
    cornersbottomlistx = []
    cornersbottomlisty = []

    lowerrowlimit = int(scaledimg.shape[0] * (90/100))
    rowpadding = int(scaledimg.shape[0]) - 10
    for i in range(len(corners)):
        ## If the y coordinate is more than 720px and less than 790px 
        if ((corners[i][1] >= lowerrowlimit) & (corners[i][1] <= rowpadding)):
            cornersbottomlist.append((int(corners[i][0]), int(corners[i][1])))
            cornersbottomlistx.append((int(corners[i][0])))
            cornersbottomlisty.append((int(corners[i][1])))

    meancornersbottomlisty = int(statistics.mean(cornersbottomlisty))
    #cv2.line(scaledimg,(scaledimg.shape[1],meancornersbottomlisty),(0,meancornersbottomlisty),(255,0,0),5)
    return meancornersbottomlisty
    
def find_left_limit(scaledimg, corners):
    global meancornersleftlistx
    leftcollimit = int(scaledimg.shape[1] * (10/100))
    for i in range(len(corners)):
        if ((corners[i][0] <= leftcollimit) & (corners[i][0] >= 10)):
        ## If the x coordinate is less than 40px and more than 10px ##
            cornersleftlist.append((int(corners[i][0]), int(corners[i][1])))
            cornersleftlistx.append((int(corners[i][0])))
            cornersleftlisty.append((int(corners[i][1])))

    meancornersleftlistx = int(statistics.mean(cornersleftlistx))
    #cv2.line(scaledimg,(meancornersleftlistx,scaledimg.shape[0]),(meancornersleftlistx,0),(255,0,0),5)
    return meancornersleftlistx

def find_right_limit(scaledimg, corners):
    global meancornersrightlistx
    rightcollimit = int(scaledimg.shape[1] * (90/100))
    colpadding = int(scaledimg.shape[1]) - 10
    for i in range(len(corners)):
        # If the x coordinate is more than 720px and less than 790px ##
        if ((corners[i][0] >= rightcollimit) & (corners[i][0] <= colpadding)):
            cornersrightlist.append((int(corners[i][0]), int(corners[i][1])))
            cornersrightlistx.append((int(corners[i][0])))
            cornersrightlisty.append((int(corners[i][1])))

    meancornersrightlistx = int(statistics.mean(cornersrightlistx))
    #cv2.line(scaledimg,(meancornersrightlistx,scaledimg.shape[0]),(meancornersrightlistx,0),(255,0,0),5)
    return meancornersrightlistx

def white_out(scaledimginsects, toplimit, bottomlimit, leftlimit, rightlimit):
    global whitenedbordersimg
    whitenedbordersimg = scaledimginsects.copy()

    topblank = 255 * np.ones(shape=[toplimit, scaledimginsects.shape[0], 3], dtype = np.uint8)
    botttomblank = 255 * np.ones(shape=[scaledimginsects.shape[0]-bottomlimit, scaledimginsects.shape[0], 3], dtype = np.uint8)

    leftblank = 255 * np.ones(shape=[scaledimginsects.shape[1], leftlimit, 3] , dtype = np.uint8)
    rightblank = 255 * np.ones(shape=[rightlimit, scaledimginsects.shape[1]-rightlimit,  3] , dtype = np.uint8)
    
    whitenedbordersimg[0:toplimit, 0:scaledimginsects.shape[0]] = topblank
    whitenedbordersimg[bottomlimit:scaledimginsects.shape[0]] = botttomblank
    
    whitenedbordersimg[0:scaledimginsects.shape[1], 0:leftlimit] = leftblank
    checkarea = whitenedbordersimg[0:rightlimit, rightlimit:scaledimginsects.shape[1]] = rightblank

    return whitenedbordersimg

def gaussian_blur(whitenedbordersimg, pixelwidth, pixelheight):
    global gaussiansmoothedimg
    gaussiansmoothedimg = cv2.GaussianBlur(whitenedbordersimg, (pixelwidth, pixelheight), 0)
    return gaussiansmoothedimg

def convert_to_grayscale_insects(gaussiansmoothedimg):
    global grayscaleimginsects
    grayscaleimginsects = cv2.cvtColor(gaussiansmoothedimg, cv2.COLOR_BGR2GRAY)
    return grayscaleimginsects

def apply_sobel_filter_insects(grayscaleimginsects, lowthresh, highthresh):
    global maskedimg
    v_edges = cv2.Sobel(grayscaleimginsects, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    h_edges = cv2.Sobel(grayscaleimginsects, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(v_edges)
    abs_grad_y = cv2.convertScaleAbs(h_edges)
    grad = cv2.addWeighted(abs_grad_x, 0.8, abs_grad_y, 0.8, 0)
    grad = cv2.bitwise_not(grad)

    ret2,th2 = cv2.threshold(grad,lowthresh,highthresh,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    masked_out=cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR) #change mask to a 3 channel image

    maskedimg = cv2.subtract(masked_out,whitenedbordersimg)
    maskedimg = cv2.cvtColor(maskedimg, cv2.COLOR_BGR2GRAY)

    return maskedimg

def threshold_image_binary(maskedimg, minVal, maxVal):
    global ret, thresh
    ret,thresh = cv2.threshold(maskedimg,minVal,maxVal,cv2.THRESH_BINARY)
    return ret, thresh
    
def find_contours(thresh, toplimit, bottomlimit, leftlimit, rightlimit):
    global cnts, rects, heightslist, widthslist
    rects = []
    heightslist = []
    widthslist = []
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        heightslist.append(h)
        widthslist.append(w)
        # Remove the overlapping big box
        if (h > 0.5*thresh.shape[0]) | (w > 0.5*thresh.shape[1]):
            pass
        elif (h > 4*w) | (w > 4*h):
            pass
        elif h >= 5:
            rect = (x,y,w,h)
            rects.append(rect)

    for rect in rects:
        if ((rect[0] < toplimit) | (rect[0] > bottomlimit) | (rect[1] < leftlimit) | (rect[1] > rightlimit)) :
            rects.remove(rect)  

    return cnts, rects, heightslist, widthslist

## See the location of the plotted rectangles ##
def draw_rectangles(scaledimg, rects):
        global outputimage, numofrects
        numofrects = []
        outputimage = scaledimg.copy()
        rectcount = 0

        # Append the rectangles and rectangle number onto the image
        for rect in rects:
            numofrects.append(rectcount)
            (x, y, w, h) = rect
            outputimage = cv2.rectangle(outputimage, (x, y), (x+w, y+h), (0,255,0), 1);
            outputimage = cv2.putText(outputimage, str(rectcount), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            rectcount += 1

        #cv2.imshow('First Iteration', outputimage)
        return outputimage, numofrects

def get_overlap(rects):
    global rectsTL, rectsBR, overlaplist
    numofrects = []
    rectcount = 0
    for rect in rects:
        numofrects.append(rectcount)
        rectcount += 1
            
    rectsTL = []
    rectsBR = []
    alliou = []
    overlaplist = []
    fulloverlaplistpairs = []
    rectanglestokeep = []
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
                    # If the overlapping area is more than 40% #
                    if (Area >= rectoneArea * (40/100) ) | (Area >= recttwoArea * (40/100)):
                        overlaplist.append([numofrects[rectone], numofrects[recttwo]])
                        #print(str(numofrects[rectone]) + " and " +  str(numofrects[recttwo]) + " overlaps. The area of overlap is " + str(Area) + " px.")

    return rectsTL, rectsBR, overlaplist

def remove_overlapping_rectangles(overlaplist, rectsTL, rectsBR):
    global newrects, uniqueoverlaplist, rectanglestoremove, remainingrects

    rectsTL = []
    rectsBR = []
    rectanglestocompare = []
    rectangleareas = []
    sortedoverlaplist = []
    uniqueoverlaplist = []
    rectanglestoremove = []
    remainingrects = []
    newrects = rects[:]
        
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

    for i in rectanglestoremove:
        for (x,y,w,h) in newrects:
            newrects[i] = (0,0,0,0)

    remainingrects = [elem for elem in numofrects if elem not in rectanglestoremove]

    return uniqueoverlaplist, rectanglestoremove, newrects, remainingrects

### CONNECTIVITY - GRAPH THEORY ###
### With reference from - https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements ###  
def get_centroids(remainingrects, newrects):
    global centroidlist, newrectscoords, uniquerectset, uniquerectstomerge, uniquerectsgrouped, mergedrectscoords
    
    newrectscoords = []
    resultslist = []
    centroidlist = []
    centroidlistx = []
    centroidlisty = []
    rectstomerge_duplicate = []
    rectstomerge = []
    sortedrectstomerge = []
    uniquerectstomerge = []
    
    mergingrectonelist = []
    mergingrecttwolist = []
    
    newmergedrectangleslistx = []
    newmergedrectangleslisty = []
    newmergedrectangleslistw = []
    newmergedrectangleslisth = []

    uniquerectsungrouped = []
    rectsungrouped1 = []
    rectsungrouped2 = []

    mergingcoords = []

    newrectsstr = str(newrects)
    newrectsstr = re.sub(", \(0, 0, 0, 0\)","",newrectsstr)
    newrectsstr = re.sub("\)","",newrectsstr)
    newrectsstr = re.sub("\(","",newrectsstr)
    newrectsstr = re.sub("\[","",newrectsstr)
    newrectsstr = re.sub("\]","",newrectsstr)
    
    newrectsstrresult = [x.strip() for x in newrectsstr.split(',')]

    for i in range(len(newrectsstrresult)):
        resultslist.append(newrectsstrresult[i])

    newxcoords = resultslist[::4]
    newxcoords = [int(i) for i in newxcoords]

    newycoords = resultslist[1::4]
    newycoords = [int(i) for i in newycoords]
        
    newwidthcoords = resultslist[2::4]
    newwidthcoords = [int(i) for i in newwidthcoords]
        
    newheightcoords = resultslist[3::4]
    newheightcoords = [int(i) for i in newheightcoords]

    for i in range(len(newheightcoords)):
        newrectscoords.append((newxcoords[i],newycoords[i],newwidthcoords[i],newheightcoords[i]))

    for (x,y,w,h) in newrectscoords:
        centroidlist.append((int(x+0.5*w),int(y+0.5*h)))
        centroidlistx.append(int(x+0.5*w))
        centroidlisty.append(int(y+0.5*h))

    mergethreshold = 5000/len(rects)
    print(mergethreshold)

    for i in range(len(centroidlistx)):
        for x in range(len(centroidlistx)):
            if centroidlistx[x] == centroidlistx[i] & centroidlisty[x] == centroidlisty[i]:
                pass
            # CHANGE THE MERGING THRESHOLD HERE
            elif (centroidlistx[x]-mergethreshold <= centroidlistx[i] <= centroidlistx[x]+mergethreshold) & (centroidlisty[x]-mergethreshold <= centroidlisty[i] <= centroidlisty[x]+mergethreshold):
                rectstomerge_duplicate.append((remainingrects[x], remainingrects[i]))

    for rect in range(len(rectstomerge_duplicate)):
        if rectstomerge_duplicate[rect][0] == rectstomerge_duplicate[rect][1]:
            pass
        else:
            rectstomerge.append((rectstomerge_duplicate[rect][0], rectstomerge_duplicate[rect][1]))

    rectstomerge = sorted(rectstomerge)
    rectstomerge = list(set(rectstomerge))
    for i in range(len(rectstomerge)):
        sortedrectstomerge.append(tuple(sorted(rectstomerge[i])))

    uniquerectstomerge = sorted(list(set(sortedrectstomerge)))

    uniquerectstomergestr = str(uniquerectstomerge)
    uniquerectstomergestr = re.sub("\)","",uniquerectstomergestr)
    uniquerectstomergestr = re.sub("\(","",uniquerectstomergestr)
    uniquerectstomergestr = re.sub("\[","",uniquerectstomergestr)
    uniquerectstomergestr = re.sub("\]","",uniquerectstomergestr)

    uniquerectspresort = [x.strip() for x in uniquerectstomergestr.split(',')]
    uniquerectset = [int(i) for i in uniquerectspresort]

    uniquerectsgrouped = [(k, list(list(zip(*g))[1])) for k, g in groupby(uniquerectstomerge, itemgetter(0))]

    for i in range(len(uniquerectsgrouped)):
        for x in range(len(uniquerectsgrouped[i])):
            uniquerectsungrouped.append((uniquerectsgrouped[i][x]))

    rectsungrouped1 = uniquerectsungrouped[::2]
    rectsungrouped2 = uniquerectsungrouped[1::2]

    for i in range(len(rectsungrouped1)):
        rectsungrouped2[i].append(rectsungrouped1[i])

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

    cluster1str = str(cluster1)
    cluster1str = re.sub("\[", "", cluster1str)
    cluster1str = re.sub("\]", "", cluster1str)
    cluster1strresult = [x.strip() for x in cluster1str.split(',')]
    cluster1toremove = [int(i) for i in cluster1strresult]

    for i in range(len(cluster1)):
        for x in cluster1[i]:
            mergingcoords.append((newrects[x]))
            
    it = iter(mergingcoords)
    allslicedcoords =[list(islice(it, 0, i)) for i in splitlist]

    allslicedx = []
    allslicedy = []
    allslicedw = []
    allslicedh = []
    mergedxcoords = []
    mergedycoords = []
    mergedwcoords = []
    mergedhcoords = []

    for i in range(len(allslicedcoords)):
        for x in range(len(allslicedcoords[i])):
            allslicedx.append((allslicedcoords[i][x][0]))
            allslicedy.append((allslicedcoords[i][x][1]))
            allslicedw.append((allslicedcoords[i][x][2]))
            allslicedh.append((allslicedcoords[i][x][3]))

    itx = iter(allslicedx)
    ity = iter(allslicedy)
    itw = iter(allslicedw)
    ith = iter(allslicedh)
    slicedcoordsx =[list(islice(itx, 0, i)) for i in splitlist]
    slicedcoordsy =[list(islice(ity, 0, i)) for i in splitlist]
    slicedcoordsw =[list(islice(itw, 0, i)) for i in splitlist]
    slicedcoordsh =[list(islice(ith, 0, i)) for i in splitlist]

    for i in range(len(slicedcoordsx)):
	    mergedxcoords.append(int(sum(slicedcoordsx[i])/len(slicedcoordsx[i])))
	    mergedycoords.append(int(sum(slicedcoordsy[i])/len(slicedcoordsy[i])))
	    mergedwcoords.append(int(sum(slicedcoordsw[i])/len(slicedcoordsw[i])))
	    mergedhcoords.append(int(sum(slicedcoordsh[i])/len(slicedcoordsh[i])))

    mergedrectscoords = []
    for i in range(len(mergedxcoords)):
        mergedrectscoords.append((mergedxcoords[i],mergedycoords[i],mergedwcoords[i],mergedhcoords[i]))

    print(mergedrectscoords)
    newrects.extend(mergedrectscoords)
    #uniquerectset = list(set(uniquerectspresort))
    #uniquerectset = [int(i) for i in uniquerectset]
    #uniquerectset = sorted(uniquerectset)
    #print(uniquerectset)

    for i in cluster1toremove:
        for (x,y,w,h) in newrects:
            newrects[i] = (0,0,0,0)

    #for i in range(len(uniquerectstomerge)):
    #    mergingrectonelist.append((rects[uniquerectstomerge[i][0]]))
    #    mergingrecttwolist.append((rects[uniquerectstomerge[i][1]]))

    #for i in range(len(mergingrectonelist)):
    #    newmergedrectangleslistx.append(round(int(mergingrectonelist[i][0]+mergingrecttwolist[i][0])/2))
    #    newmergedrectangleslisty.append(round(int(mergingrectonelist[i][1]+mergingrecttwolist[i][1])/2))
    #    newmergedrectangleslistw.append(round(int(mergingrectonelist[i][2]+mergingrecttwolist[i][2])))
    #    newmergedrectangleslisth.append(round(int(mergingrectonelist[i][3]+mergingrecttwolist[i][3])))

    #rectcount = len(numofrects)
    #for i in range(len(newmergedrectangleslistx)):
    #    newrects.append((newmergedrectangleslistx[i],newmergedrectangleslisty[i], newmergedrectangleslistw[i], newmergedrectangleslisth[i]))
    #    rectcount += 1
    #    numofrects.append(rectcount)

    return centroidlist, newrectscoords, uniquerectset, uniquerectstomerge, uniquerectsgrouped, mergedrectscoords

def draw_new_rectangles(scaledimg, newrects):
    global outputimage2, numofrects2
    numofrects2 = []

    outputimage2 = scaledimg.copy()
    rectcount2 = 0
    for rect in newrects:
        numofrects2.append(rectcount2)
        (x2, y2, w2, h2) = rect
        cv2.rectangle(outputimage2, (x2, y2), (x2+w2, y2+h2), (0,255,0), 1);
        cv2.putText(outputimage2, str(rectcount2), (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
        rectcount2 += 1

    cv2.imshow('newr',outputimage2)
    return outputimage2, numofrects2

load_image_file()
rescale_image(filename,800,800)

median_blur(scaledimg, 3)
convert_to_grayscale(mediansmoothedimg)
apply_sobel_filter_borders(grayscaleimg, 50, 15)
apply_canny_filter(mask, 5)
detect_corners_harris(edges, 5, 10)
get_corner_coordinates(addedimg, dst)

find_top_limit(scaledimg, corners)
find_bottom_limit(scaledimg, corners)
find_left_limit(scaledimg, corners)
find_right_limit(scaledimg, corners)
white_out(scaledimg, meancornerstoplisty, meancornersbottomlisty, meancornersleftlistx, meancornersrightlistx)

gaussian_blur(whitenedbordersimg, 5, 5)
convert_to_grayscale_insects(gaussiansmoothedimg)
apply_sobel_filter_insects(grayscaleimginsects, 150, 255)
threshold_image_binary(maskedimg, 125, 255)
find_contours(thresh, meancornerstoplisty, meancornersbottomlisty, meancornersleftlistx, meancornersrightlistx)
draw_rectangles(scaledimg, rects)
get_overlap(rects)
remove_overlapping_rectangles(overlaplist, rectsTL, rectsBR)
get_centroids(remainingrects, newrects)
draw_new_rectangles(scaledimg, newrects)
