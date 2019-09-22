'''
RUN ON PYTHON AND ABOVE 3.7 ONLY
* Hungarian Algorithm code was adapted from munkres (http://software.clapper.org/munkres/), date unknown *
* Accessed 7-7-2019 *
http://software.clapper.org/munkres/

Code to get the accuracy of a segmentation
1. Input Ground Truth XML
2. Input algorithm-generated XML
3. Input image
4. Get the accuracy
'''
import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import re
import numpy as np
import cv2
import xml.etree.ElementTree as xml
from PIL import Image
import csv
from munkres import Munkres, print_matrix
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import statistics
import sys
import time

### Declare empty lists ###
dataGT = []
rrectnumGT = []
xcoordsGT = []
ycoordsGT = []
widthGT = []
heightGT = []
rectsGT = []
numofrectsGT = []
rectsGTTL = []
rectsGTBR = []

dataOA = []
rectnumOA = []
xcoordsOA = []
ycoordsOA = []
widthOA = []
heightOA = []
rectsOA = []
numofrectsOA = []
rectsOATL = []
rectsOABR = []

alliou = []
overlaplist = []
truefalselist = []
sortedtruefalselist = []
tpcountslist = []
fncountslist = []
fpcountslist = []

trueiou = []

### Load GT XML data ###
def load_GT_file():
    rootGT = tkinter.Tk()
    rootGT.withdraw()

    messagebox.showinfo("GT XML", "Please select the Ground Truth XML")
    filenameGT =  filedialog.askopenfilename(initialdir = "/C:", title = "Select file",filetypes = (("xml files","*.xml"),("all files","*.*")))

    treeGT = xml.parse(filenameGT)
    rootGT = treeGT.getroot()

    for elem in rootGT:
        for subelem in elem:
            dataGT.append(subelem.text)

    return dataGT

### Load OA XML data ###
def load_OA_file():
    rootOA = tkinter.Tk()
    rootOA.withdraw()

    messagebox.showinfo("OA XML", "Please select the Algorithm XML")
    filenameOA =  filedialog.askopenfilename(initialdir = "/C:", title = "Select file",filetypes = (("xml files","*.xml"),("all files","*.*")))

    treeOA = xml.parse(filenameOA)
    rootOA = treeOA.getroot()

    for elem in rootOA:
        for subelem in elem:
            dataOA.append(subelem.text)
            
    return dataOA

### Load image ###
def load_image():

    rootIMG = tkinter.Tk()
    rootIMG.withdraw()

    messagebox.showinfo("Image", "Please select the corresponding image")

    filenameIMG =  filedialog.askopenfilename(initialdir = "/C:", title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

    W = 800
    H = 800
    oriimg = cv2.imread(filenameIMG)
    height, width, depth = oriimg.shape
    imgScaleWidth = W/width
    imgScaleHeight = H/height
    newX,newY = oriimg.shape[1]*imgScaleWidth, oriimg.shape[0]*imgScaleHeight

    newimg = cv2.resize(oriimg,(int(newX),int(newY)))

    return newimg, filenameIMG

class algorithm_accuracy:
    def __init__(self, dataGT, dataOA):
        self.dataGT = dataGT
        self.dataOA = dataOA

    def split_GT_OA(dataGT, dataOA):
        dataGT = dataGT[1:]
        dataOA = dataOA[1:]

        rectnumGT = dataGT[::5]
        xcoordsGT = dataGT[1::5]
        ycoordsGT = dataGT[2::5]
        widthGT = dataGT[3::5]
        heightGT = dataGT[4::5]

        rectnumOA = dataOA[::5]
        xcoordsOA = dataOA[1::5]
        ycoordsOA = dataOA[2::5]
        widthOA = dataOA[3::5]
        heightOA = dataOA[4::5]

        for i in range(len(rectnumGT)):
            rectsGT.append((int(xcoordsGT[i]),int(ycoordsGT[i]),int(widthGT[i]),int(heightGT[i])))

        for i in range(len(rectnumOA)):
            rectsOA.append((int(xcoordsOA[i]),int(ycoordsOA[i]),int(widthOA[i]),int(heightOA[i])))

        return rectsGT, rectnumGT, rectsOA, rectnumOA

    ### Plot GT Rectangles ###
    def draw_rectangles(newimg, rectsGT, rectnumGT, rectsOA, rectnumOA):
        global numofrectsOA, numofrectsGT
        boxnumGT = 0
        boxnumOA = 0

        for rect in rectsOA:
            numofrectsOA.append(boxnumOA)
            (x, y, w, h) = rect
            cv2.rectangle(newimg, (x, y), (x+w, y+h), (200,150,0), 2);
            cv2.putText(newimg, str(rectnumOA[boxnumOA]), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,150,0), 1)
            boxnumOA+=1

        for rect in rectsGT:
            numofrectsGT.append(boxnumGT)
            (x, y, w, h) = rect
            cv2.rectangle(newimg, (x, y), (x+w, y+h), (0,0,255), 2);
            cv2.putText(newimg, str(rectnumGT[boxnumGT]), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            boxnumGT+=1

        cv2.imshow('Detected', newimg)

        return numofrectsOA, numofrectsGT

    def save_image(newimg, filenameIMG):
        savedimagename = re.sub("C:/Users/Eugene Goh/Desktop/MSc Computing/Dissertation/Altered_Images/SelectedImages/GroundTruth_BoundingBox/", "", filenameIMG)
        savedimagename = re.sub(".jpg", "_LoG_6000x6000.jpg", savedimagename)
        cv2.imwrite(str(savedimagename),newimg)

    ### Get the TL and BR for all rectangles ###
    # To get BR Coordinates, take TL Coordinates + Height or + Width respectively
    # Assuming that: x1 < x2 and y1 < y2 applies for both rectangles, then:
    ## [][] = rectangle number, rectangle coords 
    def get_overlap(rectsGT, rectsOA):
        for i in range(len(rectsGT)):
            rectsGTTL.append((rectsGT[i][0],rectsGT[i][1]))
            rectsGTBR.append((rectsGT[i][0]+rectsGT[i][2],rectsGT[i][1]+rectsGT[i][3]))

        for i in range(len(rectsOA)):
            rectsOATL.append((rectsOA[i][0],rectsOA[i][1]))
            rectsOABR.append((rectsOA[i][0]+rectsOA[i][2],rectsOA[i][1]+rectsOA[i][3]))

    ## Get Overlapping area for two rectangles ##
        for GT in range(len(rectsGT)):
            for OA in range(len(rectsOA)):
                width = min(rectsGTBR[GT][0], rectsOABR[OA][0]) - max(rectsGTTL[GT][0], rectsOATL[OA][0])
                height = min(rectsGTBR[GT][1], rectsOABR[OA][1]) - max(rectsGTTL[GT][1], rectsOATL[OA][1])

                if width <= 0 or height <= 0:
                    alliou.append(0)
                    #print(str(rectnumGT[GT]) + " and " +  str(rectnumOA[OA]) + " do not overlap.")
                else:
                    Area = width * height
                    rectGTArea = (rectsGTBR[GT][0]-rectsGTTL[GT][0])*(rectsGTBR[GT][1]-rectsGTTL[GT][1])
                    rectOAArea = (rectsOABR[OA][0]-rectsOATL[OA][0])*(rectsOABR[OA][1]-rectsOATL[OA][1])
                    IntersectionOverUnion = Area / float(rectGTArea+rectOAArea - Area)
                    IntersectionOverUnion = round(IntersectionOverUnion,5)
                    alliou.append(IntersectionOverUnion)
                    overlaplist.append([rectnumGT[GT], rectnumOA[OA], Area])
                    #print(str(rectnumGT[GT]) + " and " +  str(rectnumOA[OA]) + " overlaps. The area of overlap is " + str(Area) + " px.")
                    
                    #print("Area of GT rectangle " + str(rectnumGT[GT]) + " is " + str(rectGTArea) + "px.")
                    #print("Area of OA rectangle " + str(rectnumOA[OA]) + " is " + str(rectOAArea)+ "px.")
                    #print("Intersection Over Union is " + str(IntersectionOverUnion))

        ioulist = [alliou[i * len(numofrectsOA):(i + 1) * len(numofrectsOA)] for i in range((len(alliou) + len(numofrectsOA) - 1) // len(numofrectsOA) )]

        return ioulist, overlaplist

    def save_overlap_csv(filenameIMG, alliou, numofrectsOA, numofrectsGT):
        savedcsvname = re.sub("C:/Users/Eugene Goh/Desktop/MSc Computing/Dissertation/Altered_Images/SelectedImages/GroundTruth_BoundingBox/", "", filenameIMG)
        savedcsvname = re.sub(".jpg", ".csv", savedcsvname)

        numofrectsOAcsv = ["Rectangle No."] + numofrectsOA

        ## Divide to get the number of rows ##
        ioulist = [alliou[i * len(numofrectsOA):(i + 1) * len(numofrectsOA)] for i in range((len(alliou) + len(numofrectsOA) - 1) // len(numofrectsOA) )]

        with open(savedcsvname, 'w', encoding='utf-8', newline = '') as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            writer.writerow(numofrectsOAcsv)

            for i in range(len(ioulist)):
                writer.writerow([numofrectsGT[i]] + ioulist[i])

    ## Apply the Hungarian algorithm ##
    ## Source code adapted from: http://software.clapper.org/munkres/ ##
    def hungarian_algorithm(ioulist):
        cost_matrix = []
        tpvalues = []

        for row in ioulist:
            cost_row = []
            for col in row:
                cost_row += [sys.maxsize - col]
            cost_matrix += [cost_row]
            #print(cost_matrix)

        m = Munkres()
        indexes = m.compute(cost_matrix)
        total = 0

        for row, column in indexes:
            value = ioulist[row][column]
            trueiou.append(value)
            total += value
            if value >= 0.5:
                truefalselist.append('True')
            else:
                truefalselist.append('False')

        #    print(f'({row}, {column}) -> {value}')
        #print(f'total profit: {total}')

        tpcounter = truefalselist.count('True')
        fncounter = len(numofrectsGT) - tpcounter
        fpcounter = abs(len(numofrectsOA) - len(numofrectsGT))

        return sortedtruefalselist, fpcounter, tpcounter, fncounter

    def cal_precision_recall_f1(tpcounter, fpcounter, fncounter):
        global pscore, rscore, f1score

        pscore = tpcounter/(tpcounter+fpcounter)
        rscore = tpcounter/(tpcounter+fncounter)
        try:
            f1score = 2*((pscore*rscore)/(pscore+rscore))
        except ZeroDivisionError:
            print("There are no Ground Truth rectangles")
            return
            

        print("Number of Ground Truth rectangles: " + str(len(numofrectsGT)))
        print("Number of algorithm generated rectangles: " + str(len(numofrectsOA)))

        print("True Positives: " + str(tpcounter))
        print("False Negatives: " + str(fncounter))
        print("False Positives: " + str(fpcounter))

        print("F1 score for IoU 0.5: " + str(f1score))

        return pscore, rscore, f1score

    def plot_pr_f1(pscorelist, rscorelist, f1scorelist):
        thresholdlist = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
        f = plt.figure(figsize=(10,3))
        ax = f.add_subplot(121)
        ax.plot(rscorelist,pscorelist)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall')

        ax2 = f.add_subplot(122)
        ax2.plot(range(len(f1scorelist)),f1scorelist)
        ax2.set_xlabel('IoU Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score for all IoU')

        f.show()
        print("Maximum f1 score is " + str(max(f1scorelist)))
        print("Mean f1 score is " + str(sum(f1scorelist)/11))
        aucf1 = auc(thresholdlist,f1scorelist)
        print("Area under f1 curve is " + str(aucf1))
        print("Area under f1 curve (normalised) is " + str((aucf1/0.5)*100))


if __name__ == '__main__':
    load_GT_file()
    load_OA_file()
    newimg, filenameIMG = load_image()
    startTime = time.time()
    algorithm_accuracy(dataGT, dataOA)
    rectsGT, rectnumGT, rectsOA, rectnumOA = algorithm_accuracy.split_GT_OA(dataGT, dataOA)
    algorithm_accuracy.draw_rectangles(newimg, rectsGT, rectnumGT, rectsOA, rectnumOA)
    algorithm_accuracy.save_image(newimg, filenameIMG)
    ioulist, overlaplist = algorithm_accuracy.get_overlap(rectsGT, rectsOA)
    #algorithm_accuracy.save_overlap_csv(filenameIMG, alliou, numofrectsOA, numofrectsGT)
    sortedtruefalselist, fpcounter, tpcounter, fncounter = algorithm_accuracy.hungarian_algorithm(ioulist)
    algorithm_accuracy.cal_precision_recall_f1(tpcounter, fpcounter, fncounter)
    #algorithm_accuracy.plot_pr_f1(pscorelist, rscorelist, f1scorelist)
    print ('The script took {0} seconds !'.format(time.time() - startTime))

