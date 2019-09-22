# RUN ON PYTHON 3.7 #
## Reads the .INSELECT file coordinates and converts it into a .XML file ##
## For standardization purposes so that the accuracy can be compared against the Ground Truth ##

import json
import os
import tkinter
from tkinter import filedialog
import re
import numpy as np
import cv2
import xml.etree.ElementTree as xml
from PIL import Image
import matplotlib as plt

## Declare empty lists ##
data = []
multiplied = []
rects = []
xcoords = []
ycoords = []
widthcoords = []
heightcoords = []
numofrects = []

## Open the directory to get the file ##
def load_inselect_file():
    global filename, rects, inselectfile
    roottk = tkinter.Tk()
    roottk.withdraw()
    filename =  filedialog.askopenfilename(initialdir = "/C:", title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    inselectfile = re.sub(".jpg",".inselect",filename)

    ## Load the .inselect file (in json format) ##
    with open(inselectfile) as json_file:
        data = json.load(json_file)

    ## Get the coordinates of the rectangles from the 'items' ##
    items = (data['items'])

    ## Regex manipulation to get the data ##
    stritems = str(items)
    editone = re.sub("fields", "", stritems)
    edittwo = re.sub(", 'rotation': 0", "", editone)
    editthree = re.sub("'rotation': 0", "", edittwo)
    editfour = re.sub("{", "", editthree)
    editfive = re.sub("}", "", editfour)
    editsix = re.sub("\\'\\': ,", "", editfive)
    editseven = re.sub(" 'rect': ", "", editsix)
    editeight = re.sub("], , ", "],", editseven)
    editnine = re.sub("\[", "", editeight)
    editten = re.sub("\]", "", editnine)
    editeleven = re.sub(", ", ",", editten)

    ## Convert the string to a list ##
    numbers = editeleven.split(",")

    ## Append the values to another list for division ##
    ## Change the numbers from str to float, round the number and change the final value to an integer ##
    for i in range(len(numbers)):
        multiplied.append(int(round(float(numbers[i])*800,3)))

    ## Save the x-coords, y-coords, width and height for each rectangle ##
    xcoords = multiplied[::4]
    ycoords = multiplied[1::4]
    widthcoords = multiplied[2::4]
    heightcoords = multiplied[3::4]

    ## Append it into a list ##
    for i in range(len(xcoords)):
        rects.append((xcoords[i],ycoords[i],widthcoords[i],heightcoords[i]))

    rects = sorted(rects)

    return rects, filename, inselectfile

def rescale_image(filename, width, height):
    ## Load the image #
    global scaledimg
    oriimg = cv2.imread(filename)
    scaledimg = cv2.resize(oriimg,(width,height))
    return scaledimg

def draw_rectangles(scaledimg, rects):
    ## Sort the rectangles ##
    boxnum = 0

    ## Draw the rectangles onto the image ##
    for rect in rects:
        numofrects.append(boxnum)
        (x, y, w, h) = rect
        cv2.rectangle(scaledimg, (x, y), (x+w, y+h), (0,255,0), 2);
        cv2.putText(scaledimg, str(boxnum), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        boxnum+=1

def save_coordinates_to_xml(filename,numofrects,xcoords,ycoords,widthcoords,heightcoords):
    ## Save the coordinates in an .xml file ##
    filenamexml = re.sub("C:/Users/Eugene Goh/Desktop/MSc Computing/Dissertation/Altered_Images/SelectedImages/GroundTruth_BoundingBox/", "", filename)
    xmlfile = re.sub(".jpg",".xml",filename)

    root = xml.Element("Information")
    imagenameelement = xml.Element("ImageName")
    imagenamesubelement = xml.SubElement(imagenameelement, "imagenamesubelement")
    imagenamesubelement.text = filenamexml
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
    rescale_image(filename,800,800)
    draw_rectangles(scaledimg,rects)
    scaledimgr = cv2.resize(scaledimg, (600, 600))
    cv2.imshow('image',scaledimgr)
    #save_coordinates_to_xml(filename,numofrects,xcoords,ycoords,widthcoords,heightcoords)
