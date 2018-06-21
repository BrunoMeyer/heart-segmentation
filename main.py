# -*- coding: utf-8 -*-

'''
 * @license
 * Copyright Bruno Henrique Meyer. All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at
 * https://github.com/BrunoMeyer/gene-selection-to-classification/blob/master/LICENSE
'''

import sys
from collections import defaultdict
import subprocess


import numpy as np
import cv2
import time
import os

import imutils
import math

SLICE = 10

THRESH_LOW_ENDO = 20
THRESH_HIGH_ENDO = 29
THRESH_LOW_EPI = 20
THRESH_HIGH_EPI = 29

RADIUS_THRESH = 24

# Change between "expert1" and "expert2"

expert_name = "expert1"
def getEndocardio(pimg, id_patient):

    # Remove the low values
    _, img = cv2.threshold(pimg,50,255,cv2.THRESH_TOZERO)


    # Execute the floodfill from the center of the circle
    # calculated for this pacient
    ldiff = THRESH_LOW_ENDO
    diff = THRESH_HIGH_ENDO
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    height, width = img.shape[:-1]
    mask1 = np.zeros((height+2, width+2), np.uint8)     # line 26
    cv2.floodFill(
        img,
        mask1,
        (circlesPatients[id_patient][0],circlesPatients[id_patient][1]),
        255,
        loDiff=(ldiff, ldiff, ldiff, ldiff),
        upDiff=(diff, diff, diff, diff)
    )
    mask1 = mask1[1:-1,1:-1]
    result = mask1*255
    for i,x in enumerate(result):
        for j,y in enumerate(x):
            disx = circlesPatients[id_patient][0] - i
            disy = circlesPatients[id_patient][1] - j
            if(math.sqrt(disx**2 + disy**2) > RADIUS_THRESH):
                result[i][j] = 0
    # Binarize the segmentation with values 0 e 255
    return result






def getEpicardio(pimg, endocardio, id_patient, systole, diastole):
    original_pimg = pimg

    # Remove the low values
    pimg[pimg < 30] = 0


    ldiff = THRESH_LOW_EPI
    diff = THRESH_HIGH_EPI



    # Remove the pixels referent to endocarde
    pimg = cv2.subtract(pimg, endocardio)

    # Performs iteratively with a median filter while doesnt have more alteration
    # This help to fix the variety in gradient of epicarde
    kernel = np.ones((5,5),np.uint8)
    copy = pimg.copy()
    aux = cv2.medianBlur(pimg,5)
    while(not (aux==copy).all()):
        aux = cv2.medianBlur(pimg,5)
        copy = aux
    pimg = aux




    # Calculate the mass center of endocarde
    thresh = endocardio
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = cnts[0]
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(safe_div(M["m10"] , M["m00"]))
    cY = int( safe_div(M["m01"] , M["m00"]))
    massCenter = (cX,cY)




    PADDING = 1
    x = massCenter[0]
    y = massCenter[1]
    img = cv2.cvtColor(pimg,cv2.COLOR_GRAY2BGR)
    height, width = img.shape[:-1]
    result = np.zeros((height, width), np.uint8)


    ANGLES = 360 # Number of directions to trace a line from the mass center
    #
    for i in range(ANGLES):
        _x = massCenter[0]
        _y = massCenter[1]
        angle = (math.pi*i*2)/ANGLES
        outEndocardio = False
        outEpicardio = False
        # For each line, find the next pixel with a padding
        # that is out of endocardio and execute floodfill
        while(True):
            _x+=math.cos(angle)
            _y+=math.sin(angle)
            x = round(_x)
            y = round(_y)
            pad_x = round(math.cos(angle)*PADDING)
            pad_y = round(math.sin(angle)*PADDING)
            if(x >= height-PADDING or x <= PADDING or y >= width-PADDING or y <= PADDING):
                break


            if(not outEndocardio):
                if( abs(pimg[x+pad_x][y+pad_y] - pimg[x][y])  < 100):
                    pass
                else:
                    if(outEpicardio):
                        break


            if(outEndocardio and abs(pimg[x+pad_x][y+pad_y] - pimg[x][y])  > 150):
                outEpicardio = True

            if(endocardio[x][y] == 0 and not outEndocardio):
                mask1 = np.zeros((height+2, width+2), np.uint8)
                cv2.floodFill(
                    pimg,
                    mask1,
                    (x+pad_x,y+pad_y),
                    255,
                    loDiff=(ldiff, ldiff, ldiff, ldiff),
                    upDiff=(diff, diff, diff, diff)
                )
                outEndocardio = True



    # Normalize and sum with endocardio segmentation
    result = pimg.copy()
    result[result < 255] = 0

    result = cv2.add(result, endocardio)
    for i,x in enumerate(result):
        for j,y in enumerate(x):
            disx = circlesPatients[id_patient][0] - i
            disy = circlesPatients[id_patient][1] - j
            if(math.sqrt(disx**2 + disy**2) > RADIUS_THRESH):
                result[i][j] = 0

    mask1 = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(
        result,
        mask1,
        (massCenter[0],massCenter[1]),
        1,
        loDiff=(ldiff, ldiff, ldiff, ldiff),
        upDiff=(diff, diff, diff, diff)
    )
    result[result != 1] = 0
    result[result == 1] = 255



    return result


# Get an slice from a 3d pgm image
def getSlice(pgmName,slc):
    try:
        subprocess.check_output(["./extractplane", pgmName, str(slc), "xy", "saida.pgm"])
    except subprocess.CalledProcessError as e:
        return []

    img = cv2.imread('saida.pgm',0)
    return img

# Find the most probable main circle in the image
# if circle argument is not None, return the image with the drawn circle
def findCircles(img, circle = None):
    if(not (circle is None) ):
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        for i in [circle]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0,0.1),1)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255,0.1),3)

        return img
    else:
        img_blur = cv2.medianBlur(img,5)
        cimg = cv2.cvtColor(img_blur,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,1,40,
                                param1=50,param2=30,minRadius=0,maxRadius=0)
        if(circles is None):
            return None

        circles = np.uint16(np.around(circles))
        return circles[0,:][0]



###### READ THE FILES PATHS ####
file = open("patients.txt")

patients = defaultdict(list)


for l in file:
    pat = l.replace("\n","").split("/")[1].replace("Pat","")
    patients[pat].append(l.replace("\n",""))


file = open("experts.txt")

experts = defaultdict(list)


for l in file:
    exp = l.replace("\n","").split("/")[1].replace("Pat","")
    experts[exp].append(l.replace("\n",""))

##################################

images = defaultdict(list)
circlesPatients = defaultdict(list)





ALL_TP = 0
ALL_FP = 0
ALL_TN = 0
ALL_FN = 0
total_images = 0

## Calculate fscore metric between two images
def calculate_metrics(img, expert, all=False):
    global ALL_TP
    global ALL_FP
    global ALL_TN
    global ALL_FN
    global total_images
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    total_images+=1
    for i,row in enumerate(img):
        for j,colum in enumerate(row):
            if(img[i][j] == 255 and expert[i][j] == 255):
                TP+=1
                ALL_TP+=1
            if(img[i][j] == 0 and expert[i][j] == 0):
                TN+=1
                ALL_TN+=1
            if(img[i][j] == 255 and expert[i][j] == 0):
                FP+=1
                ALL_FP+=1
            if(img[i][j] == 0 and expert[i][j] == 255):
                FN+=1
                ALL_FN+=1
    precision = safe_div(TP,(TP+FP))
    recall = safe_div(TP,(TP+FN))
    
    if(precision == 0 and recall ==0):
        avg = 0
    else:
        avg = 2*precision*recall/(precision+recall)
    
    if(not all):
        return avg
    return (avg,precision,recall)

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y






if(__name__ == "__main__"):
    avgs_fscore = []
    avgs_precision = []
    avgs_recall = []

    for x in patients:
        patients[x].sort()
        experts[x].sort()


        SLICE = 0
        # Load the experts segmentations
        while(True):
            for imgName in experts[x]:
                if("systole_endocarde" in imgName and expert_name in imgName):
                    systole_endocarde_expert1 = getSlice(imgName,SLICE)
                if("diastole_endocarde" in imgName and expert_name in imgName):
                    diastole_endocarde_expert1 = getSlice(imgName,SLICE)
                if("systole_epicarde" in imgName and expert_name in imgName):
                    systole_epicarde_expert1 = getSlice(imgName,SLICE)
                if("diastole_epicarde" in imgName and expert_name in imgName):
                    diastole_epicarde_expert1 = getSlice(imgName,SLICE)
            if(len(systole_endocarde_expert1) == 0):
                SLICE=0
                break


            # Calculate the median of the circle parameters for an
            # slice of an patient
            circlesPatients[x] = []
            for y in patients[x]:
                img1 = getSlice(y,SLICE)
                if(len(img1) > 0):
                    aux = findCircles(img1)
                    if(not (aux is None)):
                        circlesPatients[x].append(aux)

            arr = np.array(circlesPatients[x])
            a1 = int(np.median(  np.array(arr[arr != None])  ))
            arr = np.array(circlesPatients[x])[:,1]
            a2 = int(np.median(  np.array(arr[arr != None])  ))
            arr = np.array(circlesPatients[x])[:,2]
            r = int(np.average(  np.array(arr[arr != None])  ))
            circlesPatients[x] = [a1,a2,r]




            # Execute the segmentations and compare with the experts
            i = 0
            systole = open("./HeartDatabase/Pat"+x+"/info.txt").readlines()[0].split("=")[-1].replace(" ","").replace("\n","")

            systole = "./HeartDatabase/Pat"+x+"/img/"+systole
            systole = getSlice(systole,SLICE)
            diastole = patients[x][-1]
            diastole = getSlice(diastole,SLICE)




            img_endo= getEndocardio(systole,x)

            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_original_systole.jpg",systole)
            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_original_diastole.jpg",diastole)


            fscore,precision,recall = calculate_metrics(img_endo,systole_endocarde_expert1, all=True)
            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_systole_endo.jpg",img_endo)
            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_systole_endo_expert.jpg",systole_endocarde_expert1)
            avgs_fscore.append(fscore)
            avgs_precision.append(precision)
            avgs_recall.append(recall)
            # print(fscore)

            img_epi = getEpicardio(systole,img_endo,x,systole,diastole)
            fscore,precision,recall = calculate_metrics(img_epi,systole_epicarde_expert1, all=True)
            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_systole_epi.jpg",img_epi)
            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_systole_epi_expert.jpg",systole_epicarde_expert1)
            avgs_fscore.append(fscore)
            avgs_precision.append(precision)
            avgs_recall.append(recall)



            img_endo = getEndocardio(diastole,x)
            fscore,precision,recall = calculate_metrics(img_endo,diastole_endocarde_expert1, all=True)
            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_diastole_endo.jpg",img_endo)
            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_diastole_endo_expert.jpg",diastole_endocarde_expert1)
            avgs_fscore.append(fscore)
            avgs_precision.append(precision)
            avgs_recall.append(recall)

            img_epi = getEpicardio(diastole,img_endo,x,systole,diastole)
            fscore,precision,recall = calculate_metrics(img_epi,diastole_epicarde_expert1, all=True)
            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_diastole_epi.jpg",img_epi)
            cv2.imwrite("output/Pat_"+x+"_"+str(SLICE)+"_diastole_epi_expert.jpg",diastole_epicarde_expert1)
            avgs_fscore.append(fscore)
            avgs_precision.append(precision)
            avgs_recall.append(recall)

            SLICE+=1



    # Show the fscore average
    print("f1score average: ", sum(avgs_fscore)/len(avgs_fscore))
    print("precision average: ", sum(avgs_precision)/len(avgs_precision))
    print("recall average: ", sum(avgs_recall)/len(avgs_recall))

