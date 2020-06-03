# Camera location:45.90414414, 11.02845385
import numpy as np
import cv2

def calDis(x1,y1,x2=9.404743650520686e-11,y2=-1.7409718111593975e-10):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def threshold(altitude, greyscale, distance):
    points_int = np.loadtxt('points_int.txt')
    lane_markings = []
    for line in points_int:
        if line[2] < altitude and line[3] > greyscale and calDis(line[0], line[1])<distance:
            lane_markings.append(line)
    lane_markings = np.array(lane_markings)
    np.savetxt('lane_marking_int.txt', lane_markings)

def findLaneMarking(filename):
    img = cv2.imread(filename)
    kernel = np.ones((3,3),np.uint8) 
    dilation = cv2.dilate(img, kernel, iterations=3)
    cv2.imshow('img', dilation)
    cv2.waitkey(0)

def findLaneMarking(filename):
    img = cv2.imread(filename)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=3)
    cv2.imwrite('dialtion.jpg', dilation)
    # cv2.waitKey(0)