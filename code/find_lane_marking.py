import cv2
import numpy as np

def find_lane_marking(file_path, size = (512,512)):
    img = cv2.imread(file_path)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv = cv2.resize(hsv, size)
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    for i in range(size[0]):
        for j in range(size[1]):
            if(i<size[0]/2 or i+0.3*j-size[0]/2*1.3<0 or i-0.4*j -size[0]/2*0.6<0 or i>0.78*size[0]):
                mask[i,j] = 0
    return mask

mask = find_lane_marking('..\\data\\image\\front.jpg',(1024,1024))
cv2.imshow('mask',mask)
cv2.waitKey()
