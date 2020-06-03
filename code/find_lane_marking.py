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

def find_lanes(mask,size = (1024,1024)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.erode(mask,kernel)
    mask = cv2.dilate(mask,kernel)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    lanes = []
    pics = []
    for i in range(1,len(stats)):
        label = (labels==i).astype(np.uint8)
        y_sum, x_sum = label.sum(axis = 0), label.sum(axis = 1)
        x,y_start_arr,y_end_arr = [],[],[]
        for i in range(len(x_sum)):
            if x_sum[i] != 0:
                x.append(i)
        x_start,x_end = x[len(x)//20],x[len(x)*19//20]
        for i in range(len(label[x_start])):
            if label[x_start][i] != 0:
                y_start_arr.append(i)
            if label[x_end][i] != 0:
                y_end_arr.append(i)
        y_start = y_start_arr[len(y_start_arr)//2]
        y_end = y_end_arr[len(y_end_arr)//2]
        start = (y_start,x_start)
        end = (y_end,x_end)
        lanes.append([start,end])
        pics.append(label)
    return lanes,pics


mask = find_lane_marking('..\\data\\image\\front.jpg',(1024,1024))
lanes,pics = find_lanes(mask)

'''
for i in range(1,len(lanes)):
    label = pics[i]
    ptStart = lanes[i][0]
    ptEnd = lanes[i][1]
    point_color = (0, 255, 0) # BGR
    thickness = 1
    lineType = 4
    img = cv2.cvtColor(label*255, cv2.COLOR_GRAY2RGB);
    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    cv2.imshow('img',img)
    cv2.waitKey()
'''
