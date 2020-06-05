import cv2
import numpy as np

def find_lane_marking(file_path, size = (512,512)):
    img = cv2.imread(file_path)
    hsv = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2HSV), size)
    lower_white, upper_white = np.array([0, 0, 205]), np.array([180, 90, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    for i in range(size[0]):
        for j in range(size[1]):
            if(i<size[0]/2 or i+0.4*j-size[0]/2*1.4<0 or i-0.4*j -size[0]/2*0.6<0 or i>1*size[0]):
                mask[i,j] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.dilate(cv2.erode(mask,kernel),kernel)
    mask = cv2.dilate(mask,kernel)
    return mask

def find_lanes(mask,size = (1024,1024)):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    lanes,pics = [],[]
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
        start, end = (y_start,x_start), (y_end,x_end)
        lanes.append([start,end])
        pics.append(label)
    return lanes,pics

def find_collineation(lanes,pics):
    long_lines, short_lines, slopes, lines = [], [], [], []
    for i in range(len(lanes)):
        if(abs(lanes[i][0][0]-lanes[i][1][0])<=5 or abs(lanes[i][0][1]-lanes[i][1][1])<=5):
            short_lines.append([lanes[i],pics[i]])
        else:
            long_lines.append([lanes[i],pics[i]])
    for i in range(len(long_lines)):
        slope = (long_lines[i][0][0][1]-long_lines[i][0][1][1])/(long_lines[i][0][0][0]-long_lines[i][0][1][0])
        if len(slopes) == 0:
            slopes.append(slope)
            lines.append([long_lines[i]])
        else:
            flag = False
            for j in range(len(slopes)):
                if(slopes[j]*0.75<slope<slopes[j]*1.25 or slopes[j]*1.25<slope<slopes[j]*0.75):
                    flag = True
                    lines[j].append(long_lines[i])
            if not flag:
                slopes.append(slope)
                lines.append([long_lines[i]])
    return lines

mask = find_lane_marking('.\\data\\image\\back.jpg',(1024,1024))
cv2.imwrite('mask_back.png',mask*256)
cv2.waitKey()

lanes,pics = find_lanes(mask)

img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB);
for i in range(len(lanes)):
    print(lanes[i])
    start = lanes[i][0]
    end = lanes[i][1]
    point_color = (0, 0, 255) # BGR
    thickness = 2
    lineType = 4
    cv2.line(img, start, end, point_color, thickness, lineType)
    cv2.imshow('img',img)
    cv2.waitKey()
#front:1044;986
start = (522 ,493)
end1 = (365,713)
end2 = (653,664)
#back:1043;1049
#start = (521,524)
#end1 = (843,965)
#end2 = (127,1024)
point_color = (0, 255, 0) # BGR
thickness = 2
lineType = 4
cv2.line(img, start, end1, point_color, thickness, lineType)
cv2.line(img, start, end2, point_color, thickness, lineType)
#cv2.imwrite('front_rgb_match.png',img)
#cv2.waitKey()
'''
lines = find_collineation(lanes,pics)
for i in range(len(lines)):
    img = lines[i][0][1]*255
    print(type(img))
    cv2.imshow('img',img)
    cv2.waitKey()
    for j in range(1,len(lines[i])):
        img +=lines[i][j][1]*255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB);
    for j in range(len(lines[i])):
        ptStart = lines[i][j][0][0]
        ptEnd = lines[i][j][0][1]
        point_color = (0, 255, 0) # BGR
        thickness = 1
        lineType = 4
        cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    cv2.imshow('img',img)
    cv2.waitKey()
'''
