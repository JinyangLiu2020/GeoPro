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

def clusterLaneMarking(filename):
    points = np.loadtxt(filename)
    points = points[:,:]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=7, random_state=0).fit(points)
    labels = kmeans.labels_
    # np.savetxt('lables', labels)
    color_cluster = []
    colors = [[240,240,240],[240,0,0],[0,240,0],[0,0,240],[240,240,0],[240,0,240],[0,240,240]]
    for i in range(points.shape[0]):
        color = colors[labels[i]]
        colored_point = []
        for value in points[i][:-1]:
            colored_point.append(value)
        for value in color:
            colored_point.append(value)
        # print(colored_point)
        color_cluster.append(colored_point)
    np.savetxt('color_cluster.txt', np.array(color_cluster))

        

def findLaneMarking(filename):
    img = cv2.imread(filename)
    kernel = np.ones((10,10),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=3)
    # cv2.imwrite('dialtion.jpg', dilation)
    # dilation = cv2.imread('perfect.jpg')

    # imgray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(dilation, contours, -1, (0,120,0), 3)
    # print(contours, hierarchy)
    # cv2.imwrite('im2.jpg', dilation)

    edges = cv2.Canny(dilation,0,250,apertureSize = 3)
    cv2.imwrite('edges.jpg', edges)
    lines = cv2.HoughLines(edges,1,np.pi/180,50)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(dilation,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imwrite('houghlines3.jpg',dilation)