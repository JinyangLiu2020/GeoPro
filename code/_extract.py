# Camera location:45.90414414, 11.02845385
import matplotlib.pyplot as plt
import numpy as np
import cv2

trajectery = [[-31.494925251842638, -60.13946825281302],
[-25.816078275037732, -49.38318021009172],
[-21.756777818577362, -41.666926801333915],
[-17.47865374782263, -33.47493469711093],
[-11.861159551419213, -22.70418328163622],
[-5.639180912733428, -10.766316407903346],
[-3.0035279818818594e-11, 3.3413050104513786e-10],
[3.6968461077552655, 7.024896507340007],
[9.401178150010027, 17.79122182159975]]

def calDis(x1,y1):
    return min([((x1-p[0])**2+(y1-p[1])**2)**0.5 for p in trajectery])

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

def fitLaneMarking(filename):
    points = np.loadtxt(filename)
    row, col = points.shape
    x = points[:,0].reshape(row,1)
    y = points[:,1].reshape(row,1)
    print(min(x), max(x))

    from sklearn import linear_model
    ransac = linear_model.RANSACRegressor(residual_threshold=2)
    ransac.fit(x, y)
    inlier_mask = ransac.inlier_mask_

    # samples = np.arange(10000)/10000
    # results = np.apply_along_axis(ransac.predict, 1, samples.reshape())
    # np.savetxt('res', results)

    x_i = x[inlier_mask]
    y_i = y[inlier_mask]
    valid_row = x_i.shape[0]
    regr = linear_model.LinearRegression()
    regr.fit(x_i, y_i)

    c = regr.coef_[0][0]
    i = regr.intercept_[0]

    np.savetxt('x',x.reshape(row))
    np.savetxt('y',y.reshape(row))
    return c, i

def visualizeFit():
    x = np.loadtxt('x')
    y = np.loadtxt('y')

    c, i = 1.890783747977774, -3.914180282266038
    p1 = [-35, 15]
    p2 = [c*(-35)+i, c*15+i]

    plt.plot(p1, p2)
    plt.scatter(x, y)
    plt.show()

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