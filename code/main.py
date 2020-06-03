import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = Axes3D(fig)

def fakeTrans():
    with open('data\point_cloud.fuse') as cloud_points_file:
        lines = cloud_points_file.readlines()
        points = []
        for line in lines:
            point = [float(x) for x in line.split(' ')]
            point[0] *= 30000
            point[1] *= 30000
            point[-1] = 200 if point[-1]>50 else 0
            point.append(point[-1])
            point.append(point[-1])
            points.append(point)
        points = np.array(points)[:,:]
        np.savetxt('points_int.txt', points)

# Camera location:45.90414414, 11.02845385
def calDis(x1,y1,x2=45.90414414*30000,y2=11.02845385*30000):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def threshold(latitude, greyscale, distance):
    points_int = np.loadtxt('points_int.txt')
    lane_markings = []
    for line in points_int:
        if line[2] < latitude and line[3] > greyscale and calDis(line[0], line[1])<distance:
            lane_markings.append(line)
    lane_markings = np.array(lane_markings)
    np.savetxt('lane_marking.txt', lane_markings)

if __name__ == "__main__":
    threshold(226, 100, 5)