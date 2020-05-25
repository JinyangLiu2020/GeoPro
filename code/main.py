import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = Axes3D(fig)

# i = 0
# with open('data\point_cloud.fuse') as cloud_points_file:
#     lines = cloud_points_file.readlines()
#     points = []
#     for line in lines:
#         if i%3 == 1:
#             point = [float(x) for x in line.split(' ')]
#             points.append(point)
#         i+=1
#     points = np.array(points)
#     print(points.shape)
#     x = points[:,0]
#     y = points[:,1]
#     z = points[:,2]
#     ax.scatter(x,y,z,s=0.1, alpha=0.5)
#     plt.show()

with open('data\point_cloud.fuse') as cloud_points_file:
    lines = cloud_points_file.readlines()
    points = []
    for line in lines:
        point = [float(x) for x in line.split(' ')]
        point[0] *= 30000
        point[1] *= 30000
        points.append(point)
    points = np.array(points)[:,:3]
    np.savetxt('points.txt', points)