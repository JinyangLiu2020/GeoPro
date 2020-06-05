import matplotlib.pyplot as plt
import math
import cv2
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from code._transform import *
from code._extract import *

if __name__ == "__main__":
    # transPoints()
    # threshold(226, 100, 2)
    # drawPoints()
    # findLaneMarking('back_equ.png')
    # clusterLaneMarking('lane_marking_int.txt')
    # fitLaneMarking('lane_marking_int.txt')
    # visualizeFit()

    x_c, y_c, z_c = enu_to_cc(-937.0089799531036, -1775.5955312868261, -2.52)
    x_i, y_i, direction = cc_to_ic(x_c, y_c, z_c)
    print(x_i, y_i, direction)

    back = cv2.imread('tmp/back.png')
    for i in range(1035, 1055):
        for j in range(1025, 1045):
            back[i, j] = [0, 0, 255]
    cv2.imwrite('tmp/back_inter.png', back)

    x_c, y_c, z_c = enu_to_cc(937.0089799531036, 1775.5955312868261, -2.52)
    x_i, y_i, direction = cc_to_ic(x_c, y_c, z_c)
    print(x_i, y_i, direction)

    back = cv2.imread('tmp/front.png')
    for i in range(995, 1015):
        for j in range(1025, 1045):
            back[i, j] = [0, 0, 255]
    cv2.imwrite('tmp/front_inter.png', back)