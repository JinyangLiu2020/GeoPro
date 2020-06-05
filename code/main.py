import matplotlib.pyplot as plt
import math
import cv2
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from _transform import *
from _extract import *

if __name__ == "__main__":
    # transPoints()
    # threshold(226, 100, 2)
    # drawPoints()
    # findLaneMarking('back_equ.png')
    # clusterLaneMarking('lane_marking_int.txt')
    # fitLaneMarking('lane_marking_int.txt')
    # visualizeFit()

    x_c, y_c, z_c = enu_to_cc(937.0089799531036, 1775.5955312868261, -2.52)
    x_i, y_i, direction = cc_to_ic(x_c, y_c, z_c)
    print(cc_to_ic(enu_to_cc()))