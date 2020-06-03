import matplotlib.pyplot as plt
import math
import cv2
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from _transform import *
from _extract import findLaneMarking, clusterLaneMarking

if __name__ == "__main__":
    # transPoints()
    # threshold(226, 100, 15)
    # drawPoints()
    # findLaneMarking('back_equ.png')
    clusterLaneMarking('lane_marking_int.txt')
    pass