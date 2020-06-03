import numpy as np
import cv2
import matplotlib.pyplot as plt
x = np.loadtxt('x')
y = np.loadtxt('y')

c, i = 1.890783747977774, -3.914180282266038
p1 = [-35, 15]
p2 = [c*(-35)+i, c*15+i]

plt.plot(p1, p2)
plt.scatter(x, y)
plt.show()