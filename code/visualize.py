import numpy as np
import cv2
import matplotlib.pyplot as plt
x = np.loadtxt('tmp/x')
y = np.loadtxt('tmp/y')

intercetion = [937.0089799531036, 1775.5955312868261]

c1, i1 = 1.890783747977774, -3.914180282266038
c2, i2 = 1.8991898379945529, 3.9624015497498632
p1 = [-35, 15]
p2 = [c1*(-35)+i1, c1*15+i1]
p3 = [c2*(-35)+i2, c2*15+i2]

x_i = (i1-i2)/(c2-c1)
y_i = c1*x_i+i1
print(x_i, y_i)

plt.plot(p1, p2)
plt.plot(p1, p3)
plt.scatter(x, y)
plt.show()