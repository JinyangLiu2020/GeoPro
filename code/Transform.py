import numpy as np
import math
import cv2



def lla_to_ecef(lat, lon, alt):
    a = 6378137.0
    b = 6356752.314245

    f = (a - b)/a
    e = math.sqrt(f*(2-f))

    sin_phi = (np.sin(lat*math.pi/180))
    cos_phi = (np.cos(lat*math.pi/180))
    sin_lambda = (np.sin(lon*math.pi/180))
    cos_lambda = (np.cos(lon*math.pi/180))

    N = a/(math.sqrt(1-(e**2)*(sin_phi**2)))

    x = (alt + N)*cos_lambda*cos_phi
    y = (alt + N)*cos_phi*sin_lambda
    z = (alt + (1-e**2)*N)*sin_phi

    return x,y,z,cos_phi, sin_phi, cos_lambda, sin_lambda

def ecef_to_enu(x,y,z,cos_phi, sin_phi, cos_lambda, sin_lambda):
    x0, y0, z0, cos_phi0, sin_phi0, cos_lambda0, sin_lambda0 = lla_to_ecef(45.90414414, 11.02845385,227.5819)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    e = -sin_lambda0*dx          + cos_lambda0*dy
    n = -cos_lambda0*sin_phi0*dx - sin_phi0*sin_lambda0*dy + cos_phi0*dz
    u = cos_phi0*cos_lambda0*dx  + cos_phi0*sin_lambda0*dy + sin_phi0*dz

    return e, n, u

def enu_to_cc(e, n, u):
    qs, qx, qy, qz = 0.362114, 0.374050, 0.592222, 0.615007 # Camera parameters

    P = [n,e,-u]
    Rq = [
    		[1-2*qy**2-2*qz**2 , 2*qx*qy+2*qs*qz   , 2*qx*qz-2*qs*qy],
    		[2*qx*qy-2*qs*qz   , 1-2*qx**2-2*qz**2 , 2*qy*qz+2*qs*qx],
    		[2*qx*qz+2*qs*qy   , 2*qy*qz-2*qs*qx   , 1-2*qx**2-2*qy**2]
    	 ]
    x_c = np.dot(Rq,P)[0]
    y_c = np.dot(Rq,P)[1]
    z_c = np.dot(Rq,P)[2]

    return x_c, y_c, z_c

def cc_to_ic(x_c, y_c, z_c):
    global x_i
    global y_i
    global direction
    # front
    if z_c > 0 and z_c > abs(x_c) and z_c > abs(y_c):
        x_i = int(y_c/z_c*(resolution - 1)/2+(resolution + 1)/2)
        y_i = int(x_c/z_c*(resolution - 1)/2+(resolution + 1)/2)
        direction = 1
    # back
    if z_c < 0 and z_c < -abs(x_c) and z_c < -abs(y_c):
        x_i = int(-y_c/z_c*(resolution - 1)/2+(resolution + 1)/2)
        y_i = int(x_c/z_c*(resolution - 1)/2+(resolution + 1)/2)
        direction = 2
    # left
    if x_c < 0 and x_c < -abs(z_c) and x_c < -abs(y_c):
        x_i = int(-y_c/x_c*(resolution - 1)/2+(resolution + 1)/2)
        y_i = int(-z_c/x_c*(resolution - 1)/2+(resolution + 1)/2)
        direction = 3
    # right
    if x_c > 0 and x_c > abs(y_c) and x_c > abs(z_c):
        x_i = int(y_c/x_c*(resolution - 1)/2+(resolution + 1)/2)
        y_i = int(-z_c/x_c*(resolution - 1)/2+(resolution + 1)/2)
        direction = 4

    return x_i, y_i, direction

def lla_to_cc(lat,lon,alt,intensity):
    Img_front = np.zeros((resolution,resolution))
    Img_back = np.zeros((resolution,resolution))
    Img_left = np.zeros((resolution,resolution))
    Img_right = np.zeros((resolution,resolution))
    Img_front_with_intensity = np.zeros((resolution,resolution))
    Img_back_with_intensity = np.zeros((resolution,resolution))
    Img_left_with_intensity = np.zeros((resolution,resolution))
    Img_right_with_intensity = np.zeros((resolution,resolution))
    x, y, z, cos_phi, sin_phi, cos_lambda, sin_lambda = lla_to_ecef(lat,lon,alt)
    e, n, u = ecef_to_enu(x,y,z,cos_phi, sin_phi, cos_lambda, sin_lambda)
    x_c, y_c, z_c = enu_to_cc(e, n, u)
    x_i, y_i, direction = cc_to_ic(x_c, y_c, z_c)
    if direction == 1:
        Img_front[x_i][y_i] = 255
        Img_front_with_intensity[x_i][y_i] = intensity
    if direction == 2:
        Img_back[x_i][y_i] = 255
        Img_back_with_intensity[x_i][y_i] = intensity
    if direction == 3:
        Img_left[x_i][y_i] = 255
        Img_left_with_intensity[x_i][y_i] = intensity
    if direction == 4:
        Img_right[x_i][y_i] = 255
        Img_right_with_intensity[x_i][y_i] = intensity

    cv2.imwrite('front.png',Img_front)
    cv2.imwrite('back.png',Img_back)
    cv2.imwrite('right.png',Img_right)
    cv2.imwrite('left.png',Img_left)
    cv2.imwrite('front_with_intensity.png',Img_front_with_intensity)
    cv2.imwrite('back_with_intensity.png',Img_back_with_intensity)
    cv2.imwrite('right_with_intensity.png',Img_right_with_intensity)
    cv2.imwrite('left_with_intensity.png',Img_left_with_intensity)

if __name__ == "__main__":
    pass