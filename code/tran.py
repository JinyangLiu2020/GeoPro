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

def transformpoint(lat,lon,alt):
    x,y,z,cos_phi, sin_phi, cos_lambda, sin_lambda = lla_to_ecef(lat, lon, alt)
    e,n,u = ecef_to_enu(x,y,z,cos_phi, sin_phi, cos_lambda, sin_lambda)
    return e,n,u
