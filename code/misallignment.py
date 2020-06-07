import math


def pointwise(point_cloud,rgb):
    x = rgb[0]-point_cloud[0]
    y = rgb[1]-point_cloud[1]
    return [x,y]
    
def linewise(point_cloud,rgb):
    mid_point = [1024,point_cloud[0]*1024+point_cloud[1]]
    mid_rgb = [1024,rgb[0]*1024+rgb[1]]
    end_point = [2048,point_cloud[0]*2048+point_cloud[1]]
    end_rgb = [2048,rgb[0]*2048+rgb[1]]
    angle1 = math.atan2(1024,end_point[1]-mid_point[1])
    angle2 = math.atan2(1024,end_rgb[1]-mid_rgb[1])
    angledif = angle2-angle1
    new_mid_point = [1024,rgb[0]*1024+point_cloud[1]]
    shift = pointwise(new_mid_point,mid_rgb)
    return shift, angledif
    #shift = pointwise(mid_point,mid_rgb)