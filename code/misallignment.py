def pointwise(point_cloud,rgb):
    x = rgb[0]-point_cloud[0]
    y = rgb[1]-point_cloud[1]
    return [x,y]
    
def linewise(point_cloud,rgb):
    mid_point = [1024,point_cloud[0]*1024+point_cloud[1]]
    mid_rgb = [1024,rgb[0]*1024+rgb[1]]
    shift = pointwise(mid_point,mid_rgb)