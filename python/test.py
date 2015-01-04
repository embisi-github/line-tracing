#!/usr/bin/env python

import PIL.Image
import PIL.ImageOps
import colorsys
import math

img_filename = "./gc.jpg"

region = None
resize = None

#region = (350,870,700,1070)
#region = (0,0,1900,1000)
resize = (960,540)
#resize = (500,300)
#resize = (250,150)

max_divergence = 0.1
# do not need to use this max_divergence = 0.25
max_divergence = 0.05

max_lines = 1
#max_lines = 3
#max_lines = 20
max_lines = 40
#max_lines = 140

min_line_distance = 20
min_line_distance = 10 # Use for 960/540 image
min_line_distance = 5 # Use for 250/150 image
#min_line_distance = 2

# do not need this step_scale = 0.8
step_scale = 0.8
#step_scale = 1.2 # BEST BY FAR
#step_scale = 1.4 # Perhaps as good as 1.2
#step_scale = 2.0 # Try for large image NOT AS GOOD

# max_divergence = sin(angle)
# 5 degrees => max_divergence = 0.08
# 10 degrees => max_divergence = 0.17

#max_lines = 250
#min_line_distance = 5

import image_lib
img = image_lib.c_rw_image( min_line_distance=min_line_distance )
img.load_jpeg( jpeg_filename=img_filename, resize=resize, region=region )
img.vectorize( method="div8" ) # Was div8
img.save_grayscale( "gc_gray.png" )
img.save_vector(    "gc_vec_hsv2.png", method="hsv2")
if False:
    for p in (32,):# (1,2,4,8,16,32,64):
        img.solve_iteratively( scale=0.1, num_passes=p )
        img.generate_resolved_image()
        img.save_resolved(  "gc_unvec_%d.png"%p )
        pass
    pass
else:
    img.solve_top_down()
    img.generate_resolved_image()
    img.save_resolved(  "gc_unvec.png" )
    pass

p = 0
while (p<max_lines):
    (x,y) = img.find_greatest_gradient_avoiding_lines()
    if x is None:
        break
    for rot in (1,-1):
        (cx,cy,v) = img.find_centre_of_gradient(x,y)
        (ncx,ncy) = (cx,cy)
        line = img.new_line( cx, cy, v )
        while True:
            (ncx,ncy) = img.find_next_point_along_contour( v, cx, cy,
                                                           scale = step_scale,
                                                           max_steps = 1000,
                                                           rot=rot,
                                                           max_divergence=max_divergence,
                                                           len_power=1.0)
            if ncx is None: break
            line.add_point( ncx, ncy )
            (cx,cy) = (ncx,ncy)
            pass
        line.complete()
        pass
    img.line_set.export_distance_array_image("line_set_distance_array.png")
    p += 1
    pass
img.line_set.export_image("line_set.png")
