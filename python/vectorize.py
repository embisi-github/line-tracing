#!/usr/bin/env python

import PIL.Image
import PIL.ImageOps
import colorsys
import math

img_filename = "/Users/gstark_old/Library/Containers/com.apple.mail/Data/Library/Mail Downloads/43650F52-B756-4F29-B92B-6CB553BF2A77/Photo of Markups.JPG"
img_filename = "./gc.jpg"

import image_lib
img = image_lib.c_rw_image()
img.load_jpeg( jpeg_filename=img_filename, resize=(120,100) )
img.vectorize()
img.save_grayscale( "gc_gray.png" )
img.save_vector(    "gc_vec_hsv2.png", method="hsv2")
for p in (1,2,4,8):#,16,32,64):
    img.solve_iteratively( scale=0.1, num_passes=p )
    img.generate_resolved_image()
    img.save_resolved(  "gc_unvec_%d.png"%p )
    pass
img.solve_top_down()
img.generate_resolved_image()
img.save_resolved(  "gc_unvec.png" )
blah


#@im = PIL.Image.open("/Users/gstark_old/Library/Containers/com.apple.mail/Data/Library/Mail Downloads/43650F52-B756-4F29-B92B-6CB553BF2A77/Photo of Markups.JPG")
im = PIL.Image.open(img_filename)

w=600
h=400

(w,h)=im.size
im2 = im.resize(size=(w,h), resample=PIL.Image.BICUBIC )
#im2.show()
im2g = PIL.ImageOps.grayscale( im2 )
#im2.show()

im2_copy = im2.copy()

class rw_image( object ):
    min_x = 4
    min_y = 4
    def __init__( self, image ):
        self.image = image
        self.px = image.load()
        (self.w, self.h) = image.size
        self.div = self.div8
        self.set_div = self.set_div_hsv

        self.div = self.div2
        self.set_div = self.set_div_vec
        #self.set_div = self.set_div_hsv

        self.div = self.div8
        self.set_div = self.set_div_hsv2
        pass
    def val( self, x, y ):
        if (x<self.min_x) or (y<self.min_y): return 0
        if (x>=self.w) or (y>=self.h): return 0
        return self.px[x,y]
    def set( self, x, y, v ):
        self.px[x,y] = v
        pass
    def set_div_hue_dirn( self, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        if (d<0): d+=360
        r = (d/360.0)
        g = r
        b = r
        self.set(x,y,( int(r*255), int(g*255), int(b*255) ))
        pass
    def set_div_hue_mag( self, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        r = (l/32.0)
        if r>1.0:r=1.0
        g = r
        b = r
        self.set(x,y,( int(r*255), int(g*255), int(b*255) ))
        pass
    def set_div_hsv2( self, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        if (d<0): d+=360
        l = l/32.0
        if (l<0.2): l=0
        if (l>1.0): l=1.0
        (r,g,b) = colorsys.hsv_to_rgb( d/360.0, 1.0, l )
        self.set(x,y,( int(r*255), int(g*255), int(b*255) ))
        pass
    def set_div_hsv( self, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        if (d<0): d+=360
        (r,g,b) = colorsys.hsv_to_rgb( d/360.0, 1.0, math.pow(l/256.0,0.2))
        self.set(x,y,( int(r*255), int(g*255), int(b*255) ))
        pass
    def set_div_vec( self, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        if (d<0): d+=360
        d = int(d/360.0*256)
        l = int(l*256.0)
        self.set(x,y,(d,(l>>8)&0xff,(l&0xff)))
        pass
    def div2( self, x, y ):
        dx = self.val(x+1,y)-self.val(x-1,y)
        dy = self.val(x,y+1)-self.val(x,y-1)
        return (dx/2.0,dy/2.0)
    def div4( self, x, y ):
        dx = self.val(x+1,y)+self.val(x+1,y+1)-self.val(x,y)-self.val(x,y+1)
        dy = self.val(x,y+1)+self.val(x+1,y+1)-self.val(x,y)-self.val(x+1,y)
        return (dx/4.0,dy/4.0)
    def div8( self, x, y ):
        dx = self.val(x+1,y)-self.val(x-1,y) + (
            0.707*(self.val(x+1,y-1)+self.val(x+1,y+1)-self.val(x-1,y-1)-self.val(x-1,y+1)) )
        dy = self.val(x,y+1)-self.val(x,y-1) + (
            0.707*(self.val(x-1,y+1)+self.val(x+1,y+1)-self.val(x-1,y-1)-self.val(x+1,y-1)) )
        return (dx/4.8,dy/4.8)
    pass

px   = rw_image(im2_copy)
pxg  = rw_image(im2g)

for x in range(w):
    for y in range(h):
        (dx,dy) = pxg.div( x, y )
        l = math.sqrt(dx*dx+dy*dy)
        d = math.atan2(dy,dx)/(3.1416/180.0)
        px.set_div(x,y,l,d)
        pass
    pass

#im2_copy.show()
im2_copy.save(fp="/Users/gstark_old/gc_vec_hsv2.png",format="PNG")
