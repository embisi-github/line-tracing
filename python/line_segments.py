#!/usr/bin/env python

#a Imports
import PIL.Image
import PIL.ImageOps
import colorsys
import math
import array

#a Test vectors
pt = (662.5,883.5)

#a Line segment classes
class c_vector( object ):
    def __init__( self, dx, dy ):
        self.dx = dx
        self.dy = dy
        pass
    def copy( self ):
        return c_vector( self.dx, self.dy )
    def add( self, other, scale_self=1, scale_other=1 ):
        return c_vector( self.dx*scale_self+other.dx*scale_other,
                         self.dy*scale_self+other.dy*scale_other )
    def scale( self, scale ):
        self.dx *= scale
        self.dy *= scale
        return
    def dot( self, other ):
        return self.dx*other.dx + self.dy*other.dy
    def normal( self ):
        return c_vector( -self.dy, self.dx )
    def length( self ):
        return math.sqrt( self.dx*self.dx + self.dy*self.dy )
    def normalize( self ):
        l = self.length()
        if (l>0.000000001):
            self.dx = self.dx/l
            self.dy = self.dy/l
            return
        self.dx = 0
        self.dy = 0
        return
    def __repr__( self ):
        return "(%4.2f,%4.2f)"%(self.dx,self.dy)

class c_straight_line_segment( object ):
    def __init__( self, centre, direction, length=None ):
        if length is None:
            length = direction.length()
            print direction, length
            direction = direction.copy()
            direction.normalize()
            pass
        self.centre = centre
        self.direction = direction
        self.length = length
        pass
    def check_parallel( self, other ):
        cos_angle = self.direction.dot(other.direction)
        metric_angle = (1-cos_angle)*(1-cos_angle)
        separation = self.centre.add( other=other.centre, scale_other=-1 )
        l = separation.length()
        self_normal = self.direction.normal()
        other_normal = other.direction.normal()
        normals = self_normal.add( other=other_normal )
        metric_blah
        # Do we want to have separation involved? Do we want a 'number of merges'?
        return (metric_angle, l, )
    def check_joins_up_to( self, next ):
        s_end = self.centre.add( other=self.direction, scale_other=self.length/2.0 )
        n_end = next.centre.add( other=next.direction, scale_other=-next.length/2.0 )
        separation = n_end.add( other=s_end, scale_other=-1 )
        return separation.length()
    def can_merge_with( self, next ):
        # separation = next - self; this should be self.length*self.direction AND it should be -next.length*next.direction
        separation = next.centre.add( other=self.centre, scale_other=-1 )
        print separation
        print separation.dot(next.direction), separation.dot(self.direction)
        da = separation.dot(next.direction)-next.length
        db = separation.dot(self.direction)-self.length
        print da, db
        return da*da+db*db
    def __repr__( self ):
        return "line %s len %4.2f dirn %s"%(str(self.centre),self.length,str(self.direction))

test_lines = ( (0,0,1,1), (1,1,3,3), (3.1,3,4,5), (0.5,0.5,1.5,1.5) )
lines = []
for (x0,y0,x1,y1) in test_lines:
    centre = c_vector( (x0+x1)/2.0, (y0+y1)/2.0 )
    direction = c_vector( (x1-x0)*1.0, (y1-y0)*1.0 )
    lines.append( c_straight_line_segment( centre=centre, direction=direction ) )
    pass
for i in range(len(lines)):
    print i, lines[i]
    pass

for i in range(len(lines)-1):
    print i, lines[i].check_joins_up_to( lines[i+1] )

#for i in range(len(lines)-1):
#    print i+1, lines[0].check_parallel( lines[i+1] )


# x [300,500]; y [870;1070]
img_filename = "/Users/gstark_old/gc_vec.png"
im = PIL.Image.open(img_filename)
(w,h) = im.size
vector_dx = array.array('d',range(w*h))
vector_dy = array.array('d',range(w*h))
vector_sz    = array.array('d',range(w*h))
vector_angle = array.array('d',range(w*h))
solved = array.array('d',range(w*h))
px = im.load()
scale = 1/256.0
for y in range(h):
    for x in range(w):
        (r,g,b) = px[x,y]
        d = (r/255.0*2*3.14159)
        l = (g<<8)|b
        #if (l<512): l=0
        dx = l*math.cos(d)*scale
        dy = l*math.sin(d)*scale
        vector_dx[y*w+x] = dx
        vector_dy[y*w+x] = dy
        vector_sz[y*w+x] = l*scale
        vector_angle[y*w+x] = (d*180.0/3.14159265)
        pass
    pass

(yt,yb)=(870,1070)
(xl,xr)=(350,700)
for yy in range(yb-yt):
    y = yy+yt
    for xx in range(xr-xl):
        x = xx+xl
        if vector_sz[y*w+x] > 10.0:
            print x,y,vector_sz[y*w+x],vector_angle[y*w+x]
            pass
        pass
    pass
