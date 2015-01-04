#!/usr/bin/env python

#a Imports
import PIL.Image
import PIL.ImageOps
import colorsys
import math
import array

#a Test vectors
rows = ( (0,0,3,4,5,6,7,8),
         (5,6,7,5,2,3,1,4),
         (4,6,2,3,8,9,1,2),
         (2,5,6,1,2,3,1,2),
         (9,8,7,6,5,4,3,2)
         )

w = len(rows[0])
h = len(rows)
vector_dx = array.array('d',range(w*h))
vector_dy = array.array('d',range(w*h))
solved = array.array('d',range(w*h))

for y in range(h-1):
    for x in range(w-1):
        a0 = rows[y][x]+0.0
        a1 = rows[y][x+1]+0.0
        b0 = rows[y+1][x]+0.0
        b1 = rows[y+1][x+1]+0.0
        vector_dx[y*w+x] = b1+a1-b0-a0
        vector_dy[y*w+x] = b1-a1+b0-a0
        pass
    pass

#a Load image
img_filename = "/Users/gstark_old/gc_vec.png"

im = PIL.Image.open(img_filename)
(w,h) = im.size
vector_dx = array.array('d',range(w*h))
vector_dy = array.array('d',range(w*h))
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
        pass
    pass

#a Solving functions
#f solve_iteratively
def solve_iteratively( vectors, solved, w, h, scale, num_passes ):
    (vector_dx, vector_dy) = vectors
    deltas = array.array('d',range(w*h))
    for i in range(h*w): solved[i] = 0
    def do_pass( scale, w=w, h=h, deltas=deltas, solved=solved, vector_dx=vector_dx, vector_dy=vector_dy ):
        for i in range(h*w): deltas[i]=0
        for y in range(h-1):
            for x in range(w-1):
                a0 = solved[y*w+x]
                a1 = solved[y*w+x+1]
                b0 = solved[(y+1)*w+x]
                b1 = solved[(y+1)*w+x+1]
                (sdx, sdy) = (b1+a1-b0-a0, b1-a1+b0-a0)
                (dx,dy) = (vector_dx[y*w+x], vector_dy[y*w+x])
                dx -= sdx
                dy -= sdy
                dxy1 = (dx+dy)
                dxy2 = (dx-dy)
                deltas[y*w+x]     -= dxy1
                deltas[(y+1)*w+x+1] += dxy1
                deltas[y*w+x+1]   += dxy2
                deltas[(y+1)*w+x]   -= dxy2
                pass
            pass
        for y in range(h):
            deltas[y*w+0] *= 2
            deltas[y*w+w-1] *= 2
            pass
        for x in range(w):
            deltas[0*w+x] *= 2
            deltas[(h-1)*w+x] *= 2
            pass
        for y in range(h):
            for x in range(w):
                solved[y*w+x] += deltas[y*w+x]*scale
                pass
            pass
        s0 = solved[0*w+0]
        s1 = solved[0*w+1]
        for y in range(h):
            for x in range(w):
                if ((x+y)&1)==1:
                    solved[y*w+x] -= s0+s1
                else:
                    solved[y*w+x] -= s0
                pass
            pass
        pass
    for p in range(num_passes):
        do_pass(scale=0.1)
        pass
    pass

#f solve_top_down
def solve_top_down( vectors, solved, w, h ):
    """
    Note that if we have
    a0 a1
    b0 b1
    then
    dx0 = b1+a1-b0-a0
    dy0 = b1-a1+b0-a0
    dx0+dy0 = 2*b1 - 2*a0
    dx0-dy0 = 2*a1 - 2*b0
    b0 = a1 + (dy0-dx0)/2
    b1 = a0 + (dy0+dx0)/2

    If we have the rows as:
    0  0  a2 a3 a4 ...
    b0 b1 b2 b3 b4 ...
    c0 c1 c2 c3 c4 ...

    then b1=(dy0+dx0)/2 and b0=(dy0-dx0)/2

    So, if we have a1 and b1, we can go across the top row and work out a2/b2

    dxn = bn+1 + an+1 - an - bn
    dyn = bn+1 - an+1 - an + bn

    dxn+dyn = 2*bn+1 - 2*an
    dxy-dyn = 2*an+1 - 2*bn
    an+1 = bn + (dxn-dyn)/2
    bn+1 = an + (dxn+dyn)/2

    So we can work out the whole of row a/b

    Then we can evaluate c0&c1, and c1&c2, and c2&c3, and so on
    Of course we get two values for c1 - so we can average?

    Then we can evaluate row d
    """
    (vector_dx, vector_dy) = vectors
    for i in range(h*w): solved[i] = 0
    def solve_bottom_of_square( x, y, average_left=False, solved=solved, w=w, h=h ):
        (a0,a1) = (solved[y*w+x], solved[y*w+x+1] )
        (dx,dy) = (vector_dx[y*w+x], vector_dy[y*w+x])
        b0 = a1 + (dy-dx)/2
        b1 = a0 + (dy+dx)/2
        if average_left:
            solved[(y+1)*w+x] = (solved[(y+1)*w+x] + b0)/2.0
        else:
            solved[(y+1)*w+x] = b0
        solved[(y+1)*w+x+1] = b1
        pass
    def solve_right_of_square( x, y, average_top=False, solved=solved, w=w, h=h ):
        (a0,b0) = (solved[y*w+x], solved[(y+1)*w+x] )
        (dx,dy) = (vector_dx[y*w+x], vector_dy[y*w+x])
        a1 = b0 + (dx-dy)/2
        b1 = a0 + (dx+dy)/2
        if average_top:
            solved[y*w+x+1] = (solved[y*w+x+1] + a1)/2.0
            pass
        else:
            solved[y*w+x+1] = a1
            pass
        solved[(y+1)*w+x+1] = b1
        pass
    # First ensure a0,a1,b0,b1 are correct
    solve_bottom_of_square( 0,0 )
    # Now to top row
    for x in range(w-2):
        solve_right_of_square( x+1,0 )
        pass
    # Now top down
    for y in range(h-2):
        for x in range(w-1):
            solve_bottom_of_square( x, y+1, average_left=(x>0) )
            pass
        pass
    pass


#a Test
#solve_iteratively( (vector_dx, vector_dy), solved, w=w, h=h, scale=0.1, num_passes=100 )

solve_top_down( (vector_dx, vector_dy), solved, w=w, h=h )

if px is not None:
    min=solved[0]
    max=solved[0]
    for i in range(w*h):
        if solved[i]>max: max=solved[i]
        if solved[i]<min: min=solved[i]
        pass
    scale = 255/(max-min)
    for y in range(h):
        for x in range(w):
            s = int((solved[y*w+x]-min)*scale)
            if (s<0): s=0
            if (s>255): s=255
            px[x,y] = (s,s,s)
            pass
        pass
    im.save(fp="/Users/gstark_old/gc_unvec.png",format="PNG")
    pass
blah



