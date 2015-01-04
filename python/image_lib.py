#a Imports
import PIL.Image
import PIL.ImageOps
import colorsys
import math
import array
import time

#a Classes
#c c_line_set
class c_line_set(object):
    def __init__( self, w, h, min_distance=8 ):
        self.w = w
        self.h = h
        self.distance_array = array.array('d',range(self.w*self.h))
        for i in range(self.w*self.h): self.distance_array[i]=self.w*self.h
        self.min_distance = min_distance
        self.lines = []
        pass
    def add_line( self, line ):
        self.lines.append(line)
        pass
    def is_closer_than( self, x, y ):
        return (self.distance_array[int(y)*self.w+int(x)]<self.min_distance)
    def completed_line( self, l ):
        if l.num_points()==1:
            self.added_point( l.point(0) )
            return
        for i in range(l.num_points()-1):
            self.added_segment( l.point(i), l.point(i+1) )
            pass
        pass
    def added_point( self, pt ):
        (x,y) = pt
        for dx in range( self.min_distance*2+1 ):
            mx = x-self.min_distance+dx
            for dy in range( self.min_distance*2+1 ):
                my = y-self.min_distance+dy
                self.add_distance(mx,my,from_point=(x,y))
                pass
            pass
        pass
    def added_segment( self, pt0, pt1 ):
        dx = pt1[0]-pt0[0]
        dy = pt1[1]-pt0[1]
        l = math.sqrt(dx*dx+dy*dy)
        for i in range(int(l)+1):
            pt = ( pt0[0]+(dx*i)/l, pt0[1]+(dy*i)/l )
            self.added_point( pt )
            pass
        pass
    def add_distance( self, x,y, from_point=None ):
        if (x<0) or (y<0): return
        if (x>=self.w) or (y>=self.h): return
        self.distance_array[int(y)*self.w+int(x)] = 0
        pass
    def export_distance_array_image( self, filename ):
        image = PIL.Image.new("L",(self.w,self.h),0)
        px = image.load()
        for x in range(self.w):
            for y in range(self.h):
                d = self.distance_array[y*self.w+x]
                if (d<0):d=0
                if (d>255):d=255
                px[x,y]=d
                pass
            pass
        image.save(filename)
        pass
    def export_image( self, filename ):
        image = PIL.Image.new("RGB",(self.w,self.h),(0,0,0))
        px = image.load()
        hue = 0
        for l in self.lines:
            (r,g,b) = colorsys.hsv_to_rgb( hue/360.0, 1.0, 1.0 )
            r = int(r*255)
            g = int(g*255)
            b = int(b*255)
            n = l.num_points()
            for i in range(n-1):
                (sx,sy) = l.point(i)
                (dx,dy) = l.point(i+1)
                dx -= sx
                dy -= sy
                for j in range(40):
                    x = int(sx+(dx*j)/40.0)
                    y = int(sy+(dy*j)/40.0)
                    px[x,y]=(r,g,b)
                    pass
                pass
            (x,y) = l.point(n-1)
            x=int(x)
            y=int(y)
            px[x,y]=(r,g,b)
            hue += 3
            pass
        image.save(filename)
        pass
    pass

#c c_line
class c_line(object):
    def __init__( self, line_set, x,y, v ):
        self.line_set = line_set
        self.v = v
        self.pts = []
        self.add_point( x,y )
        pass
    def num_points( self ):
        return len(self.pts)
    def point( self, n ):
        return self.pts[n]
    def add_point( self, x, y ):
        self.pts.append( (x,y) )
        pass
    def complete( self ):
        self.line_set.completed_line(self)
    pass

#c c_rw_image
class c_rw_image( object ):
    min_x = 4
    min_y = 4
    #f __init__
    def __init__( self, jpeg_filename=None, min_line_distance=20 ):
        self.image = None
        self.grayscale_image = None
        self.vector_dx = None
        self.vector_dy = None
        self.resolved = None
        self.resolved_image = None
        self.line_set = None
        self.min_line_distance = min_line_distance

        if jpeg_filename is not None:
            self.load_jpeg(jpeg_filename)
            pass

        self.init_time = time.time()
        self.info_stack = []

        pass
    #f info_msg
    def info_msg( self, msg ):
        print msg
        pass
    #f info_start
    def info_start( self, msg ):
        t = time.time()-self.init_time
        self.info_msg( "Info start (@%4.2f):%s"%(t,msg) )
        self.info_stack.append( (t,msg) )
        pass
    #f info_end
    def info_end( self ):
        et = time.time()-self.init_time
        (st,msg) = self.info_stack.pop()
        t = et-st
        self.info_msg( "Info done (took %4.2fs):(@ %4.2f):%s"%(t,et,msg) )
        pass
    #f load_jpeg
    def load_jpeg( self, jpeg_filename, resize=None, region=None ):
        """
        resize is a (w,h) to scale the image down to initially
        region is a (lx,ty,rx,by) region inside the scaled image to use for analysis
        """
        image = PIL.Image.open(jpeg_filename)
        self.image = image
        scaled_image = image
        self.info_msg("Jpeg size %s"%(str(image.size)))
        if resize is not None:
            scaled_image = image.resize(size=resize, resample=PIL.Image.BICUBIC )
            pass
        cropped_image = scaled_image
        if region is not None:
            cropped_image = scaled_image.crop(region)
            x = cropped_image.load()
            pass
        grayscale_image = PIL.ImageOps.grayscale( cropped_image )
        self.info_msg("Grayscale image size %s"%(str(grayscale_image.size)))
        self.grayscale_image = grayscale_image
        self.grayscale_px = grayscale_image.load()
        (self.w, self.h) = grayscale_image.size
        self.area = self.w*self.h
        pass
    #f vectorize
    def vectorize( self, method="div4" ):
        div = {"div2":self.div2,
               "div4":self.div4,
               "div8":self.div8,
               }[method]
        self.vector_dx = array.array('d',range(self.area))
        self.vector_dy = array.array('d',range(self.area))
        self.info_start("Vectorizing")
        for x in range(self.w):
            for y in range(self.h):
                (self.vector_dx[y*self.w+x],self.vector_dy[y*self.w+x]) = div( x, y )
                pass
            pass
        self.info_end()
        pass
    #f save_grayscale
    def save_grayscale( self, grayscale_filename ):
        self.info_start("Saving grayscale image")
        self.grayscale_image.save(grayscale_filename)
        self.info_end()
        pass
    #f generate_resolved_image
    def generate_resolved_image( self ):
        self.info_start("Generate resolved image")
        self.resolved_image = PIL.Image.new("L",(self.w,self.h),0)
        self.line_set = c_line_set( self.w, self.h, min_distance=self.min_line_distance )
        px = self.resolved_image.load()
        self.info_start("Find min/max")
        min=self.resolved[0]
        max=self.resolved[0]
        for i in range(self.area):
            if self.resolved[i]>max: max=self.resolved[i]
            if self.resolved[i]<min: min=self.resolved[i]
            pass
        self.info_end()
        scale = 255/(max-min)
        for y in range(self.h):
            for x in range(self.w):
                s = int((self.resolved[y*self.w+x]-min)*scale)
                if (s<0): s=0
                if (s>255): s=255
                px[x,y] = s
                pass
            pass
        self.info_end()
        pass
    #f save_resolved
    def save_resolved( self, resolved_filename ):
        self.info_start("Saving resolved image")
        self.resolved_image.save(resolved_filename)
        self.info_end()
        pass
    #f save_vector
    def save_vector( self, vector_filename, method="vec" ):
        set_div = {"hsv":self.set_div_hsv,
                   "hsv2":self.set_div_hsv2,
                   "hue_mag":self.set_div_hue_mag,
                   "hue_dirn":self.set_div_hue_dirn,
                   "vec":self.set_div_vec,
                   }[method]
        self.info_start("Saving vector image with conversion")
        image  = PIL.Image.new("RGB",(self.w,self.h),(0,0,0))
        px = image.load()
        for x in range(self.w):
            for y in range(self.h):
                (dx,dy) = (self.vector_dx[y*self.w+x],self.vector_dy[y*self.w+x])
                l = math.sqrt(dx*dx+dy*dy)
                d = math.atan2(dy,dx)/(3.1416/180.0)
                set_div(px,x,y,l,d)
                pass
            pass
        self.info_end()
        image.save(fp=vector_filename)
        pass
    #f val
    def val( self, x, y ):
        if (x<self.min_x) or (y<self.min_y): return 0
        if (x>=self.w) or (y>=self.h): return 0
        return self.grayscale_px[x,y]
    #f set_div_hue_dirn
    def set_div_hue_dirn( self, px, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        if (d<0): d+=360
        r = (d/360.0)
        g = r
        b = r
        px[x,y] = ( int(r*255), int(g*255), int(b*255) )
        pass
    #f set_div_hue_mag
    def set_div_hue_mag( self, px, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        r = (l/32.0)
        if r>1.0:r=1.0
        g = r
        b = r
        px[x,y] = ( int(r*255), int(g*255), int(b*255) )
        pass
    #f set_div_hsv2
    def set_div_hsv2( self, px, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        if (d<0): d+=360
        l = l/32.0
        if (l<0.2): l=0
        if (l>1.0): l=1.0
        (r,g,b) = colorsys.hsv_to_rgb( d/360.0, 1.0, l )
        px[x,y] = ( int(r*255), int(g*255), int(b*255) )
        pass
    #f set_div_hsv
    def set_div_hsv( self, px, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        if (d<0): d+=360
        (r,g,b) = colorsys.hsv_to_rgb( d/360.0, 1.0, math.pow(l/256.0,0.2))
        px[x,y] = ( int(r*255), int(g*255), int(b*255) )
        pass
    #f set_div_vec
    def set_div_vec( self, px, x, y, l, d ):
        """
        l is 0 to 256; d is -360 to 360
        """
        if (d<0): d+=360
        d = int(d/360.0*256)
        l = int(l*256.0)
        px[x,y] = (d,(l>>8)&0xff,(l&0xff))
        pass
    #f div2
    def div2( self, x, y ):
        """
        A divergence (vector gradient) function accounting for EW and NS of xy, ignoring xy
        """
        dx = self.val(x+1,y)-self.val(x-1,y)
        dy = self.val(x,y+1)-self.val(x,y-1)
        return (dx/2.0,dy/2.0)
    #f div4
    def div4( self, x, y ):
        """
        A divergence (vector gradient) function accounting for 4 directions E, S, SE from xy
        """
        dx = self.val(x+1,y)+self.val(x+1,y+1)-self.val(x,y)-self.val(x,y+1)
        dy = self.val(x,y+1)+self.val(x+1,y+1)-self.val(x,y)-self.val(x+1,y)
        return (dx/4.0,dy/4.0)
    #f div8
    def div8( self, x, y ):
        """
        A divergence (vector gradient) function accounting for 8 directions NSEW from xy
        """
        dx = self.val(x+1,y)-self.val(x-1,y) + (
            0.707*(self.val(x+1,y-1)+self.val(x+1,y+1)-self.val(x-1,y-1)-self.val(x-1,y+1)) )
        dy = self.val(x,y+1)-self.val(x,y-1) + (
            0.707*(self.val(x-1,y+1)+self.val(x+1,y+1)-self.val(x-1,y-1)-self.val(x+1,y-1)) )
        return (dx/4.8,dy/4.8)
    pass


    #f solve_iteratively
    def solve_iteratively( self, scale, num_passes ):
        self.info_start( "Solve iteratively %d passes"%num_passes )
        self.resolved = array.array('d',range(self.area))
        self.resolved_image = None
        deltas = array.array('d',range(self.area))
        for i in range(self.area): self.resolved[i] = 0
        def do_pass( scale, deltas=deltas, self=self ):
            for i in range(self.area): deltas[i]=0
            for y in range(self.h-1):
                for x in range(self.w-1):
                    a0 = self.resolved[y*self.w+x]
                    a1 = self.resolved[y*self.w+x+1]
                    b0 = self.resolved[(y+1)*self.w+x]
                    b1 = self.resolved[(y+1)*self.w+x+1]
                    (sdx, sdy) = (b1+a1-b0-a0, b1-a1+b0-a0)
                    (dx,dy) = (self.vector_dx[y*self.w+x], self.vector_dy[y*self.w+x])
                    dx -= sdx
                    dy -= sdy
                    dxy1 = (dx+dy)
                    dxy2 = (dx-dy)
                    deltas[y*self.w+x]     -= dxy1
                    deltas[(y+1)*self.w+x+1] += dxy1
                    deltas[y*self.w+x+1]   += dxy2
                    deltas[(y+1)*self.w+x]   -= dxy2
                    pass
                pass
            for y in range(self.h):
                deltas[y*self.w+0] *= 2
                deltas[y*self.w+self.w-1] *= 2
                pass
            for x in range(self.w):
                deltas[0*self.w+x] *= 2
                deltas[(self.h-1)*self.w+x] *= 2
                pass
            for y in range(self.h):
                for x in range(self.w):
                    self.resolved[y*self.w+x] += deltas[y*self.w+x]*scale
                    pass
                pass
            s0 = self.resolved[0*self.w+0]
            s1 = self.resolved[0*self.w+1]
            for y in range(self.h):
                for x in range(self.w):
                    if ((x+y)&1)==1:
                        self.resolved[y*self.w+x] -= s0+s1
                    else:
                        self.resolved[y*self.w+x] -= s0
                    pass
                pass
            pass
        for p in range(num_passes):
            self.info_start("Iterative pass %d"%p)
            do_pass(scale=0.1)
            self.info_end()
            pass
        self.info_end()
        pass
    
    #f solve_top_down
    def solve_top_down( self ):
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
        self.resolved = array.array('d',range(self.area))
        self.resolved_image = None
        for i in range(self.area): self.resolved[i] = 0
        def solve_bottom_of_square( x, y, average_left=False, self=self ):
            (a0,a1) = (self.resolved[y*self.w+x], self.resolved[y*self.w+x+1] )
            (dx,dy) = (self.vector_dx[y*self.w+x], self.vector_dy[y*self.w+x])
            b0 = a1 + (dy-dx)/2
            b1 = a0 + (dy+dx)/2
            if average_left:
                self.resolved[(y+1)*self.w+x] = (self.resolved[(y+1)*self.w+x] + b0)/2.0
                pass
            else:
                self.resolved[(y+1)*self.w+x] = b0
                pass
            self.resolved[(y+1)*self.w+x+1] = b1
            pass
        def solve_right_of_square( x, y, average_top=False, self=self ):
            (a0,b0) = (self.resolved[y*self.w+x], self.resolved[(y+1)*self.w+x] )
            (dx,dy) = (self.vector_dx[y*self.w+x], self.vector_dy[y*self.w+x])
            a1 = b0 + (dx-dy)/2
            b1 = a0 + (dx+dy)/2
            if average_top:
                self.resolved[y*self.w+x+1] = (self.resolved[y*self.w+x+1] + a1)/2.0
                pass
            else:
                self.resolved[y*self.w+x+1] = a1
                pass
            self.resolved[(y+1)*self.w+x+1] = b1
            pass
        # First ensure a0,a1,b0,b1 are correct
        solve_bottom_of_square( 0,0 )
        # Now to top row
        for x in range(self.w-2):
            solve_right_of_square( x+1,0 )
            pass
        # Now top down
        for y in range(self.h-2):
            for x in range(self.w-1):
                solve_bottom_of_square( x, y+1, average_left=(x>0) )
                pass
            pass
        pass
    #f find_greatest_gradient
    def find_greatest_gradient( self ):
        self.info_start("Find greatest gradient")
        (p,max,maxsq) = (None,-1000,-1000)
        for i in range(self.area):
            if ((i%self.w)<=self.min_x): continue
            if ((i/self.w)<=self.min_y): continue
            if self.vector_dx[i]>max:
                d = self.vector_dx[i]*self.vector_dx[i] + self.vector_dy[i]*self.vector_dy[i]
                if d>maxsq:
                    max = math.sqrt(d)
                    maxsq = d
                    p = i
                    pass
                pass
            pass
        x = p % self.w
        y = p / self.w
        self.info_msg("Greatest gradient at (%d,%d) of %4.2f"%(x,y,max))
        self.info_end()
        return (x,y)
    #f find_greatest_gradient_avoiding_lines
    def find_greatest_gradient_avoiding_lines( self ):
        self.info_start("Find greatest gradient avoiding line set")
        (p,max,maxsq) = (None,-1000,-1000)
        for y in range(self.h-self.min_y*4):
            for x in range(self.w-self.min_x*4):
                if self.line_set.is_closer_than(x+self.min_x*2,y+self.min_y*2): continue
                i = self.w*(y+self.min_y*2)+x+self.min_x*2
                vdx = self.vector_dx[i]
                vdx_sq = vdx*vdx
                if vdx_sq>maxsq:
                    d = vdx_sq + self.vector_dy[i]*self.vector_dy[i]
                    if d>maxsq:
                        max = math.sqrt(d)
                        maxsq = d
                        p = i
                        pass
                    pass
                pass
            pass
        x = p % self.w
        y = p / self.w
        self.info_msg("Greatest gradient avoiding lines at (%d,%d) of %4.2f"%(x,y,max))
        self.info_end()
        return (x,y)
    #f weighted_average
    def weighted_average( self, v, x, y ):
        if (x>=self.w-1): return 0
        if (y>=self.h-1): return 0
        fx = int(x)
        fy = int(y)
        dx = x-fx
        dy = y-fy
        d00p = (1-dx)*(1-dy)
        d01p =   (dx)*(1-dy)
        d10p = (1-dx)*(  dy)
        d11p = (  dx)*(  dy)
        d00 = v[fy*self.w+fx]
        d01 = v[fy*self.w+fx+1]
        d10 = v[(fy+1)*self.w+fx]
        d11 = v[(fy+1)*self.w+fx+1]
        return d00*d00p + d01*d01p + d10*d10p + d11*d11p
    #f gradient_at
    def gradient_at( self, x, y ):
        dx = self.weighted_average( self.vector_dx, x, y )
        dy = self.weighted_average( self.vector_dy, x, y )
        return (dx,dy)
    #f resolved_at
    def resolved_at( self, x, y ):
        d = self.weighted_average( self.resolved, x, y )
        return d
    #f find_centre_of_gradient
    def find_centre_of_gradient( self, x, y, scale=0.33, num_points=10, min_step=3.0 ):
        self.info_start("Find centre of gradient at (%4.2f,%4.2f)"%(x,y))
        (dx,dy) = self.gradient_at(x,y)
        l = math.sqrt(dx*dx+dy*dy)
        if (l<0.00001): return (x,y,self.resolved_at(x,y))
        ndx = dx/l
        ndy = dy/l
        gradients_along_line = []
        for i in range(1+2*num_points):
            p = (i-num_points)
            tx = x+p*ndx*scale
            ty = y+p*ndy*scale
            (gx,gy) = self.gradient_at(tx,ty)
            d = self.resolved_at(tx,ty)
            gradients_along_line.append( (tx,ty,gx,gy,d) )
            #print i,"(%4.2f,%4.2f):%4.2f:(%4.2f,%4.2f)"%(tx,ty,d,gx,gy)
            pass
        l = num_points-1
        r = num_points+1
        while ((l>0) and (gradients_along_line[l][4]-gradients_along_line[l-1][4])>min_step): l-=1
        while ((r<num_points*2) and (gradients_along_line[r+1][4]-gradients_along_line[r][4])>min_step): r+=1
        #print l,r
        #print gradients_along_line[l][4], gradients_along_line[r][4]
        contour_height = (gradients_along_line[l][4] + gradients_along_line[r][4])/2.0
        #print contour_height
        ls = gradients_along_line[l][4]
        for i in range(r-l+1):
            if contour_height>gradients_along_line[l+i][4]:
                ls = l+i
                pass
            pass
        dh = gradients_along_line[ls+1][4] - gradients_along_line[ls][4]
        dch = contour_height - gradients_along_line[ls][4]
        #print ls, dch/dh
        #print gradients_along_line[ls]
        #print gradients_along_line[ls+1]
        x = gradients_along_line[ls+1][0]*dch/dh + (1-dch/dh)*gradients_along_line[ls][0]
        y = gradients_along_line[ls+1][1]*dch/dh + (1-dch/dh)*gradients_along_line[ls][1]
        print (x,y)
        self.info_end()
        return (x,y,contour_height)
    #f find_where_value
    def find_where_value( self, v, x, y, max_diff=2.0 ):
        """
        Starting at (x,y), find where the resolved value is v
        If cannot find it, return (None,None)

        Note that hopefully at n% along a (unit) gradient direction the value should be r(x,y)+div*n%
        ?Is this true? Is this -50% to +50% for the gradient change?

        So if we are at value 'p' = r(x,y), and we want to be at 'v', we expect

        p+div*n% = v
        div*n% = v-p
        n% = (v-p)/div
        Here, div is the magnitude of the gradient
        The direction of the gradient is dx/div,dy/div
        And 'n%' along the gradient is (x,y) + n%*direction
        Hence we want to try (x + (v-p)/div * dx / div, y + (v-p)/div * dy / div )
        or x + dx*(v-p)/(div*div), y + dy*(v-p)/(div*div), 
        """
        p = self.resolved_at( x, y )
        (dx,dy) = self.gradient_at( x, y )
        div_sq = dx*dx+dy*dy
        if (div_sq<0.0001): return (None,None)
        nx = x + dx*(v-p)/div_sq
        ny = y + dy*(v-p)/div_sq
        np = self.resolved_at(nx,ny)
        print "Target: ",v,math.sqrt(div_sq),(v-p)/math.sqrt(div_sq)
        print "In:",x,y,p
        print "Guess:",nx,ny,np
        abs_pdiff = p-v
        if abs_pdiff<0: abs_pdiff=-abs_pdiff
        abs_npdiff = np-v
        if abs_npdiff<0: abs_npdiff=-abs_npdiff
        if abs_npdiff<abs_pdiff:
            abs_pdiff = abs_npdiff
            x=nx
            y=ny
        if abs_pdiff>max_diff: return (None,None)
        return (x,y)
    #f find_where_value
    def find_where_value( self, v, x, y, max_diff=0.5, max_divs=6, reduction=2.0 ):
        """
        Starting at (x,y), find where the resolved value is v
        If cannot find it, return (None,None)

        Use (x,y) as (x0,y0)
        Use (x,y) + direction(gradient) as (x1,y1)

        Get p0 = val(x0,y0), p1=val(x1,y1)
        If p0>p1 then swap 0/1
        [ At this point p0<=p1 ]
        If v<p0 then return None
        If v>p1 then return None
        > Find x2/y2/p2 = midpoint
        > If v<p2 then x1/y1/p1=x2/y2/p2
        > Else x0/y0/p0=x2/y2/p2
        > Repeat 'n' times
        If p2 more than max_diff from v then return None
        return x2/y2
        """
        (dx,dy) = self.gradient_at( x, y )
        l = math.sqrt(dx*dx+dy*dy)
        if (l<0.0001): return (None,None)
        l *= reduction
        (x0,y0)=(x-dx/l,y-dy/l)
        (x1,y1)=(x+dx/l,y+dy/l)
        p0 = self.resolved_at( x0, y0 )
        p1 = self.resolved_at( x1, y1 )
        #print "In:",v,x,y
        #print "p0:",p0,x0,y0
        #print "p1:",p1,x1,y1
        if (p0>p1):
            (x0,y0,p0,x1,y1,p1) = (x1,y1,p1,x0,y0,p0)
            pass
        if (v<p0):
            if (p0-v)<max_diff: return (x0,y0)
            return (None,None)
        if (v>p1):
            if (v-p1)<max_diff: return (x1,y1)
            return (None,None)
        for i in range(max_divs):
            (x,y)=( (x0+x1)/2, (y0+y1)/2 )
            p = self.resolved_at( x, y )
            if (v<p):
                (x1,y1,p1)=(x,y,p)
                pass
            else:
                (x0,y0,p0)=(x,y,p)
                pass
            pass
        (x,y)=( (x0+x1)/2, (y0+y1)/2 )
        p = self.resolved_at( x, y )
        abs_pdiff = p-v
        if abs_pdiff<0: abs_pdiff=-abs_pdiff
        if abs_pdiff>max_diff: return (None,None)
        #print "Out:",p,x,y,abs_pdiff
        return (x,y)
    #f find_next_point_along_contour
    def find_next_point_along_contour( self, v, cx, cy, scale=0.8, rot=1, max_divergence=0.1, max_steps=1000, min_segments_before_divergence=5, len_power=0.8 ):
        print "find_next_point_along_contour",v,cx,cy
        (cdx,cdy) = self.gradient_at(cx,cy)
        l = math.sqrt(cdx*cdx+cdy*cdy)
        cdx = cdx / l
        cdy = cdy / l
        tdx = -rot*scale*cdy
        tdy = +rot*scale*cdx

        (ncx,ncy) = self.find_where_value( v, cx+2*tdx, cy+2*tdy )
        if ncx is None:
            print "Bail due to not finding x,y on initial contour along gradient"
            return (None,None)
        dirx = (ncx-cx)
        diry = (ncy-cy)
        l = math.sqrt(dirx*dirx+diry*diry)
        dirx = dirx/l
        diry = diry/l
        divergence_errors = 0
        (dirx,diry) = (-diry,dirx)
        for i in range(max_steps):
            (nccx,nccy) = self.find_where_value( v, ncx+tdx, ncy+tdy )
            if nccx is None:
                print "Bail due to not finding x,y on contour along gradient"
                break
            ncdx = (nccx-cx)
            ncdy = (nccy-cy)
            l = math.pow( ncdx*ncdx+ncdy*ncdy, 0.5*len_power )
            divergence = (ncdx*dirx + ncdy*diry)/l
            #print "divergence",divergence,ncdx,ncdy,cdx,cdy
            # divergence = len(line segment) . cos( angle between gradient and line segment )
            # angle should be close to +-90 if there is little divergence from the gradient
            # cos should be close to 0
            # Note as the line segment gets longer we get more fussy
            # So we divide by len(line segment) ^ 0.8
            # l = ((len*len)^0.5)^0.8
            if divergence<0: divergence=-divergence
            if (divergence>max_divergence):
                divergence_errors += 1
                if (i>min_segments_before_divergence):
                    print "Bail due to too much curve",divergence_errors,i,divergence,max_divergence
                    break
                pass
            ncx = nccx
            ncy = nccy
            #print i,divergence,(ncx,ncy),self.resolved_at(ncx,ncy)
            pass
        if i==0: return (None,None)
        if (divergence_errors>=i): return (None,None)
        return (ncx,ncy)
    #f new_line
    def new_line( self, x, y, v ):
        l = c_line(self.line_set,x,y,v)
        self.line_set.add_line( l )
        return l
