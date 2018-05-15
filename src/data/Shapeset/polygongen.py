import math, copy, numpy

class Polygongen(object):
    """
    Generate formal representation of random polygons images with the iterator method
    
    outputs of iterator
    'rval_points' - a 4-d array (batchsize*nb_poly_max*nvert*2)  coordinates of the vertices of each polygon
    'rval_nbpol' - a 1-d array (batchsize) giving the number of polygon in each image
    'nb_poly_max - an int giving the max number of polygon per image
    'batchsize' - an int giving the number of generated image per iteration
    'rval_bg' - a 1-d array (batchsize) giving the background color of each image
    'rval_fg' - a 2-d array (batchsize*nb_poly_max) giving the background color of each polygon
    'rval_poly_id' - a 2-d array (batchsize*nb_poly_max) giving the identity of each polygon (fill with -1 where not
present
    'self.img_shape' - a 1-d array (2) giving the dimension of the image in pixel
    'self.n_vertices' - a 1-d array giving the number of vertices of each polygon identity
    
    """
    
    def __init__(self,
                img_shape, # 1d tuple (2) giving the image dimension
                n_vert_list, # 1-d array giving the number of vertices of each polygon identity
                #@warning: can't be changed during iterating (you must instanciate a new instance)
                #please less than 255 vertices
                poly_type = 2, # 0> (equilateral triangle, square, circle), 1> (isoceles triangle,rectangle,ellipse),
                # 2> (triangle,parallelogram,ellipse)
                nb_poly_min = 1, # minimum number of polygons in the image ( >= 1)
                #(if overlap rejection is True and rejectionmax != -1, you may obtain an image with less polygons)
                nb_poly_max = 3, # maximum number of polygons in the image (please <= 256 for the depthmap (pixel depth=8))
                bg_min = 0.0, # max background color
                bg_max = 0.0, # min background color
                fg_min = 1.0, # max polygon color
                fg_max = 1.0, # min polygon color
                inv_chance = 0.5, # chance to invert the fg and bg color
                rot_min = 0.0, # min rotation angle 0>0 rad
                rot_max = 0.0, # max rotation angle 1>2*pi rad
                rotation_resolution=256, # number of possible rotations between 0 and 2*Pi
                #@warning: can't be changed during iterating (you must instanciate a new instance)
                pos_min = 0.5, # min relative translation 0.5 is middle
                pos_max = 0.5, # max relative translation 0.5 is middle
                scale_min = 0.5, # min scale (the original scale is 0.5 * the image dimension)
                scale_max = 0.5, # max scale (the original scale is 0.5 * the image dimension)
                overlap_bool = True, # if false no overlap rejection (for too much polygons > too much overlap rejection so
                #should be put to false), if false objects may be hidden
                overlap_max = 1, # max overlap between 2 objects (0 to 1)
                #(at 0.5 2 objects may hide a third one)
                rejectionmax = 100, # max overlap rejection per draw (if we obtain this nb of rejection we don't try to put
                #another polygon and give the image with less polygons), if -1 > we continue rejection untill we find a
                #solution (@warning: may cause an infinite loop, @todo:think about a way to dodge that but still giving a
                #good number of polygons)
                ):
    
        # -------------- Parameter validation
        if len(img_shape) != 2:
            raise ValueError('shape must be seq of 2 ints', img_shape)
    
        def check_range(low, high, name, upper=1.0):
            if min(low, high) < 0.0 or \
                    max(low, high) > upper or \
                    low > high:
                raise ValueError('invalid range', (name, low, high))
    
        check_range(fg_min, fg_max, 'fg')
        check_range(bg_min, bg_max, 'bg')
        check_range(pos_min, pos_max, 'pos')
        check_range(rot_min, rot_max, 'rot')
        check_range(scale_min, scale_max, 'scale')
        check_range(0, overlap_max, 'overlap')
    
    
        if int(nb_poly_min)<=0 or int(nb_poly_min) > int(nb_poly_max) or int(nb_poly_max)>255:
            raise ValueError('nb_poly_min must be > 1 and < nb_poly_max < = 255 ', [nb_poly_min, nb_poly_max])
        if int(poly_type)<0 or int(poly_type)>2:
            raise ValueError('poly_type must be between 0 and 2 ', poly_type)
        # ------------------------------------
    
        # ----------------- Static function definition
        def circ_pos(t):
            """return the coordinates of the 2*pi*t angle point on the unit circle"""
            angle = t * math.pi * 2.0
            return [math.cos(angle), math.sin(angle)]
    
        def n_sided(n):
            """return n equally spaced points on circle of radius 0.5 with horizontal down edge"""
            return 0.5 * numpy.asarray([circ_pos((1.0*t)/n+1.0/(2*n)+1.0/4) for t in xrange(n)])
    
        def arearatio(n):
            """return the area ratio of the regular polygone with n vertice in comparison to the circle one"""
            return n*math.sin(2*math.pi/n)/(2*math.pi)
    
        def gen_rot(theta):
            """return the rotation matrix associated with angle theta around the origin"""
            c, s = math.cos(theta), math.sin(theta)
            return numpy.asarray(([c, s], [-s, c]))
    
    
        # -------------- Class attributes initialisation
        self.img_shape = numpy.asarray(img_shape)
        self.n_vertices = copy.copy(n_vert_list)  # not to give the pointer
        self.poly_type = int(poly_type)
        self.nb_poly_min = int(nb_poly_min)
        self.nb_poly_max = int(nb_poly_max)
    
        self.poly_points = [n_sided(n) for n in n_vert_list] # the regular points of each polygon identity
        self.aratio =[arearatio(n) for n in n_vert_list] # the area ratio of each polygon identity
    
        self.bg_min = bg_min
        self.bg_max = bg_max
        self.fg_min = fg_min
        self.fg_max = fg_max
        self.inv_chance = inv_chance
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rot_max = rot_max
        self.rot_min = rot_min
        self.rotation_resolution = rotation_resolution
    
        # pre-calculate the rotation matrices
        rot_step = 2 * math.pi / rotation_resolution
        self.rot_rads = [t * rot_step for t in xrange(rotation_resolution)]
        self.rot_mats = [gen_rot(rot_rad) for rot_rad in self.rot_rads]
    
        self.overlap_bool = overlap_bool
        self.overlap_max = overlap_max
        self.rejectionmax = rejectionmax
    
        self.n_reject = 0 # total number of rejections on polygon sampling
        self.n_iteration = 0 # total number of iterations of polygon sampling (rejections not counted)
    
    # method to radomly create a new polygon of type shape_id in an image (by taking in account overlaps with previous
    # objects)
    def _corners(self,rng, shape_id, rval_pos=[], rval_scale=[], rval_poly_id=[], rval_points=[], nbpol=0, nbrejection=0):
    
        self.n_iteration += 1
    
        img_shape = self.img_shape
    
        points_orig = self.poly_points[shape_id] # the original vertices of the regular polygon (around the 0.5 circle)
        poly_type = self.poly_type
    
        scale_min = self.scale_min
        scale_max = self.scale_max
        pos_min = self.pos_min
        pos_max = self.pos_max
    
        rotation_resolution = self.rotation_resolution
    
        rejectionmax = self.rejectionmax
    
        if poly_type < 2: # if we want regular polygon or type 1 no previous rotation
            rot_idx1 = 0 # no rotation will keep the initial orientation of the polygone before scaling
            # (rectangle, isocele triangle...)
        else:
            rot_idx1 = rng.randint(0, rotation_resolution/(len(points_orig)))
            if rot_idx1 > (rotation_resolution-1)/(2*len(points_orig)):
                rot_idx1 = min(rotation_resolution-(rotation_resolution-1)/(len(points_orig))+rot_idx1,rotation_resolution-1)
            # this rotation permits to give a random polygon (no regular)
            # we constrained the rotation because we have symetry and we want the base to be the horizontal down edge
    
    
        rotmin = min(int(self.rot_min*(rotation_resolution-1)), rotation_resolution-1)
        rotmax = min(int(self.rot_max*(rotation_resolution-1)), rotation_resolution-1)
        if rotmin >= rotmax:
            rot_idx3 = rotmin
        else:
            rot_idx3 = rng.randint(rotmin, rotmax)
        # real random rotation between rotmin and rotmax
    
        # take the rotation matrix
        r1 = self.rot_mats[rot_idx1]
        r3 = self.rot_mats[rot_idx3]
    
        while True: # for the rejection
    
            if poly_type < 1: # if we want regular polygon the deformation should be the same on both axis
                a = rng.uniform(low=scale_min, high=scale_max, size=1) * img_shape
                s = numpy.asarray([a[0],a[0]])
            else:
                s = rng.uniform(low=scale_min, high=scale_max, size=2) * img_shape
                if s[1] > s[0]: # to keep the base in the horizontal direction s[0] must be bigger
                    tmp = s[0]
                    s[0] = s[1]
                    s[1] = tmp
    
            if rot_idx1 == 0: # poly_type 0
                r2 = self.rot_mats[0]
            elif s[0] == s[1]: # poly_type 1
                r2 = self.rot_mats[rotation_resolution-rot_idx1]
            else: # general case
                # after the scaling we need to come back to 0 rad rotation, we need to calculate the exact inverse
                # rotation according to s and rot_idx1 because it might not be in the rotation samples (and
                # the base won't be exactly horizontal)
                rot2 = -math.atan(s[1]/s[0]*math.tan(self.rot_rads[rot_idx1]))
                co, si = math.cos(rot2), math.sin(rot2)
                r2 = numpy.asarray(([co, si], [-si, co]))
            # coordinate of the vertices of the random polygon
            points = numpy.dot(numpy.dot(numpy.dot(points_orig, r1) * s, r2), r3) # the order is important
    
            # find the max-min box of the vertices position
            maxpts = (points.max(0)+1)/img_shape # +1 because the shapes shouldn't touch the border of the image
            minpts = (points.min(0)-1)/img_shape # -1 because the shapes shouldn't touch the border of the image
    
            # sample a translation and keep the entire object in the image
            t = (numpy.asarray([rng.uniform(low=max(pos_min,-minpts[0]),high=min(pos_max,1-maxpts[0])),\
            rng.uniform(low=max(pos_min,-minpts[1]),high=min(pos_max,1-maxpts[1]))])).reshape(2) * img_shape
    
            # apply the translation
            points = points+t
    
            # overlap rejection code
            if self.overlap_bool:
    
                rejectionbool=False # when there is too much rejection done it will be set to true
                i = 0 # to scan all the already existing polygons of the image
                while (not rejectionbool) and i<nbpol :
    
                    vectc = t-rval_pos[i] # vector from the center of the ith polygon to the new one
                    distancec = math.sqrt(pow(vectc[0],2) + pow(vectc[1],2)) # its length
    
                    # projection of the vertices on vectc for the old ith polygon and -vectc for the new one
                    proj1 = numpy.dot(rval_points[i]-rval_pos[i],vectc) / distancec
                    proj2 = numpy.dot(points-t,-vectc) / distancec
    
                    proj = (proj1.max()+proj2.max()) # take the max value on both
                    distb = proj >= distancec # if the sum is > distancec there may be an overlap
    
                    if distb and (nbrejection <=rejectionmax or rejectionmax == -1 ): # if overlap and rejection
                        overlapfactor = 1 - math.exp(-(proj-distancec)/(distancec/math.log(2)))
    
                        # boolean to say when the estimated overlap is too big
                        overlapbool = s[0]*s[1]*self.aratio[shape_id] * overlapfactor / \
                            (rval_scale[i,0]*rval_scale[i,1]*self.aratio[rval_poly_id[i]]) > self.overlap_max
    
                        if overlapbool:
                            nbrejection += 1 # keep track of the current number of rejection
                            self.n_reject += 1 # keep track of the total number of rejection
                            rejectionbool = True # we do a rejection
                    i += 1 # go to the next polygon
    
                if rejectionbool:
                    continue #if rejection > resample s and t
    
            break #if we are here > polygon is good we can return it
        return points, rot_idx1, rot_idx3 , s, t, nbrejection
    
    
    def iterator(self, batchsize, seed = 0):
    
        nb_poly_max = self.nb_poly_max
        # initialisation of the return vector @warning: this is done only at the iterator initialisation
        rval_nvert = numpy.ndarray((batchsize, nb_poly_max), dtype='uint8')
        rval_pos = numpy.ndarray((batchsize, nb_poly_max, 2), dtype='int32')
        rval_rot1 = numpy.ndarray((batchsize, nb_poly_max,), dtype='uint8')
        rval_rot3 = numpy.ndarray((batchsize, nb_poly_max,), dtype='uint8')
        rval_scale = numpy.ndarray((batchsize, nb_poly_max,2), dtype='float64')
        rval_seed = numpy.ndarray((batchsize, nb_poly_max,), dtype='int32')
        rval_nbpol = numpy.ndarray(batchsize, dtype='int8')
        rval_points = [None]*batchsize*nb_poly_max
        rval_fg = numpy.ndarray((batchsize, nb_poly_max), dtype='uint8')
        rval_bg = numpy.ndarray(batchsize, dtype='uint8')
    
        rng = numpy.random.RandomState(seed)
    
        while True:
            rejectionmax = self.rejectionmax
    
            # here we init the return value that need to be initialize at all iteration
            rval_poly_id = -1 * numpy.ones((batchsize, nb_poly_max,),dtype='int8')
    
            if nb_poly_max != self.nb_poly_max: #if the size change, reinitialise output vector
                nb_poly_max = min(self.nb_poly_max,255)
                self.nb_poly_min = min(nb_poly_max,self.nb_poly_min)
                rval_nvert = numpy.ndarray((batchsize, nb_poly_max), dtype='uint8')
                rval_pos = numpy.ndarray((batchsize, nb_poly_max, 2), dtype='int32')
                rval_rot1 = numpy.ndarray((batchsize, nb_poly_max,), dtype='uint8')
                rval_rot3 = numpy.ndarray((batchsize, nb_poly_max,), dtype='uint8')
                rval_scale = numpy.ndarray((batchsize, nb_poly_max,2), dtype='float64')
                rval_seed = numpy.ndarray((batchsize, nb_poly_max,), dtype='int32')
                rval_nbpol = numpy.ndarray(batchsize, dtype='int8')
                rval_points = [None]*batchsize*nb_poly_max
                rval_fg = numpy.ndarray((batchsize, nb_poly_max), dtype='uint8')
                rval_bg = numpy.ndarray(batchsize, dtype='uint8')
    
            for j in xrange(batchsize): # for all the batch
                nbrejection = 0
    
                fg_min, bg_min = self.bg_min, self.fg_min
                fg_max, bg_max = self.bg_max, self.fg_max
                if rng.rand() > self.inv_chance: #(invert the colors)
                    fg_min, bg_min = bg_min, fg_min
                    fg_max, bg_max = bg_max, fg_max
    
                # pick a number of polygons
                nbpol = int(rng.randint(self.nb_poly_min, 1 + nb_poly_max))
                # pick a background color
                bg = int(rng.randint(int(bg_min * 255), 1 + int(bg_max * 255)))
    
                i = 0
                while i < nbpol: # for each polygon of the image
                    # pick a polygon type
                    poly_id = rng.randint(0, len(self.poly_points))
    
                    nvert = self.n_vertices[poly_id]
    
                    # create the vertices and save the parameters of generation (r1,r2,s,t)
                    points, r_rad1, r_rad3, s, t, nbrejection = \
                        self._corners(rng, poly_id, rval_pos[j], rval_scale[j],rval_poly_id[j],\
                        rval_points[nb_poly_max*j:nb_poly_max*j+i],i, nbrejection)
    
                    # if we made too much rejection we just leave the loop without have made all the polygons
                    if (nbrejection >= rejectionmax) and (not (rejectionmax == -1) ):
                        print 'Too many rejections : stop polygon sampling'
                        nbpol = i
                    else: # else save the sampled polygon in the output list
                        fg = int(rng.randint(int(fg_min * 255), 1 + int(fg_max * 255)))
                        rval_nvert[j,i] = nvert
                        rval_points[nb_poly_max*j+i] = points
                        rval_fg[j,i] = fg
                        rval_pos[j,i] = t
                        rval_rot1[j,i] = r_rad1
                        rval_rot3[j,i] = r_rad3
                        rval_scale[j,i] = s
                        rval_seed[j,i] = seed
                        rval_poly_id[j,i] = poly_id
                        i = i+1
                rval_bg[j] = bg
                rval_nbpol[j] = nbpol
            yield {'rval_points':rval_points, 'rval_nbpol':rval_nbpol, 'nb_poly_max':nb_poly_max, 'batchsize':batchsize,\
                    'rval_bg':rval_bg, 'rval_fg':rval_fg, 'rval_poly_id':rval_poly_id, 'img_shape':self.img_shape,\
                    'n_vertices':self.n_vertices}
